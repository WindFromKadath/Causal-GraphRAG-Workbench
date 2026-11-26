"""Stage 4.3 Thin LangGraph Stateful Cyclic Orchestrator (Zero Legacy LangChain Chains).

Wraps the pure-Python Stage 3 GraphRAG/CoT/ToT/HyperGraph/A-MEM kernel inside a clean,
inspectable `langgraph.graph.StateGraph` supporting:
1. Pre-Filter Tenant & ACL Guard (`acl_guard`)
2. Multi-Query Rewrite + Self-Query Filter Extraction (`query_rewrite`)
3. Coarse Hybrid Recall Top-40 -> Local ONNX/CPU Rerank Top-8 (`recall_and_rerank`)
4. Corrective RAG Conditional Retry Loop (`expand_query -> recall_and_rerank`) when evidence is weak
5. Stage 3 Core Reasoning (`graph_reason`: PPR/Hybrid + Adaptive Router + CoT/ToT + GroundingAudit)
6. Grounding Self-Healing Retry (`self_heal_grounding -> graph_reason`) when grounding fails
7. Human-in-the-Loop (HITL) A-MEM Write-Back Approval via native `interrupt()` + `PostgresSaver` / `MemorySaver`
8. Node-level SSE Event Streaming (`stream_events`) + Full PG/OpenTelemetry Trace Recording (`record_trace`).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.schema import TextChunk
from llm_graphrag_kg.eval.llm_judge import detect_refusal
from llm_graphrag_kg.observability.tracing import RequestTraceRecord, TraceRecorder, WaterfallStep
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.retrieval.query_rewrite import (
    expand_query_for_retry,
    fuse_multi_query_hits,
    rewrite_query,
)
from llm_graphrag_kg.retrieval.reranker import CrossEncoderReranker, create_reranker


class GraphRAGAgentState(TypedDict, total=False):
    request_id: str
    thread_id: str
    tenant_id: str
    acl_roles: list[str]
    question: str
    mode: str
    reasoning: str
    enable_writeback: bool
    require_hitl_approval: bool
    use_llm_rewrite: bool

    # Funnel state
    rewritten_queries: list[str]
    self_query_filters: dict[str, Any]
    coarse_hits: list[dict[str, Any]]
    reranked_hits: list[dict[str, Any]]
    reranker_backend: str
    reranker_latency_ms: float
    evidence_score: float
    retry_count: int
    max_retries: int

    # Reasoning & grounding state
    reasoning_question: str
    answer: str
    route_decision: dict[str, Any]
    grounding_score: float | None
    ungrounded_citations: list[str]
    confidence: float
    grounding_retry_count: int
    max_grounding_retries: int
    query_result_payload: dict[str, Any]

    # A-MEM & HITL state
    pending_memory_candidate: dict[str, Any] | None
    human_approval: dict[str, Any] | None
    writeback_decision: dict[str, Any] | None

    # Observability
    terminal_status: str
    waterfall: list[dict[str, Any]]
    started_at: float


class AgentRunOutput(BaseModel):
    request_id: str
    thread_id: str
    tenant_id: str
    acl_roles: list[str]
    question: str
    status: str  # completed | interrupted_hitl | error
    answer: str = ""
    rewritten_queries: list[str] = Field(default_factory=list)
    reranker_backend: str = ""
    reranked_hits: list[dict[str, Any]] = Field(default_factory=list)
    evidence_score: float = 0.0
    retry_count: int = 0
    grounding_retry_count: int = 0
    grounding_score: float | None = None
    route_decision: dict[str, Any] = Field(default_factory=dict)
    interrupt_payload: dict[str, Any] | None = None
    writeback_decision: dict[str, Any] | None = None
    waterfall: list[WaterfallStep] = Field(default_factory=list)
    total_ms: float = 0.0
    query_result: dict[str, Any] = Field(default_factory=dict)


class GraphRAGAgentRunner:
    """Thin LangGraph cyclic state machine orchestrating Stage 3 GraphRAG + Stage 4.3 Retrieval & HITL."""

    def __init__(
        self,
        pipeline: GraphRAGPipeline,
        *,
        reranker: CrossEncoderReranker | None = None,
        tracer: TraceRecorder | None = None,
        checkpointer_mode: str | None = None,  # memory | postgres
        min_evidence_score: float = 0.25,
        coarse_top_k: int = 40,
        rerank_top_n: int = 8,
        max_retrieval_retries: int = 1,
        max_grounding_retries: int = 1,
    ) -> None:
        self.pipeline = pipeline
        cfg = pipeline.config
        self.reranker = reranker or create_reranker(
            backend=cfg.env.reranker_backend,
            model_name=cfg.env.reranker_model,
            onnx_file=cfg.env.reranker_onnx_file,
            model_dir=cfg.env.reranker_model_dir,
            max_length=cfg.env.reranker_max_length,
            redis_url=cfg.env.redis_url if cfg.env.vector_backend == "pgvector" else None,
        )
        self.tracer = tracer or TraceRecorder(
            service_name=cfg.env.otel_service_name,
            postgres_dsn=cfg.env.postgres_dsn if cfg.env.vector_backend == "pgvector" else None,
            persist_to_pg=(cfg.env.vector_backend == "pgvector"),
            otlp_endpoint=cfg.env.otel_exporter_otlp_endpoint,
        )
        self.checkpointer_mode = (checkpointer_mode or cfg.env.langgraph_checkpointer).strip().lower()
        self.min_evidence_score = float(min_evidence_score)
        self.coarse_top_k = max(8, int(coarse_top_k))
        self.rerank_top_n = max(2, int(rerank_top_n))
        self.max_retrieval_retries = max(0, int(max_retrieval_retries))
        self.max_grounding_retries = max(0, int(max_grounding_retries))

        self._exit_stack = ExitStack()
        self._checkpointer = self._create_checkpointer()
        self._graph = self._compile_graph()

    def _create_checkpointer(self) -> Any:
        if self.checkpointer_mode == "postgres":
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import-not-found]

            cm = PostgresSaver.from_conn_string(self.pipeline.config.env.postgres_dsn)
            saver = self._exit_stack.enter_context(cm)
            saver.setup()
            return saver
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import-not-found]

        return MemorySaver()

    def close(self) -> None:
        self._exit_stack.close()

    def _append_step(
        self,
        state: GraphRAGAgentState,
        node: str,
        t0: float,
        summary: dict[str, Any],
        status: str = "ok",
    ) -> list[dict[str, Any]]:
        wf = list(state.get("waterfall") or [])
        wf.append(
            WaterfallStep(
                node=node,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                status=status,
                summary=summary,
            ).model_dump()
        )
        return wf

    def _coarse_recall_for_query(
        self,
        q_text: str,
        *,
        tenant_id: str,
        acl_roles: list[str],
    ) -> list[tuple[TextChunk, float]]:
        q_vec = self.pipeline.embedder.embed([q_text])[0]
        vs = self.pipeline.vector_store
        if hasattr(vs, "search_hybrid"):
            try:
                return vs.search_hybrid(
                    q_vec,
                    q_text,
                    top_k=self.coarse_top_k,
                    tenant_id=tenant_id,
                    acl_roles=acl_roles,
                )
            except TypeError:
                return vs.search_hybrid(q_vec, q_text, top_k=self.coarse_top_k)
        return vs.search(q_vec, top_k=self.coarse_top_k)

    def _compile_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
        from langgraph.types import interrupt  # type: ignore[import-not-found]

        # ---- Node 1: Tenant & ACL Pre-Filter Guard ----
        def acl_guard(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            tid = (state.get("tenant_id") or self.pipeline.config.env.default_tenant_id).strip() or "default"
            raw_roles = state.get("acl_roles") or [
                r.strip() for r in self.pipeline.config.env.default_acl_roles.split(",") if r.strip()
            ]
            roles = [str(r).strip() for r in raw_roles if str(r).strip()] or ["public"]
            # Propagate active tenant/ACL context into enterprise stores when present
            if hasattr(self.pipeline.vector_store, "default_tenant_id"):
                self.pipeline.vector_store.default_tenant_id = tid
                self.pipeline.vector_store.default_acl_roles = roles
            if hasattr(self.pipeline.graph_store, "default_tenant_id"):
                self.pipeline.graph_store.default_tenant_id = tid
                self.pipeline.graph_store.default_acl_roles = roles
            if hasattr(self.pipeline.memory_store, "default_tenant_id"):
                self.pipeline.memory_store.default_tenant_id = tid
                self.pipeline.memory_store.default_acl_roles = roles
            return {
                "tenant_id": tid,
                "acl_roles": roles,
                "reasoning_question": state["question"],
                "waterfall": self._append_step(state, "acl_guard", t0, {"tenant_id": tid, "acl_roles": roles}),
            }

        # ---- Node 2: Query Rewrite + Multi-Query + Self-Query ----
        def query_rewrite_node(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            rw = rewrite_query(
                state["question"],
                llm=self.pipeline.llm if state.get("use_llm_rewrite") else None,
                use_llm=bool(state.get("use_llm_rewrite", False)),
            )
            return {
                "rewritten_queries": rw.rewritten_queries,
                "self_query_filters": rw.self_query_filters.model_dump(),
                "waterfall": self._append_step(
                    state,
                    "query_rewrite",
                    t0,
                    {"strategy": rw.strategy, "query_count": len(rw.rewritten_queries)},
                ),
            }

        # ---- Node 3: Multi-Query Coarse Recall Top-40 -> ONNX CPU Rerank Top-8 ----
        def recall_and_rerank_node(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            queries = list(state.get("rewritten_queries") or [state["question"]])
            tid = state["tenant_id"]
            roles = state["acl_roles"]

            per_q_hits = [
                self._coarse_recall_for_query(q, tenant_id=tid, acl_roles=roles)
                for q in queries[:4]
            ]
            fused_coarse = fuse_multi_query_hits(per_q_hits, top_k=self.coarse_top_k)
            rr = self.reranker.rerank(
                state["question"],
                fused_coarse,
                top_n=self.rerank_top_n,
                tenant_id=tid,
            )
            top_score = rr.candidates[0].rerank_score if rr.candidates else 0.0
            coarse_payload = [
                {"chunk_id": ch.chunk_id, "score": sc} for ch, sc in fused_coarse[:12]
            ]
            reranked_payload = [
                {
                    "chunk_id": c.chunk.chunk_id,
                    "doc_id": c.chunk.doc_id,
                    "coarse_rank": c.coarse_rank,
                    "final_rank": c.final_rank,
                    "rerank_score": c.rerank_score,
                    "snippet": c.chunk.text[:160],
                }
                for c in rr.candidates
            ]
            return {
                "coarse_hits": coarse_payload,
                "reranked_hits": reranked_payload,
                "reranker_backend": rr.backend_used,
                "reranker_latency_ms": rr.latency_ms,
                "evidence_score": top_score,
                "waterfall": self._append_step(
                    state,
                    "recall_and_rerank",
                    t0,
                    {
                        "coarse_count": len(fused_coarse),
                        "reranked_count": len(rr.candidates),
                        "backend": rr.backend_used,
                        "top_score": top_score,
                        "cache_hit": rr.cache_hit,
                    },
                ),
            }

        # ---- Node 4: Corrective RAG Query Expansion Loop ----
        def expand_query_node(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            next_retry = int(state.get("retry_count", 0)) + 1
            expanded = expand_query_for_retry(
                state["question"],
                list(state.get("rewritten_queries") or [state["question"]]),
                retry_index=next_retry,
            )
            return {
                "retry_count": next_retry,
                "rewritten_queries": expanded,
                "waterfall": self._append_step(
                    state,
                    "expand_query",
                    t0,
                    {"retry_count": next_retry, "expanded_queries": expanded},
                ),
            }

        def check_evidence(state: GraphRAGAgentState) -> str:
            ev_score = float(state.get("evidence_score", 0.0))
            retries = int(state.get("retry_count", 0))
            max_r = int(state.get("max_retries", self.max_retrieval_retries))
            if (not state.get("reranked_hits") or ev_score < self.min_evidence_score) and retries < max_r:
                return "retry_retrieval"
            return "proceed_reason"

        # ---- Node 5: Stage 3 Core GraphRAG Reasoning Kernel ----
        def graph_reason_node(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            q_for_reason = state.get("reasoning_question") or state["question"]
            # Execute pure-Python Stage 3 kernel (disable auto writeback so LangGraph HITL controls it)
            res = self.pipeline.query(
                q_for_reason,
                mode=state.get("mode") or self.pipeline.config.retrieval.default_mode,
                reasoning=state.get("reasoning") or self.pipeline.config.reasoning.default_paradigm,
                enable_writeback=False,
                auto_load=False,
            )
            rm = res.metadata.get("reasoning")
            g_score: float | None = None
            ungrounded: list[str] = []
            conf = 0.0
            if isinstance(rm, dict):
                ga = rm.get("grounding_audit")
                if isinstance(ga, dict):
                    g_score = float(ga.get("grounding_score", 0.0))
                    ungrounded = list(ga.get("ungrounded_citations", []))
                conf = float(rm.get("confidence", 0.0))
            route_dec = res.metadata.get("route_decision")
            route_dict = dict(route_dec) if isinstance(route_dec, dict) else {}

            return {
                "answer": res.answer,
                "route_decision": route_dict,
                "grounding_score": g_score,
                "ungrounded_citations": ungrounded,
                "confidence": conf,
                "query_result_payload": res.model_dump(),
                "waterfall": self._append_step(
                    state,
                    "graph_reason",
                    t0,
                    {
                        "mode": res.mode,
                        "paradigm": res.metadata.get("reasoning_paradigm", "direct"),
                        "grounding_score": g_score,
                        "ungrounded_count": len(ungrounded),
                    },
                ),
            }

        # ---- Node 6: Grounding Self-Healing Retry ----
        def self_heal_grounding_node(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            next_g = int(state.get("grounding_retry_count", 0)) + 1
            bad_cites = state.get("ungrounded_citations") or []
            healed_q = (
                f"{state['question']}\n"
                f"[GROUNDING_SELF_HEAL: 严格仅引用检索到的实体与切片 ID，禁止编造引用 {bad_cites[:4]}]"
            )
            return {
                "grounding_retry_count": next_g,
                "reasoning_question": healed_q,
                "waterfall": self._append_step(
                    state,
                    "self_heal_grounding",
                    t0,
                    {"grounding_retry_count": next_g, "ungrounded": bad_cites[:4]},
                ),
            }

        def check_grounding(state: GraphRAGAgentState) -> str:
            g_score = state.get("grounding_score")
            ungrounded = state.get("ungrounded_citations") or []
            g_retries = int(state.get("grounding_retry_count", 0))
            max_g = int(state.get("max_grounding_retries", self.max_grounding_retries))
            if g_score is not None and (g_score < 1.0 or len(ungrounded) > 0) and g_retries < max_g:
                return "self_heal"
            return "proceed_writeback"

        # ---- Node 7: A-MEM Write-Back Gate + Optional HITL interrupt() ----
        def writeback_gate_node(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            if not state.get("enable_writeback"):
                return {
                    "pending_memory_candidate": None,
                    "waterfall": self._append_step(state, "writeback_gate", t0, {"enabled": False}),
                }
            payload = state.get("query_result_payload") or {}
            trace_d = payload.get("trace") or {}
            evidences = payload.get("evidences") or []
            rm = (payload.get("metadata") or {}).get("reasoning") or {}
            target_eids: list[str] = []
            cited_chunks: list[str] = []
            if isinstance(rm, dict):
                for br in rm.get("branches", []):
                    if isinstance(br, dict) and br.get("status") == "selected" and br.get("target_entity_id"):
                        target_eids.append(str(br["target_entity_id"]))
                for st in rm.get("steps", []):
                    if isinstance(st, dict):
                        cited_chunks.extend(str(c) for c in st.get("cited_chunk_ids", []))
            if not cited_chunks:
                cited_chunks = [str(ev.get("chunk_id")) for ev in evidences[:4] if isinstance(ev, dict)]

            candidate = {
                "question": state["question"],
                "answer": state.get("answer", ""),
                "grounding_score": state.get("grounding_score"),
                "ungrounded_citations": list(state.get("ungrounded_citations") or []),
                "confidence": float(state.get("confidence", 0.0)),
                "condition_entity_ids": list(trace_d.get("seed_entity_ids") or [])[:5],
                "target_entity_ids": target_eids[:3],
                "source_chunk_ids": cited_chunks,
                "reasoning_paradigm": str((payload.get("metadata") or {}).get("reasoning_paradigm", "auto")),
            }
            return {
                "pending_memory_candidate": candidate,
                "waterfall": self._append_step(
                    state,
                    "writeback_gate",
                    t0,
                    {"enabled": True, "require_hitl": bool(state.get("require_hitl_approval"))},
                ),
            }

        def route_hitl(state: GraphRAGAgentState) -> str:
            if state.get("pending_memory_candidate") and state.get("require_hitl_approval"):
                return "hitl_approval"
            return "commit_memory"

        def hitl_memory_approval_node(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            candidate = state.get("pending_memory_candidate") or {}
            # Native LangGraph HITL interrupt(): persists full graph checkpoint & waits for Command(resume=...)
            decision = interrupt(
                {
                    "type": "amem_writeback_approval",
                    "tenant_id": state["tenant_id"],
                    "question": candidate.get("question", ""),
                    "answer_preview": str(candidate.get("answer", ""))[:300],
                    "grounding_score": candidate.get("grounding_score"),
                    "confidence": candidate.get("confidence"),
                    "source_chunk_ids": candidate.get("source_chunk_ids", []),
                }
            )
            approval = decision if isinstance(decision, dict) else {"approved": bool(decision)}
            return {
                "human_approval": approval,
                "waterfall": self._append_step(
                    state,
                    "hitl_memory_approval",
                    t0,
                    {"approved": bool(approval.get("approved", False)), "reviewer": approval.get("reviewer", "human")},
                ),
            }

        def commit_memory_node(state: GraphRAGAgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            candidate = state.get("pending_memory_candidate")
            if not candidate:
                return {
                    "writeback_decision": None,
                    "waterfall": self._append_step(state, "commit_memory", t0, {"skipped": True}),
                }
            if state.get("require_hitl_approval"):
                approval = state.get("human_approval") or {}
                if not approval.get("approved", False):
                    dec = {
                        "accepted": False,
                        "gate_failed": "GATE_HUMAN_HITL_REJECTED",
                        "reason": f"Rejected by reviewer `{approval.get('reviewer', 'human')}`.",
                    }
                    return {
                        "writeback_decision": dec,
                        "waterfall": self._append_step(
                            state, "commit_memory", t0, {"accepted": False, "hitl": "rejected"}
                        ),
                    }

            mem_cfg = self.pipeline.config.memory
            wb = self.pipeline.memory_store.evaluate_and_write_back(
                question=candidate["question"],
                answer=candidate["answer"],
                grounding_score=candidate["grounding_score"],
                ungrounded_citations=candidate["ungrounded_citations"],
                confidence=candidate["confidence"],
                condition_entity_ids=candidate["condition_entity_ids"],
                target_entity_ids=candidate["target_entity_ids"],
                source_chunk_ids=candidate["source_chunk_ids"],
                reasoning_paradigm=candidate["reasoning_paradigm"],
                min_grounding=mem_cfg.min_writeback_grounding,
                min_confidence=mem_cfg.min_writeback_confidence,
                enable_conflict_check=mem_cfg.enable_conflict_check,
                promote_to_hyperedge=mem_cfg.promote_memory_to_hyperedge,
                graph_store=self.pipeline.graph_store,
            )
            dec_dict = wb.model_dump()
            return {
                "writeback_decision": dec_dict,
                "waterfall": self._append_step(
                    state,
                    "commit_memory",
                    t0,
                    {"accepted": wb.accepted, "gate_failed": wb.gate_failed},
                ),
            }

        # ---- Node 8: Record PG + OpenTelemetry Trace ----
        def record_trace_node(state: GraphRAGAgentState) -> dict[str, Any]:
            ans = state.get("answer", "")
            term_status = "business_refusal" if detect_refusal(ans) else "success"
            total_ms = round((time.perf_counter() - float(state.get("started_at", time.perf_counter()))) * 1000, 2)
            wf_steps = [WaterfallStep(**w) for w in (state.get("waterfall") or []) if isinstance(w, dict)]
            trace_rec = RequestTraceRecord(
                request_id=state["request_id"],
                thread_id=state["thread_id"],
                tenant_id=state["tenant_id"],
                acl=list(state.get("acl_roles") or []),
                question=state["question"],
                rewritten_queries=list(state.get("rewritten_queries") or []),
                route_decision=dict(state.get("route_decision") or {}),
                coarse_hits=list(state.get("coarse_hits") or []),
                reranked_hits=list(state.get("reranked_hits") or []),
                grounding_score=state.get("grounding_score"),
                terminal_status=term_status,
                waterfall=wf_steps,
                total_ms=total_ms,
                model_versions={
                    "llm": self.pipeline.config.env.llm_model,
                    "embedding": self.pipeline.config.env.embedding_model,
                    "reranker": f"{state.get('reranker_backend', 'auto')}:{self.reranker.model_name}",
                },
            )
            self.tracer.record(trace_rec)
            return {"terminal_status": term_status}

        builder = StateGraph(GraphRAGAgentState)
        builder.add_node("acl_guard", acl_guard)
        builder.add_node("query_rewrite", query_rewrite_node)
        builder.add_node("recall_and_rerank", recall_and_rerank_node)
        builder.add_node("expand_query", expand_query_node)
        builder.add_node("graph_reason", graph_reason_node)
        builder.add_node("self_heal_grounding", self_heal_grounding_node)
        builder.add_node("writeback_gate", writeback_gate_node)
        builder.add_node("hitl_memory_approval", hitl_memory_approval_node)
        builder.add_node("commit_memory", commit_memory_node)
        builder.add_node("record_trace", record_trace_node)

        builder.add_edge(START, "acl_guard")
        builder.add_edge("acl_guard", "query_rewrite")
        builder.add_edge("query_rewrite", "recall_and_rerank")
        builder.add_conditional_edges(
            "recall_and_rerank",
            check_evidence,
            {"retry_retrieval": "expand_query", "proceed_reason": "graph_reason"},
        )
        builder.add_edge("expand_query", "recall_and_rerank")
        builder.add_conditional_edges(
            "graph_reason",
            check_grounding,
            {"self_heal": "self_heal_grounding", "proceed_writeback": "writeback_gate"},
        )
        builder.add_edge("self_heal_grounding", "graph_reason")
        builder.add_conditional_edges(
            "writeback_gate",
            route_hitl,
            {"hitl_approval": "hitl_memory_approval", "commit_memory": "commit_memory"},
        )
        builder.add_edge("hitl_memory_approval", "commit_memory")
        builder.add_edge("commit_memory", "record_trace")
        builder.add_edge("record_trace", END)

        return builder.compile(checkpointer=self._checkpointer)

    def _build_output(self, state: dict[str, Any], *, interrupted: dict[str, Any] | None = None) -> AgentRunOutput:
        wf = [WaterfallStep(**w) for w in (state.get("waterfall") or []) if isinstance(w, dict)]
        total_ms = round((time.perf_counter() - float(state.get("started_at", time.perf_counter()))) * 1000, 2)
        return AgentRunOutput(
            request_id=str(state.get("request_id", "")),
            thread_id=str(state.get("thread_id", "")),
            tenant_id=str(state.get("tenant_id", "default")),
            acl_roles=list(state.get("acl_roles") or []),
            question=str(state.get("question", "")),
            status="interrupted_hitl" if interrupted is not None else "completed",
            answer=str(state.get("answer", "")),
            rewritten_queries=list(state.get("rewritten_queries") or []),
            reranker_backend=str(state.get("reranker_backend", "")),
            reranked_hits=list(state.get("reranked_hits") or []),
            evidence_score=float(state.get("evidence_score", 0.0)),
            retry_count=int(state.get("retry_count", 0)),
            grounding_retry_count=int(state.get("grounding_retry_count", 0)),
            grounding_score=state.get("grounding_score"),
            route_decision=dict(state.get("route_decision") or {}),
            interrupt_payload=interrupted,
            writeback_decision=state.get("writeback_decision"),
            waterfall=wf,
            total_ms=total_ms,
            query_result=dict(state.get("query_result_payload") or {}),
        )

    def run(
        self,
        question: str,
        *,
        thread_id: str | None = None,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
        mode: str = "ppr",
        reasoning: str = "auto",
        enable_writeback: bool = False,
        require_hitl_approval: bool = False,
        use_llm_rewrite: bool = False,
    ) -> AgentRunOutput:
        """Execute the full cyclic LangGraph state machine (may pause at HITL interrupt)."""
        tid = thread_id or f"thread_{uuid.uuid4().hex[:10]}"
        req_id = f"req_{uuid.uuid4().hex[:12]}"
        init_state: GraphRAGAgentState = {
            "request_id": req_id,
            "thread_id": tid,
            "tenant_id": tenant_id or self.pipeline.config.env.default_tenant_id,
            "acl_roles": acl_roles
            or [r.strip() for r in self.pipeline.config.env.default_acl_roles.split(",") if r.strip()],
            "question": question,
            "mode": mode,
            "reasoning": reasoning,
            "enable_writeback": enable_writeback,
            "require_hitl_approval": require_hitl_approval,
            "use_llm_rewrite": use_llm_rewrite,
            "retry_count": 0,
            "max_retries": self.max_retrieval_retries,
            "grounding_retry_count": 0,
            "max_grounding_retries": self.max_grounding_retries,
            "waterfall": [],
            "started_at": time.perf_counter(),
        }
        cfg = {"configurable": {"thread_id": tid}}
        out_state = self._graph.invoke(init_state, config=cfg)
        interrupted: dict[str, Any] | None = None
        if "__interrupt__" in out_state and out_state["__interrupt__"]:
            raw_int = out_state["__interrupt__"][0]
            interrupted = getattr(raw_int, "value", raw_int)
        return self._build_output(out_state, interrupted=interrupted)

    def resume_hitl(
        self,
        thread_id: str,
        *,
        approved: bool,
        reviewer: str = "human_reviewer",
    ) -> AgentRunOutput:
        """Resume a checkpointed LangGraph thread paused at `hitl_memory_approval`."""
        from langgraph.types import Command  # type: ignore[import-not-found]

        cfg = {"configurable": {"thread_id": thread_id}}
        out_state = self._graph.invoke(
            Command(resume={"approved": approved, "reviewer": reviewer}),
            config=cfg,
        )
        return self._build_output(out_state, interrupted=None)

    def stream_events(
        self,
        question: str,
        *,
        thread_id: str | None = None,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
        mode: str = "ppr",
        reasoning: str = "auto",
        enable_writeback: bool = False,
        require_hitl_approval: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield SSE-ready node-level events (`acl_guard -> query_rewrite -> ... -> final`)."""
        tid = thread_id or f"thread_{uuid.uuid4().hex[:10]}"
        req_id = f"req_{uuid.uuid4().hex[:12]}"
        init_state: GraphRAGAgentState = {
            "request_id": req_id,
            "thread_id": tid,
            "tenant_id": tenant_id or self.pipeline.config.env.default_tenant_id,
            "acl_roles": acl_roles
            or [r.strip() for r in self.pipeline.config.env.default_acl_roles.split(",") if r.strip()],
            "question": question,
            "mode": mode,
            "reasoning": reasoning,
            "enable_writeback": enable_writeback,
            "require_hitl_approval": require_hitl_approval,
            "use_llm_rewrite": False,
            "retry_count": 0,
            "max_retries": self.max_retrieval_retries,
            "grounding_retry_count": 0,
            "max_grounding_retries": self.max_grounding_retries,
            "waterfall": [],
            "started_at": time.perf_counter(),
        }
        cfg = {"configurable": {"thread_id": tid}}
        for chunk in self._graph.stream(init_state, config=cfg, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    raw_int = update[0] if isinstance(update, (list, tuple)) and update else update
                    yield {
                        "event": "hitl_interrupt",
                        "node": "hitl_memory_approval",
                        "thread_id": tid,
                        "request_id": req_id,
                        "data": getattr(raw_int, "value", raw_int),
                    }
                else:
                    wf = update.get("waterfall") if isinstance(update, dict) else None
                    last_step = wf[-1] if isinstance(wf, list) and wf else {}
                    yield {
                        "event": "node_update",
                        "node": node_name,
                        "thread_id": tid,
                        "request_id": req_id,
                        "data": last_step,
                    }
        snap = self._graph.get_state(cfg)
        final_vals = snap.values if snap is not None else {}
        yield {
            "event": "final",
            "node": "end",
            "thread_id": tid,
            "request_id": req_id,
            "data": self._build_output(final_vals).model_dump(),
        }

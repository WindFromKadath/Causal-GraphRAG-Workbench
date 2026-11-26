"""Stage 4.3 offline unit tests: Query Rewrite, ONNX/Lexical Reranker, LangGraph Cyclic/HITL, OTel & Eval v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from llm_graphrag_kg.agent.graph import GraphRAGAgentRunner
from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.schema import EvidenceItem, QueryResult, SubgraphTrace, TextChunk
from llm_graphrag_kg.eval.llm_judge import (
    JudgeQuestionScore,
    _summarize_mode_scores,
    classify_answerability_v2,
    classify_refusal_kind,
    compute_retrieval_ir_metrics,
)
from llm_graphrag_kg.observability.tracing import RequestTraceRecord, TraceRecorder, WaterfallStep
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.retrieval.query_rewrite import (
    expand_query_for_retry,
    fuse_multi_query_hits,
    rewrite_query,
)
from llm_graphrag_kg.retrieval.reranker import CrossEncoderReranker


def test_query_rewrite_preserves_original_and_extracts_self_query_filters() -> None:
    q = "离心泵机组同时出现2倍频振动升高和轴承温升异常，查一下 SHH 告警阈值表与诊断代码"
    res = rewrite_query(q, use_llm=False)
    assert res.rewritten_queries[0] == q
    assert len(res.rewritten_queries) >= 2
    assert "table" in res.self_query_filters.element_types
    assert "code" in res.self_query_filters.element_types
    assert "SHH" in res.self_query_filters.exact_codes

    expanded = expand_query_for_retry(q, res.rewritten_queries, retry_index=1)
    assert len(expanded) > len(res.rewritten_queries)

    c1 = TextChunk(chunk_id="c1", doc_id="d1", source="a.md", index=0, text="alpha", start_char=0, end_char=5)
    c2 = TextChunk(chunk_id="c2", doc_id="d1", source="a.md", index=1, text="beta", start_char=6, end_char=10)
    fused = fuse_multi_query_hits([[(c1, 0.9), (c2, 0.5)], [(c2, 0.95), (c1, 0.4)]], top_k=2)
    # c1 ranked #1 in original query (1.25x anchor weight) beats c2 ranked #1 in rewrite (1.0x)
    assert fused[0][0].chunk_id == "c1"


def test_reranker_lexical_and_redis_cache_and_onnx_cpu_mock() -> None:
    c_irrel = TextChunk(
        chunk_id="c_noise",
        doc_id="d1",
        source="a.md",
        index=0,
        text="食堂午餐菜单包括红烧肉与番茄炒蛋。",
        start_char=0,
        end_char=18,
    )
    c_rel = TextChunk(
        chunk_id="c_target",
        doc_id="d1",
        source="a.md",
        index=1,
        text="离心泵2倍频振动升高与轴承温升异常的根因是轴系不对中，需执行激光对中。",
        start_char=19,
        end_char=56,
    )

    class _FakeRedisCache:
        def __init__(self) -> None:
            self.kv: dict[str, str] = {}

        def get(self, k: str) -> str | None:
            return self.kv.get(k)

        def setex(self, k: str, ttl: int, val: str) -> None:
            _ = ttl
            self.kv[k] = val

    cache = _FakeRedisCache()
    rr = CrossEncoderReranker(backend="hybrid_lexical", redis_client=cache)
    res1 = rr.rerank("离心泵2倍频振动升高轴承温升异常", [(c_irrel, 0.9), (c_rel, 0.6)], top_n=2, tenant_id="t1")
    assert not res1.cache_hit
    assert res1.backend_used == "hybrid_lexical"
    assert res1.candidates[0].chunk.chunk_id == "c_target"

    # Second identical call hits Redis rerank cache
    res2 = rr.rerank("离心泵2倍频振动升高轴承温升异常", [(c_irrel, 0.9), (c_rel, 0.6)], top_n=2, tenant_id="t1")
    assert res2.cache_hit
    assert res2.candidates[0].chunk.chunk_id == "c_target"

    # Verify ONNX CPU inference path cleanly converts logits -> sigmoid probabilities
    rr_onnx = CrossEncoderReranker(backend="off")
    rr_onnx.backend = "onnx"
    rr_onnx._onnx_ready = True
    rr_onnx._input_names = {"input_ids", "attention_mask"}

    class _FakeEnc:
        def __init__(self) -> None:
            self.ids = [0, 12, 34, 2]
            self.attention_mask = [1, 1, 1, 1]
            self.type_ids = [0, 0, 0, 0]

    class _FakeTok:
        def encode_batch(self, pairs: list[tuple[str, str]]) -> list[_FakeEnc]:
            return [_FakeEnc() for _ in pairs]

    class _FakeSess:
        def run(self, _out: Any, _inp: dict[str, Any]) -> list[np.ndarray]:
            return [np.array([[-3.0], [4.0]], dtype=np.float32)]

    rr_onnx._tokenizer = _FakeTok()
    rr_onnx._session = _FakeSess()
    res_onnx = rr_onnx.rerank("test query", [(c_irrel, 0.9), (c_rel, 0.2)], top_n=2)
    assert res_onnx.backend_used == "onnx_cpu"
    assert res_onnx.candidates[0].chunk.chunk_id == "c_target"
    assert res_onnx.candidates[0].rerank_score > 0.95


def test_langgraph_agent_corrective_retry_grounding_self_heal_hitl_and_otel(tmp_path: Path) -> None:
    cfg = AppConfig(project_root=tmp_path)
    cfg.env.llm_provider = "mock"
    cfg.env.embedding_provider = "local_hash"
    cfg.env.reranker_backend = "hybrid_lexical"
    cfg.env.langgraph_checkpointer = "memory"

    pipeline = GraphRAGPipeline(cfg)
    c1 = TextChunk(
        chunk_id="doc_pump::c0",
        doc_id="doc_pump",
        source="pump.md",
        index=0,
        text="离心泵2倍频振动升高与轴承温升异常的根因是联轴器轴系不对中，应按激光对中规程复核。",
        start_char=0,
        end_char=42,
    )
    vecs = pipeline.embedder.embed([c1.text])
    pipeline.vector_store.add_chunks([c1], vecs)
    pipeline.memory_store.storage_dir = tmp_path

    # Stub pipeline.query so first call has ungrounded citation (triggers grounding self-heal!),
    # second call passes 100% GroundingAudit!
    call_counter = {"n": 0}

    def fake_query(question: str, **kwargs: Any) -> QueryResult:
        _ = kwargs
        call_counter["n"] += 1
        healed = call_counter["n"] >= 2 or "GROUNDING_SELF_HEAL" in question
        return QueryResult(
            question=question,
            mode="ppr",
            answer="离心泵2倍频振动升高与轴承温升异常的根因是联轴器轴系不对中，建议执行激光对中规程。",
            evidences=[
                EvidenceItem(
                    chunk_id=c1.chunk_id,
                    doc_id=c1.doc_id,
                    source=c1.source,
                    text=c1.text,
                    score=0.95,
                    matched_via="hybrid",
                )
            ],
            trace=SubgraphTrace(seed_entity_ids=["e_vib", "e_temp"], expanded_entity_ids=["e_vib", "e_temp"]),
            metadata={
                "reasoning_paradigm": "tot",
                "route_decision": {"route": "exclusive_diagnosis", "target_paradigm": "tot"},
                "reasoning": {
                    "confidence": 0.91,
                    "grounding_audit": {
                        "grounding_score": 1.0 if healed else 0.5,
                        "ungrounded_citations": [] if healed else ["fake_ent_999"],
                        "passed": healed,
                    },
                    "branches": [{"branch_id": "H1", "status": "selected", "target_entity_id": "e_misalign"}],
                    "steps": [{"step_id": 1, "cited_chunk_ids": ["doc_pump::c0"]}],
                },
            },
        )

    pipeline.query = fake_query  # type: ignore[method-assign]
    tracer = TraceRecorder(service_name="test-graphrag", persist_to_pg=False)
    # Set min_evidence_score=0.999 to force Corrective RAG retry loop!
    runner = GraphRAGAgentRunner(
        pipeline,
        tracer=tracer,
        checkpointer_mode="memory",
        min_evidence_score=0.999,
        max_retrieval_retries=1,
        max_grounding_retries=1,
    )

    # 1. Run with HITL interrupt enabled -> should pause at hitl_memory_approval!
    out_paused = runner.run(
        "离心泵2倍频振动升高和轴承温升异常怎么办？",
        thread_id="thread_hitl_1",
        enable_writeback=True,
        require_hitl_approval=True,
    )
    assert out_paused.status == "interrupted_hitl"
    assert out_paused.retry_count == 1  # Corrective RAG retry fired!
    assert out_paused.grounding_retry_count == 1  # Grounding self-heal fired!
    assert out_paused.grounding_score == 1.0
    assert out_paused.interrupt_payload is not None
    assert out_paused.interrupt_payload["type"] == "amem_writeback_approval"

    # 2. Resume HITL with approval=True -> commits verified memory & records OTel spans!
    out_done = runner.resume_hitl("thread_hitl_1", approved=True, reviewer="lead_engineer")
    assert out_done.status == "completed"
    assert out_done.writeback_decision is not None
    assert out_done.writeback_decision["accepted"] is True
    assert len(tracer.local_traces) == 1
    assert tracer.span_collector is not None and len(tracer.span_collector.spans) >= 5

    # 3. Verify HITL rejection cleanly blocks memory write-back
    out_rej_pause = runner.run(
        "离心泵2倍频振动升高和轴承温升异常怎么办？",
        thread_id="thread_hitl_2",
        enable_writeback=True,
        require_hitl_approval=True,
    )
    assert out_rej_pause.status == "interrupted_hitl"
    out_rej = runner.resume_hitl("thread_hitl_2", approved=False, reviewer="qa_auditor")
    assert out_rej.writeback_decision is not None
    assert out_rej.writeback_decision["accepted"] is False
    assert out_rej.writeback_decision["gate_failed"] == "GATE_HUMAN_HITL_REJECTED"

    # 4. Verify SSE event streaming yields node updates + final event
    events = list(runner.stream_events("离心泵2倍频振动升高怎么办？", enable_writeback=False))
    ev_nodes = [e["node"] for e in events if e["event"] == "node_update"]
    assert "acl_guard" in ev_nodes and "query_rewrite" in ev_nodes and "recall_and_rerank" in ev_nodes
    assert events[-1]["event"] == "final"
    runner.close()


def test_eval_v2_ir_metrics_answerability_and_execution_error_separation() -> None:
    ir = compute_retrieval_ir_metrics(["c9", "c2", "c5", "c1"], gold_ids={"c5"}, ks=(1, 3, 5, 10))
    assert ir["hit_at_1"] == 0.0
    assert ir["hit_at_3"] == 1.0
    assert ir["hit_at_5"] == 1.0
    assert pytest.approx(ir["mrr_at_10"], rel=1e-4) == 1.0 / 3.0

    assert classify_answerability_v2(answer_accuracy=0.85, semantic_coverage=0.80, is_refusal=False) == "FULL"
    assert classify_answerability_v2(answer_accuracy=0.55, semantic_coverage=0.50, is_refusal=False) == "PARTIAL"
    assert classify_answerability_v2(answer_accuracy=0.10, semantic_coverage=0.10, is_refusal=True) == "NONE"
    assert (
        classify_answerability_v2(answer_accuracy=0.0, semantic_coverage=0.0, is_refusal=False, execution_error=True)
        == "EXECUTION_ERROR"
    )

    assert classify_refusal_kind("信息不足，无法回答。", retrieved_evidence_count=0) == "business_refusal"
    assert (
        classify_refusal_kind("信息不足，无法回答。", retrieved_evidence_count=4, retrieval_evidence_recall=0.8)
        == "semantic_refusal"
    )
    assert classify_refusal_kind("根因为轴系不对中。", retrieved_evidence_count=4) == "none"

    # Execution error MUST NOT corrupt quality average denominator!
    s_good = JudgeQuestionScore(
        question_id="q1",
        question_type="Complex",
        mode="ppr",
        answer_accuracy=0.90,
        factuality_f1=0.90,
        semantic_similarity=0.90,
        faithfulness=0.95,
        semantic_coverage=0.85,
        answerability="FULL",
        hit_at_5=1.0,
        mrr_at_10=1.0,
    )
    s_err = JudgeQuestionScore(
        question_id="q2",
        question_type="Complex",
        mode="ppr",
        answer_accuracy=0.0,
        factuality_f1=0.0,
        semantic_similarity=0.0,
        faithfulness=0.0,
        semantic_coverage=0.0,
        answerability="EXECUTION_ERROR",
        execution_error=True,
    )
    summary = _summarize_mode_scores("ppr", [s_good, s_err])
    assert summary.total == 2
    assert summary.valid_scored_count == 1
    assert summary.execution_error_count == 1
    assert pytest.approx(summary.avg_answer_accuracy, rel=1e-4) == 0.90
    assert summary.avg_mrr_at_10 == 1.0

    # Quick check on RequestTraceRecord model
    tr = RequestTraceRecord(question="q", waterfall=[WaterfallStep(node="acl_guard", duration_ms=1.2)])
    assert tr.waterfall[0].node == "acl_guard"

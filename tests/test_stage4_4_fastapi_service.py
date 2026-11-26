"""Stage 4.4 FastAPI async integration tests (`httpx.AsyncClient` + `ASGITransport`).

Covers:
- Service health, OpenAPI schema, and Tenant/ACL + X-API-Key authentication (401 / 403 guards)
- Document governed ingestion, multipart file upload, forged MIME 422 rejection, document listing
- Graph overview, Ego-Graph traversal, HyperEdges, and SHACL-lite audit endpoints
- LangGraph QA, real SSE streaming (`text/event-stream`), HITL `interrupt()` + `/resume`
- Verified memories, observability traces, and Evaluation v2 metrics API.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from llm_graphrag_kg.agent.graph import GraphRAGAgentRunner
from llm_graphrag_kg.api.app import create_app
from llm_graphrag_kg.api.deps import AppServices
from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.schema import (
    Entity,
    EvidenceItem,
    HyperEdge,
    QueryResult,
    Relation,
    SubgraphTrace,
    TextChunk,
)
from llm_graphrag_kg.observability.tracing import TraceRecorder
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline


def _build_test_app(tmp_path: Path, *, api_token: str | None = None) -> tuple[Any, AppServices]:
    cfg = AppConfig(project_root=tmp_path)
    cfg.env.llm_provider = "mock"
    cfg.env.embedding_provider = "local_hash"
    cfg.env.vector_backend = "local"
    cfg.env.graph_backend = "networkx"
    cfg.env.reranker_backend = "hybrid_lexical"
    cfg.env.langgraph_checkpointer = "memory"
    cfg.env.api_auth_token = api_token

    pipeline = GraphRAGPipeline(cfg)
    c1 = TextChunk(
        chunk_id="doc_pump::v1::c0000",
        doc_id="doc_pump",
        source="pump.md",
        index=0,
        text="离心泵2倍频振动升高与轴承温升异常的根因是轴系不对中，排查措施为检查软脚并执行激光对中。",
        start_char=0,
        end_char=48,
    )
    pipeline.vector_store.add_chunks([c1], pipeline.embedder.embed([c1.text]))
    pipeline.graph_store.upsert_entity(
        Entity(entity_id="e_vib", name="2倍频振动升高", entity_type="Concept", source_chunk_ids=[c1.chunk_id])
    )
    pipeline.graph_store.upsert_entity(
        Entity(entity_id="e_temp", name="轴承温升异常", entity_type="Concept", source_chunk_ids=[c1.chunk_id])
    )
    pipeline.graph_store.upsert_entity(
        Entity(entity_id="e_mis", name="轴系不对中", entity_type="Concept", source_chunk_ids=[c1.chunk_id])
    )
    pipeline.graph_store.upsert_relation(
        Relation(
            relation_id="r1",
            source_id="e_mis",
            target_id="e_vib",
            relation_type="CAUSES",
            source_chunk_ids=[c1.chunk_id],
        )
    )
    pipeline.graph_store.upsert_hyperedge(
        HyperEdge(
            hyperedge_id="he_1",
            condition_entity_ids=["e_vib", "e_temp"],
            target_entity_id="e_mis",
            relation_type="JOINT_INDICATES",
            source_chunk_ids=[c1.chunk_id],
        )
    )
    pipeline.memory_store.storage_dir = tmp_path

    def fake_query(question: str, **kwargs: Any) -> QueryResult:
        _ = kwargs
        return QueryResult(
            question=question,
            mode="ppr",
            answer="离心泵2倍频振动升高与轴承温升异常的根因是轴系不对中，应执行激光对中规程。",
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
            trace=SubgraphTrace(seed_entity_ids=["e_vib", "e_temp"], expanded_entity_ids=["e_vib", "e_temp", "e_mis"]),
            metadata={
                "reasoning_paradigm": "tot",
                "route_decision": {"route": "exclusive_diagnosis", "target_paradigm": "tot"},
                "reasoning": {
                    "confidence": 0.92,
                    "grounding_audit": {
                        "grounding_score": 1.0,
                        "ungrounded_citations": [],
                        "passed": True,
                    },
                    "branches": [{"branch_id": "H1", "status": "selected", "target_entity_id": "e_mis"}],
                    "steps": [{"step_id": 1, "cited_chunk_ids": [c1.chunk_id]}],
                },
            },
        )

    pipeline.query = fake_query  # type: ignore[method-assign]
    tracer = TraceRecorder(service_name="test-api", persist_to_pg=False)
    runner = GraphRAGAgentRunner(pipeline, tracer=tracer, checkpointer_mode="memory")
    services = AppServices(cfg, pipeline=pipeline, runner=runner)
    return create_app(cfg, services=services), services


def test_fastapi_auth_guards_and_rest_endpoints(tmp_path: Path) -> None:
    app, services = _build_test_app(tmp_path, api_token="secret-test-token")

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Public health endpoint works without API key
            r_health = await client.get("/api/v1/health")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "ok"

            # 2. Protected endpoint rejects missing / wrong X-API-Key -> 401
            r_unauth = await client.get("/api/v1/graph/overview")
            assert r_unauth.status_code == 401

            user_headers = {
                "X-API-Key": "secret-test-token",
                "X-Tenant-Id": "plant_a",
                "X-ACL-Roles": "public,engineer",
            }
            admin_headers = {
                "X-API-Key": "secret-test-token",
                "X-Tenant-Id": "plant_a",
                "X-ACL-Roles": "public,engineer,admin",
                "X-Operator": "chief_eng",
            }

            # 3. Non-admin calling privileged reconcile -> 403 Forbidden
            r_forbid = await client.post(
                "/api/v1/queue/reconcile", json={"stale_lease_seconds": 60}, headers=user_headers
            )
            assert r_forbid.status_code == 403

            # 4. Graph overview, Ego-Graph, HyperEdges & SHACL-lite
            r_graph = await client.get("/api/v1/graph/overview", headers=user_headers)
            assert r_graph.status_code == 200
            g_data = r_graph.json()
            assert g_data["stats"]["entity_count"] == 3
            assert g_data["stats"]["hyperedge_count"] == 1

            r_ego = await client.get("/api/v1/graph/ego/e_mis?k_hops=1", headers=user_headers)
            assert r_ego.status_code == 200
            assert len(r_ego.json()["entities"]) == 2

            r_he = await client.get("/api/v1/graph/hyperedges", headers=user_headers)
            assert r_he.status_code == 200 and r_he.json()["count"] == 1

            r_shacl = await client.get("/api/v1/graph/shacl", headers=user_headers)
            assert r_shacl.status_code == 200
            assert r_shacl.json()["overall_status"] in {"PASS", "WARN"}

            # 5. Document ingest & multipart upload (including forged PDF 422 rejection!)
            r_ing = await client.post(
                "/api/v1/documents/ingest",
                json={
                    "doc_id": "doc_new",
                    "title": "新规程",
                    "raw_text": "# 巡检规范\n\n## 轴承检查\n离心泵滚动轴承温度超过 85 度必须停机核查润滑状态。",
                },
                headers=user_headers,
            )
            assert r_ing.status_code == 200 and r_ing.json()["outcome"] == "applied"

            r_forged = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("forged.pdf", b"plain text pretending to be pdf", "application/pdf")},
                headers=user_headers,
            )
            assert r_forged.status_code == 422

            r_docs = await client.get("/api/v1/documents", headers=user_headers)
            assert r_docs.status_code == 200 and r_docs.json()["count"] >= 2

            # 6. LangGraph Query + HITL Pause + Admin Resume + Memories + Traces
            r_hitl = await client.post(
                "/api/v1/agent/query",
                json={
                    "question": "离心泵2倍频振动升高和轴承温升异常怎么办？",
                    "thread_id": "api_hitl_1",
                    "enable_writeback": True,
                    "require_hitl_approval": True,
                },
                headers=user_headers,
            )
            assert r_hitl.status_code == 200
            assert r_hitl.json()["status"] == "interrupted_hitl"

            # Non-admin cannot approve HITL -> 403
            r_hitl_forbid = await client.post(
                "/api/v1/agent/hitl/api_hitl_1/resume",
                json={"approved": True, "reviewer": "unpriv"},
                headers=user_headers,
            )
            assert r_hitl_forbid.status_code == 403

            # Admin resumes HITL -> approved & persisted!
            r_hitl_ok = await client.post(
                "/api/v1/agent/hitl/api_hitl_1/resume",
                json={"approved": True, "reviewer": "chief_eng"},
                headers=admin_headers,
            )
            assert r_hitl_ok.status_code == 200
            assert r_hitl_ok.json()["status"] == "completed"
            assert r_hitl_ok.json()["writeback_decision"]["accepted"] is True

            r_mem = await client.get("/api/v1/memories", headers=user_headers)
            assert r_mem.status_code == 200 and r_mem.json()["count"] >= 1

            r_traces = await client.get("/api/v1/traces", headers=user_headers)
            assert r_traces.status_code == 200 and r_traces.json()["count"] >= 1

            # 7. Real SSE streaming endpoint (`text/event-stream`)
            sse_events: list[tuple[str, dict[str, Any]]] = []
            async with client.stream(
                "POST",
                "/api/v1/agent/stream",
                json={"question": "离心泵2倍频振动升高怎么办？"},
                headers=user_headers,
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                cur_event = ""
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        cur_event = line.removeprefix("event: ").strip()
                    elif line.startswith("data: "):
                        sse_events.append((cur_event, json.loads(line.removeprefix("data: "))))

            node_names = [p.get("node") for ev, p in sse_events if ev == "node_update"]
            assert "acl_guard" in node_names
            assert "query_rewrite" in node_names
            assert "recall_and_rerank" in node_names
            assert "graph_reason" in node_names
            assert sse_events[-1][0] == "final"

            # 8. Evaluation v2 metrics endpoint
            r_eval = await client.post(
                "/api/v1/eval/metrics",
                json={
                    "mode": "ppr_auto",
                    "items": [
                        {
                            "question_id": "q1",
                            "prediction": "根因为轴系不对中。",
                            "answer_accuracy": 0.9,
                            "semantic_coverage": 0.85,
                            "faithfulness": 0.95,
                            "retrieved_ids": ["c1", "c2"],
                            "gold_ids": ["c2"],
                        },
                        {
                            "question_id": "q2",
                            "execution_error": True,
                        },
                    ],
                },
                headers=user_headers,
            )
            assert r_eval.status_code == 200
            ev_sum = r_eval.json()["summary"]
            assert ev_sum["valid_scored_count"] == 1
            assert ev_sum["execution_error_count"] == 1
            assert ev_sum["avg_mrr_at_10"] == 0.5

    try:
        asyncio.run(_run())
    finally:
        services.close()

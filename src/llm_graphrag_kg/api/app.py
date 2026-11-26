"""Stage 4.4 Production FastAPI Application (Async REST + SSE Streaming + OpenAPI + Tenant/ACL Auth)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from llm_graphrag_kg.api.deps import AppServices, get_auth_context, get_services, require_admin
from llm_graphrag_kg.api.schemas import (
    AgentQueryRequest,
    AuthContext,
    DocumentIngestRequest,
    DocumentRollbackRequest,
    EvalSummaryRequest,
    HitlResumeRequest,
    InfraHealthResponse,
    InfraServiceItem,
    ReconcileRequest,
    ServiceHealthResponse,
    WorkerConsumeRequest,
)
from llm_graphrag_kg.config import AppConfig, load_config
from llm_graphrag_kg.eval.llm_judge import (
    JudgeQuestionScore,
    _summarize_mode_scores,
    classify_answerability_v2,
    classify_refusal_kind,
    compute_retrieval_ir_metrics,
    detect_refusal,
)
from llm_graphrag_kg.pipelines.doc_governance import govern_document
from llm_graphrag_kg.pipelines.incremental_worker import DocumentChangeEvent
from llm_graphrag_kg.pipelines.migrate_enterprise import check_enterprise_infra
from llm_graphrag_kg.providers.storage.neo4j_store import Neo4jGraphStore
from llm_graphrag_kg.providers.storage.pgvector_store import PgVectorStore
from llm_graphrag_kg.viz.schema_validator import validate_kg_shacl_lite


def _apply_store_scope(services: AppServices, auth: AuthContext) -> None:
    """Bind caller's verified `tenant_id` and `acl_roles` onto enterprise stores before read/audit."""
    for store in (
        services.pipeline.vector_store,
        services.pipeline.graph_store,
        services.pipeline.memory_store,
    ):
        if hasattr(store, "default_tenant_id"):
            store.default_tenant_id = auth.tenant_id
            store.default_acl_roles = list(auth.acl_roles)


def _format_sse(event_name: str, payload: Any) -> str:
    data_str = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_name}\ndata: {data_str}\n\n"


def create_app(
    config: AppConfig | None = None,
    *,
    services: AppServices | None = None,
) -> FastAPI:
    """Create a configured, OpenAPI-documented FastAPI app for Stage 4.4 & 4.5."""
    cfg = config or (services.config if services is not None else load_config())

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if getattr(application.state, "services", None) is None:
            application.state.services = services or AppServices(cfg)
        try:
            yield
        finally:
            svc: AppServices | None = getattr(application.state, "services", None)
            if svc is not None and services is None:
                svc.close()

    app = FastAPI(
        title="Causal-GraphRAG-Workbench API",
        version="4.5.0",
        description=(
            "Causal-GraphRAG-Workbench production service: "
            "PG16/pgvector + Neo4j 5.26 + Redis Streams + Thin LangGraph (Corrective RAG & HITL) + SSE."
        ),
        lifespan=lifespan,
    )
    if services is not None:
        app.state.services = services

    origins = [o.strip() for o in cfg.env.api_cors_origins.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # 1. Health & Infrastructure Diagnostics
    # ------------------------------------------------------------------
    @app.get("/api/v1/health", response_model=ServiceHealthResponse, tags=["Health"])
    async def health(svc: AppServices = Depends(get_services)) -> ServiceHealthResponse:
        env = svc.config.env
        return ServiceHealthResponse(
            status="ok",
            vector_backend=env.vector_backend,
            graph_backend=env.graph_backend,
            checkpointer=svc.runner.checkpointer_mode,
            llm_model=env.llm_model,
            embedding_model=env.embedding_model,
            reranker_backend=svc.runner.reranker.backend,
        )

    @app.get("/api/v1/infra/health", response_model=InfraHealthResponse, tags=["Health"])
    async def infra_health(
        svc: AppServices = Depends(get_services),
        _auth: AuthContext = Depends(get_auth_context),
    ) -> InfraHealthResponse:
        items = check_enterprise_infra(svc.config)
        mapped = [
            InfraServiceItem(service=i.service, ok=i.ok, endpoint=i.endpoint, detail=i.detail)
            for i in items
        ]
        return InfraHealthResponse(all_healthy=all(i.ok for i in mapped), services=mapped)

    @app.get("/api/v1/tenants", tags=["Health"])
    async def list_tenants(
        svc: AppServices = Depends(get_services),
        _auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        """Discover all active tenants across PostgreSQL and Neo4j for multi-dataset switching."""
        tenants: set[str] = {svc.config.env.default_tenant_id}
        vs = svc.pipeline.vector_store
        if isinstance(vs, PgVectorStore):
            with vs._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT tenant_id FROM child_chunks WHERE is_deleted = FALSE ORDER BY tenant_id"
                )
                tenants.update(str(r[0]) for r in cur.fetchall() if r and r[0])
        gs = svc.pipeline.graph_store
        if isinstance(gs, Neo4jGraphStore):
            with gs._session() as sess:
                rows = sess.run(
                    "MATCH (e:Entity) WHERE coalesce(e.is_deleted, false) = false "
                    "RETURN DISTINCT e.tenant_id AS tid ORDER BY tid"
                ).data()
                tenants.update(str(r["tid"]) for r in rows if r.get("tid"))
        sorted_t = sorted(t for t in tenants if t)
        return {"count": len(sorted_t), "tenants": sorted_t}

    # ------------------------------------------------------------------
    # 2. Documents, Quality Gates, Versioning & Redis Queue / DLQ
    # ------------------------------------------------------------------
    @app.get("/api/v1/documents", tags=["Documents"])
    async def list_documents(
        include_deleted: bool = Query(default=False),
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        vs = svc.pipeline.vector_store
        if isinstance(vs, PgVectorStore):
            with vs._connect() as conn, conn.cursor() as cur:
                where_del = "" if include_deleted else "AND d.is_deleted = FALSE"
                cur.execute(
                    f"""
                    SELECT d.doc_id, d.title, d.source_type, d.source_uri, d.active_version,
                           d.status, d.is_deleted, d.acl, d.missing_pages, d.updated_at,
                           (SELECT COUNT(*) FROM child_chunks c
                            WHERE c.tenant_id = d.tenant_id AND c.doc_id = d.doc_id
                              AND c.version_id = d.active_version AND c.is_deleted = FALSE) AS chunk_cnt
                    FROM documents d
                    WHERE d.tenant_id = %s
                      AND d.acl && %s::text[]
                      {where_del}
                    ORDER BY d.updated_at DESC
                    LIMIT 200
                    """,
                    (auth.tenant_id, auth.acl_roles),
                )
                docs = [
                    {
                        "doc_id": str(r[0]),
                        "title": str(r[1]),
                        "source_type": str(r[2]),
                        "source_uri": str(r[3]),
                        "active_version": int(r[4]),
                        "status": str(r[5]),
                        "is_deleted": bool(r[6]),
                        "acl": list(r[7] or []),
                        "missing_pages": list(r[8] or []),
                        "updated_at": r[9].isoformat() if hasattr(r[9], "isoformat") else str(r[9]),
                        "chunk_count": int(r[10]),
                    }
                    for r in cur.fetchall()
                ]
                return {"tenant_id": auth.tenant_id, "count": len(docs), "documents": docs}

        # Local vector store fallback
        chunks = vs.get_all_chunks()
        grouped: dict[str, dict[str, Any]] = {}
        for c in chunks:
            entry = grouped.setdefault(
                c.doc_id,
                {
                    "doc_id": c.doc_id,
                    "title": Path(c.source).name,
                    "source_type": Path(c.source).suffix.lstrip(".") or "text",
                    "source_uri": c.source,
                    "active_version": int(c.metadata.get("version_id", 1)),
                    "status": "ready",
                    "is_deleted": False,
                    "acl": list(c.metadata.get("acl", auth.acl_roles)),
                    "missing_pages": [],
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
        return {"tenant_id": auth.tenant_id, "count": len(grouped), "documents": list(grouped.values())}

    @app.post("/api/v1/documents/ingest", tags=["Documents"])
    async def ingest_document_text(
        body: DocumentIngestRequest,
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        eff_acl = body.acl or auth.acl_roles
        if svc.worker is not None:
            evt = DocumentChangeEvent(
                tenant_id=auth.tenant_id,
                doc_id=body.doc_id,
                version_id=body.version_id,
                op="upsert",
                source_uri=body.source_uri or f"{body.doc_id}.md",
                title=body.title or body.doc_id,
                acl=eff_acl,
                raw_text=body.raw_text,
                operator=auth.operator,
            )
            if body.via_redis:
                stream_id = svc.worker.enqueue_event(evt)
                return {"enqueued": True, "stream_id": stream_id, "event_id": evt.event_id}
            res = svc.worker.process_event(evt)
            if res.outcome == "rejected_quality_gate":
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=res.model_dump())
            return res.model_dump()

        # Offline local store governed ingestion fallback
        gov = govern_document(
            doc_id=body.doc_id,
            raw_text=body.raw_text,
            title=body.title or body.doc_id,
            source_uri=body.source_uri or f"{body.doc_id}.md",
            version_id=body.version_id,
            acl=eff_acl,
        )
        if gov.status == "rejected":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"outcome": "rejected_quality_gate", "reason": gov.rejection_reason},
            )
        text_chunks = [c.to_text_chunk() for c in gov.child_chunks]
        vecs = svc.pipeline.embedder.embed([c.breadcrumb_text for c in gov.child_chunks])
        svc.pipeline.vector_store.add_chunks(text_chunks, vecs)
        return {
            "outcome": "applied",
            "doc_id": body.doc_id,
            "version_id": body.version_id,
            "status": gov.status,
            "parent_chunks": len(gov.parent_chunks),
            "child_chunks": len(gov.child_chunks),
            "reused_embeddings": 0,
            "computed_embeddings": len(gov.child_chunks),
        }

    @app.post("/api/v1/documents/upload", tags=["Documents"])
    async def upload_document_file(
        file: UploadFile = File(...),
        doc_id: str | None = Form(default=None),
        version_id: int = Form(default=1),
        acl: str | None = Form(default=None),
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        filename = Path(file.filename or "upload.md").name
        eff_doc_id = (doc_id or f"doc_{Path(filename).stem}").strip()
        eff_acl = [r.strip() for r in acl.split(",") if r.strip()] if acl else auth.acl_roles
        raw_bytes = await file.read()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            tmp_path.write_bytes(raw_bytes)

            if svc.worker is not None:
                evt = DocumentChangeEvent(
                    tenant_id=auth.tenant_id,
                    doc_id=eff_doc_id,
                    version_id=version_id,
                    op="upsert",
                    source_uri=filename,
                    title=Path(filename).stem,
                    acl=eff_acl,
                    file_path=str(tmp_path),
                    operator=auth.operator,
                )
                res = svc.worker.process_event(evt)
                if res.outcome == "rejected_quality_gate":
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=res.model_dump()
                    )
                return res.model_dump()

            gov = govern_document(
                doc_id=eff_doc_id,
                file_path=tmp_path,
                title=Path(filename).stem,
                source_uri=filename,
                version_id=version_id,
                acl=eff_acl,
            )
            if gov.status == "rejected":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={"outcome": "rejected_quality_gate", "reason": gov.rejection_reason},
                )
            text_chunks = [c.to_text_chunk() for c in gov.child_chunks]
            vecs = svc.pipeline.embedder.embed([c.breadcrumb_text for c in gov.child_chunks])
            svc.pipeline.vector_store.add_chunks(text_chunks, vecs)
            return {
                "outcome": "applied",
                "doc_id": eff_doc_id,
                "version_id": version_id,
                "parser_used": gov.parser_used,
                "status": gov.status,
                "parent_chunks": len(gov.parent_chunks),
                "child_chunks": len(gov.child_chunks),
            }

    @app.post("/api/v1/documents/{doc_id}/rollback", tags=["Documents"])
    async def rollback_document(
        doc_id: str,
        body: DocumentRollbackRequest,
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        vs = svc.pipeline.vector_store
        if not isinstance(vs, PgVectorStore):
            raise HTTPException(status_code=400, detail="Version rollback requires PostgreSQL backend.")
        ok = vs.rollback_document_version(
            doc_id,
            body.target_version,
            tenant_id=auth.tenant_id,
            operator=auth.operator,
        )
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"No ready chunks found for `{doc_id}` version v{body.target_version}.",
            )
        return {"doc_id": doc_id, "active_version": body.target_version, "rolled_back": True}

    @app.delete("/api/v1/documents/{doc_id}", tags=["Documents"])
    async def delete_document(
        doc_id: str,
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        if svc.worker is None:
            raise HTTPException(status_code=400, detail="Soft-delete requires enterprise PG backend.")
        res = svc.worker.process_event(
            DocumentChangeEvent(
                tenant_id=auth.tenant_id,
                doc_id=doc_id,
                op="delete",
                acl=auth.acl_roles,
                operator=auth.operator,
            )
        )
        return res.model_dump()

    @app.get("/api/v1/queue/dlq", tags=["Queue"])
    async def list_dlq(
        limit: int = Query(default=20, ge=1, le=200),
        svc: AppServices = Depends(get_services),
        _auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        if svc.worker is None:
            return {"count": 0, "entries": []}
        entries = svc.worker.list_dlq(limit=limit)
        return {"count": len(entries), "entries": entries}

    @app.post("/api/v1/queue/worker/consume", tags=["Queue"])
    async def consume_worker_batch(
        body: WorkerConsumeRequest,
        svc: AppServices = Depends(get_services),
        _auth: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        if svc.worker is None:
            raise HTTPException(status_code=400, detail="Redis Streams worker requires PG backend.")
        results = svc.worker.consume_batch(max_messages=body.max_messages, block_ms=body.block_ms)
        return {"processed": len(results), "results": [r.model_dump() for r in results]}

    @app.post("/api/v1/queue/reconcile", tags=["Queue"])
    async def run_reconciliation(
        body: ReconcileRequest,
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        if svc.worker is None:
            raise HTTPException(status_code=400, detail="Reconciliation requires enterprise PG backend.")
        rep = svc.worker.reconcile(
            tenant_id=auth.tenant_id,
            stale_lease_seconds=body.stale_lease_seconds,
            purge_superseded=body.purge_superseded,
        )
        return rep.model_dump()

    # ------------------------------------------------------------------
    # 3. Knowledge Graph, Ego-Graph, HyperEdges & SHACL-lite
    # ------------------------------------------------------------------
    @app.get("/api/v1/graph/overview", tags=["Graph"])
    async def graph_overview(
        limit_nodes: int = Query(default=200, ge=1, le=2000),
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        _apply_store_scope(svc, auth)
        gs = svc.pipeline.graph_store
        if isinstance(gs, Neo4jGraphStore):
            entities = gs.get_all_entities(tenant_id=auth.tenant_id, acl_roles=auth.acl_roles)
            relations = gs.get_all_relations(tenant_id=auth.tenant_id, acl_roles=auth.acl_roles)
            hyperedges = gs.get_all_hyperedges(tenant_id=auth.tenant_id, acl_roles=auth.acl_roles)
            communities = gs.get_all_communities(tenant_id=auth.tenant_id, acl_roles=auth.acl_roles)
        else:
            entities = gs.get_all_entities()
            relations = gs.get_all_relations()
            hyperedges = gs.get_all_hyperedges() if hasattr(gs, "get_all_hyperedges") else []
            communities = gs.get_all_communities()

        kept_e = entities[:limit_nodes]
        kept_ids = {e.entity_id for e in kept_e}
        kept_r = [r for r in relations if r.source_id in kept_ids and r.target_id in kept_ids]
        return {
            "tenant_id": auth.tenant_id,
            "stats": {
                "entity_count": len(entities),
                "relation_count": len(relations),
                "hyperedge_count": len(hyperedges),
                "community_count": len(communities),
            },
            "entities": [e.model_dump() for e in kept_e],
            "relations": [r.model_dump() for r in kept_r],
            "hyperedges": [h.model_dump() for h in hyperedges],
            "communities": [c.model_dump() for c in communities],
        }

    @app.get("/api/v1/graph/ego/{entity_id}", tags=["Graph"])
    async def graph_ego(
        entity_id: str,
        k_hops: int = Query(default=2, ge=0, le=4),
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        _apply_store_scope(svc, auth)
        gs = svc.pipeline.graph_store
        if isinstance(gs, Neo4jGraphStore):
            nodes, rels = gs.get_k_hop_subgraph(
                [entity_id], k_hops=k_hops, tenant_id=auth.tenant_id, acl_roles=auth.acl_roles
            )
        else:
            nodes, rels = gs.get_k_hop_subgraph([entity_id], k_hops=k_hops)
        if not nodes:
            raise HTTPException(status_code=404, detail=f"Entity `{entity_id}` not found or forbidden.")
        return {
            "tenant_id": auth.tenant_id,
            "seed_entity_id": entity_id,
            "k_hops": k_hops,
            "entities": [e.model_dump() for e in nodes],
            "relations": [r.model_dump() for r in rels],
        }

    @app.get("/api/v1/graph/hyperedges", tags=["Graph"])
    async def list_hyperedges(
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        _apply_store_scope(svc, auth)
        gs = svc.pipeline.graph_store
        if isinstance(gs, Neo4jGraphStore):
            hes = gs.get_all_hyperedges(tenant_id=auth.tenant_id, acl_roles=auth.acl_roles)
        else:
            hes = gs.get_all_hyperedges() if hasattr(gs, "get_all_hyperedges") else []
        return {"tenant_id": auth.tenant_id, "count": len(hes), "hyperedges": [h.model_dump() for h in hes]}

    @app.get("/api/v1/graph/shacl", tags=["Graph"])
    async def run_shacl_audit(
        profile: str | None = Query(
            default=None,
            description="Optional ontology profile: default | fault_diagnosis | fmea_kg_rdf | failuresensoriq",
        ),
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        _apply_store_scope(svc, auth)
        audit_cfg = svc.config
        allowed_profiles = {
            "default": "configs/default.yaml",
            "fault_diagnosis": "configs/fault_diagnosis.yaml",
            "fmea_kg_rdf": "configs/fmea_kg_rdf.yaml",
            "failuresensoriq": "configs/failuresensoriq.yaml",
        }
        if profile and profile in allowed_profiles:
            prof_path = svc.config.project_root / allowed_profiles[profile]
            if prof_path.is_file():
                audit_cfg = load_config(config_path=prof_path, project_root=svc.config.project_root)
        rep = validate_kg_shacl_lite(svc.pipeline.graph_store, svc.pipeline.vector_store, audit_cfg)
        return rep.model_dump()

    # ------------------------------------------------------------------
    # 4. LangGraph Cyclic QA, Real-Time SSE Stream & HITL A-MEM Approval
    # ------------------------------------------------------------------
    @app.post("/api/v1/agent/query", tags=["Agent"])
    async def agent_query(
        body: AgentQueryRequest,
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        out = svc.runner.run(
            body.question,
            thread_id=body.thread_id,
            tenant_id=auth.tenant_id,
            acl_roles=auth.acl_roles,
            mode=body.mode,
            reasoning=body.reasoning,
            enable_writeback=body.enable_writeback,
            require_hitl_approval=body.require_hitl_approval,
            use_llm_rewrite=body.use_llm_rewrite,
        )
        return out.model_dump()

    @app.post("/api/v1/agent/stream", tags=["Agent"])
    async def agent_stream_sse(
        body: AgentQueryRequest,
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> StreamingResponse:
        def event_generator() -> Any:
            try:
                for ev in svc.runner.stream_events(
                    body.question,
                    thread_id=body.thread_id,
                    tenant_id=auth.tenant_id,
                    acl_roles=auth.acl_roles,
                    mode=body.mode,
                    reasoning=body.reasoning,
                    enable_writeback=body.enable_writeback,
                    require_hitl_approval=body.require_hitl_approval,
                ):
                    yield _format_sse(str(ev.get("event", "node_update")), ev)
            except Exception as exc:  # noqa: BLE001
                yield _format_sse("error", {"error": str(exc)})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/v1/agent/hitl/{thread_id}/resume", tags=["Agent"])
    async def resume_hitl(
        thread_id: str,
        body: HitlResumeRequest,
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        out = svc.runner.resume_hitl(
            thread_id,
            approved=body.approved,
            reviewer=body.reviewer or auth.operator,
        )
        return out.model_dump()

    @app.get("/api/v1/memories", tags=["Agent"])
    async def list_memories(
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        _apply_store_scope(svc, auth)
        ms = svc.pipeline.memory_store
        if hasattr(ms, "dsn"):
            records = ms.get_all(tenant_id=auth.tenant_id, acl_roles=auth.acl_roles)
        else:
            records = ms.get_all()
        return {
            "tenant_id": auth.tenant_id,
            "count": len(records),
            "memories": [r.model_dump() for r in records],
        }

    # ------------------------------------------------------------------
    # 5. Observability Request Traces & Evaluation v2
    # ------------------------------------------------------------------
    @app.get("/api/v1/traces", tags=["Observability"])
    async def list_traces(
        limit: int = Query(default=20, ge=1, le=200),
        svc: AppServices = Depends(get_services),
        auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        traces = svc.runner.tracer.list_recent_pg_traces(tenant_id=auth.tenant_id, limit=limit)
        return {
            "tenant_id": auth.tenant_id,
            "count": len(traces),
            "traces": [t.model_dump() for t in traces],
        }

    @app.post("/api/v1/eval/metrics", tags=["Evaluation"])
    async def compute_eval_v2_metrics(
        body: EvalSummaryRequest,
        _auth: AuthContext = Depends(get_auth_context),
    ) -> dict[str, Any]:
        scores: list[JudgeQuestionScore] = []
        for item in body.items:
            ir = compute_retrieval_ir_metrics(item.retrieved_ids, item.gold_ids, ks=(1, 3, 5, 10))
            is_ref = detect_refusal(item.prediction) if not item.execution_error else False
            ans_state = classify_answerability_v2(
                answer_accuracy=item.answer_accuracy,
                semantic_coverage=item.semantic_coverage,
                is_refusal=is_ref,
                execution_error=item.execution_error,
            )
            ref_kind = (
                classify_refusal_kind(item.prediction, retrieved_evidence_count=len(item.retrieved_ids))
                if is_ref
                else "none"
            )
            scores.append(
                JudgeQuestionScore(
                    question_id=item.question_id,
                    question_type=item.question_type,
                    mode=body.mode,
                    answer_accuracy=item.answer_accuracy,
                    factuality_f1=item.factuality_f1,
                    semantic_similarity=item.semantic_similarity,
                    faithfulness=item.faithfulness,
                    semantic_coverage=item.semantic_coverage,
                    retrieval_evidence_recall=ir["hit_at_5"],
                    is_refusal=is_ref,
                    answerability=ans_state,
                    refusal_kind=ref_kind,
                    execution_error=item.execution_error,
                    hit_at_5=ir["hit_at_5"],
                    mrr_at_10=ir["mrr_at_10"],
                )
            )
        summary = _summarize_mode_scores(body.mode, scores)
        return {
            "summary": summary.model_dump(),
            "questions": [s.model_dump() for s in scores],
        }

    # Optional Stage 4.5 built React SPA mounting (serves production console at `/` when `frontend/dist` exists)
    dist_dir = cfg.project_root / "frontend" / "dist"
    assets_dir = dist_dir / "assets"
    if dist_dir.is_dir() and (dist_dir / "index.html").is_file():
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        async def serve_spa_root() -> FileResponse:
            return FileResponse(dist_dir / "index.html")

    return app

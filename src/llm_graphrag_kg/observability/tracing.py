"""Stage 4.3 Full-Pipeline Observability: PostgreSQL `request_traces` + OpenTelemetry Export.

Captures per-request enterprise telemetry compatible with Langfuse/Jaeger/OTLP collectors:
- Request waterfall (`acl_guard -> query_rewrite -> recall_and_rerank -> graph_reason -> writeback`)
- Model version snapshot (`llm_model`, `embedding_model`, `reranker_model`)
- Terminal status (`success | business_refusal | hitl_pending | error`)
- Durable PostgreSQL `request_traces` ledger + query API for Stage 4.4/4.5 dashboards.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class WaterfallStep(BaseModel):
    node: str
    duration_ms: float = 0.0
    status: str = "ok"
    summary: dict[str, Any] = Field(default_factory=dict)


class RequestTraceRecord(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    thread_id: str = ""
    tenant_id: str = "default"
    acl: list[str] = Field(default_factory=lambda: ["public"])
    question: str
    rewritten_queries: list[str] = Field(default_factory=list)
    route_decision: dict[str, Any] = Field(default_factory=dict)
    coarse_hits: list[dict[str, Any]] = Field(default_factory=list)
    reranked_hits: list[dict[str, Any]] = Field(default_factory=list)
    grounding_score: float | None = None
    terminal_status: str = "success"  # success | business_refusal | hitl_pending | error
    waterfall: list[WaterfallStep] = Field(default_factory=list)
    total_ms: float = 0.0
    model_versions: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat())


class _InMemorySpanCollector:
    """Lightweight OpenTelemetry SpanExporter capturing finished spans for local audit & tests."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult  # type: ignore[import-not-found]

        for sp in spans:
            self.spans.append(
                {
                    "name": sp.name,
                    "trace_id": format(sp.context.trace_id, "032x"),
                    "span_id": format(sp.context.span_id, "016x"),
                    "attributes": dict(sp.attributes or {}),
                    "start_ns": sp.start_time,
                    "end_ns": sp.end_time,
                }
            )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class TraceRecorder:
    """Dual OpenTelemetry + PostgreSQL `request_traces` observability recorder."""

    def __init__(
        self,
        *,
        service_name: str = "llm-graphrag-kg",
        postgres_dsn: str | None = None,
        persist_to_pg: bool = False,
        otlp_endpoint: str | None = None,
    ) -> None:
        self.service_name = service_name
        self.postgres_dsn = postgres_dsn
        self.persist_to_pg = persist_to_pg and bool(postgres_dsn)
        self.otlp_endpoint = otlp_endpoint
        self.local_traces: list[RequestTraceRecord] = []
        self.span_collector: _InMemorySpanCollector | None = None
        self._tracer: Any = None
        self._pg_schema_ready = False
        self._init_otel()

    def ensure_pg_schema(self) -> None:
        if self._pg_schema_ready or not self.postgres_dsn:
            return
        import psycopg  # type: ignore[import-not-found]

        ddl = [
            """
            CREATE TABLE IF NOT EXISTS request_traces (
                request_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL DEFAULT '',
                tenant_id TEXT NOT NULL DEFAULT 'default',
                acl TEXT[] NOT NULL DEFAULT ARRAY['public']::TEXT[],
                question TEXT NOT NULL,
                rewritten_queries TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                route_decision JSONB NOT NULL DEFAULT '{}'::JSONB,
                coarse_hits JSONB NOT NULL DEFAULT '[]'::JSONB,
                reranked_hits JSONB NOT NULL DEFAULT '[]'::JSONB,
                grounding_score DOUBLE PRECISION,
                terminal_status TEXT NOT NULL DEFAULT 'success',
                waterfall JSONB NOT NULL DEFAULT '[]'::JSONB,
                total_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                model_versions JSONB NOT NULL DEFAULT '{}'::JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS thread_id TEXT NOT NULL DEFAULT '';",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS acl TEXT[] NOT NULL DEFAULT ARRAY['public']::TEXT[];",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS route_decision JSONB NOT NULL DEFAULT '{}'::JSONB;",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS coarse_hits JSONB NOT NULL DEFAULT '[]'::JSONB;",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS reranked_hits JSONB NOT NULL DEFAULT '[]'::JSONB;",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS grounding_score DOUBLE PRECISION;",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS terminal_status TEXT NOT NULL DEFAULT 'success';",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS waterfall JSONB NOT NULL DEFAULT '[]'::JSONB;",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS total_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0;",
            "ALTER TABLE request_traces ADD COLUMN IF NOT EXISTS model_versions JSONB NOT NULL DEFAULT '{}'::JSONB;",
        ]
        with psycopg.connect(self.postgres_dsn) as conn, conn.cursor() as cur:
            for stmt in ddl:
                cur.execute(stmt)
            conn.commit()
        self._pg_schema_ready = True

    def _init_otel(self) -> None:
        try:
            from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore[import-not-found]

            provider = TracerProvider(resource=Resource.create({"service.name": self.service_name}))
            collector = _InMemorySpanCollector()
            provider.add_span_processor(SimpleSpanProcessor(collector))
            self.span_collector = collector
            self._tracer = provider.get_tracer(self.service_name)
        except ImportError:
            self._tracer = None

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Any:
        if self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(name) as sp:
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        sp.set_attribute(
                            k,
                            v if isinstance(v, (bool, str, int, float)) else json.dumps(v, ensure_ascii=False),
                        )
            yield sp

    def record(self, trace: RequestTraceRecord) -> None:
        """Emit root OpenTelemetry span and persist to local buffer + optional PostgreSQL `request_traces`."""
        self.local_traces.append(trace)
        with self.span(
            "graphrag.request",
            attributes={
                "rag.request_id": trace.request_id,
                "rag.thread_id": trace.thread_id,
                "rag.tenant_id": trace.tenant_id,
                "rag.terminal_status": trace.terminal_status,
                "rag.grounding_score": trace.grounding_score if trace.grounding_score is not None else -1.0,
                "rag.total_ms": trace.total_ms,
                "rag.model_versions": trace.model_versions,
            },
        ):
            for step in trace.waterfall:
                with self.span(
                    f"graphrag.node.{step.node}",
                    attributes={
                        "rag.node": step.node,
                        "rag.duration_ms": step.duration_ms,
                        "rag.status": step.status,
                    },
                ):
                    pass

        if self.persist_to_pg and self.postgres_dsn:
            self._persist_pg(trace)

    def _persist_pg(self, trace: RequestTraceRecord) -> None:
        import psycopg  # type: ignore[import-not-found]

        self.ensure_pg_schema()
        with psycopg.connect(self.postgres_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO request_traces (
                    request_id, thread_id, tenant_id, acl, question,
                    rewritten_queries, route_decision, coarse_hits, reranked_hits,
                    grounding_score, terminal_status, waterfall, total_ms, model_versions
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s, %s, %s::jsonb, %s, %s::jsonb
                )
                ON CONFLICT (request_id) DO UPDATE SET
                    grounding_score = EXCLUDED.grounding_score,
                    terminal_status = EXCLUDED.terminal_status,
                    waterfall = EXCLUDED.waterfall,
                    total_ms = EXCLUDED.total_ms
                """,
                (
                    trace.request_id,
                    trace.thread_id,
                    trace.tenant_id,
                    trace.acl,
                    trace.question,
                    trace.rewritten_queries,
                    json.dumps(trace.route_decision, ensure_ascii=False),
                    json.dumps(trace.coarse_hits, ensure_ascii=False),
                    json.dumps(trace.reranked_hits, ensure_ascii=False),
                    trace.grounding_score,
                    trace.terminal_status,
                    json.dumps([w.model_dump() for w in trace.waterfall], ensure_ascii=False),
                    trace.total_ms,
                    json.dumps(trace.model_versions, ensure_ascii=False),
                ),
            )
            conn.commit()

    def list_recent_pg_traces(self, *, tenant_id: str | None = None, limit: int = 20) -> list[RequestTraceRecord]:
        """Read recent traces from PostgreSQL `request_traces` for observability APIs/dashboards."""
        if not self.postgres_dsn:
            return list(reversed(self.local_traces))[:limit]
        import psycopg  # type: ignore[import-not-found]

        self.ensure_pg_schema()

        with psycopg.connect(self.postgres_dsn) as conn, conn.cursor() as cur:
            if tenant_id:
                cur.execute(
                    """
                    SELECT request_id, thread_id, tenant_id, acl, question,
                           rewritten_queries, route_decision, coarse_hits, reranked_hits,
                           grounding_score, terminal_status, waterfall, total_ms, model_versions, created_at
                    FROM request_traces
                    WHERE tenant_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT request_id, thread_id, tenant_id, acl, question,
                           rewritten_queries, route_decision, coarse_hits, reranked_hits,
                           grounding_score, terminal_status, waterfall, total_ms, model_versions, created_at
                    FROM request_traces
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
            out: list[RequestTraceRecord] = []
            for r in rows:
                wf_raw = r[11] if isinstance(r[11], list) else []
                out.append(
                    RequestTraceRecord(
                        request_id=str(r[0]),
                        thread_id=str(r[1]),
                        tenant_id=str(r[2]),
                        acl=list(r[3] or []),
                        question=str(r[4]),
                        rewritten_queries=list(r[5] or []),
                        route_decision=dict(r[6] or {}),
                        coarse_hits=list(r[7] or []),
                        reranked_hits=list(r[8] or []),
                        grounding_score=float(r[9]) if r[9] is not None else None,
                        terminal_status=str(r[10]),
                        waterfall=[WaterfallStep(**w) for w in wf_raw if isinstance(w, dict)],
                        total_ms=float(r[12] or 0.0),
                        model_versions=dict(r[13] or {}),
                        created_at=r[14].isoformat() if hasattr(r[14], "isoformat") else str(r[14]),
                    )
                )
            return out

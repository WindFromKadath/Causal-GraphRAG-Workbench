"""Stage 4.2 Asynchronous, Idempotent Incremental Document Ingestion Worker (Redis Streams + PG16 + Neo4j).

Enforces JavaGuide enterprise incremental update laws:
1. Redis Streams (`XADD` / `XREADGROUP` / `XACK`) + Dead Letter Queue (`graphrag:doc_events:dlq`).
2. Strict monotonic version guard (`version_id < active_version` rejected as stale -> zero out-of-order overwrite).
3. Idempotent duplicate delivery (`same version_id + same SHA-256 content_hashes` -> zero redundant writes).
4. SHA-256 `content_hash` incremental embedding reuse (unchanged chunks copy existing `halfvec` -> 0 re-embeds!).
5. Optimistic lease state machine (`pending -> processing (claim_token, claimed_at) -> ready`) staged
   invisibly alongside active version, followed by atomic `documents.active_version` pointer flip.
6. Synchronized PG + Neo4j soft-delete + Redis cache invalidation + periodic Reconciliation (stale lease
   recovery, leaked soft-delete repair, and optional superseded version purge).
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient
from llm_graphrag_kg.pipelines.doc_governance import ChildChunkRecord, govern_document
from llm_graphrag_kg.providers.storage.neo4j_store import Neo4jGraphStore
from llm_graphrag_kg.providers.storage.pgvector_store import PgVectorStore, _vec_to_pg_literal


class DocumentChangeEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    tenant_id: str = "default"
    doc_id: str
    version_id: int = 1
    op: str = "upsert"  # upsert | delete
    source_uri: str = ""
    title: str = ""
    acl: list[str] = Field(default_factory=lambda: ["public"])
    file_path: str | None = None
    raw_text: str | None = None
    attempt: int = 0
    operator: str = "worker"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat())


class ProcessEventResult(BaseModel):
    event_id: str
    tenant_id: str
    doc_id: str
    version_id: int
    op: str
    # applied | idempotent_noop | stale_version_ignored | rejected_quality_gate | soft_deleted | failed_dlq
    outcome: str
    parent_chunks: int = 0
    child_chunks: int = 0
    reused_embeddings: int = 0
    computed_embeddings: int = 0
    neo4j_soft_deleted: dict[str, int] = Field(default_factory=dict)
    detail: str = ""
    elapsed_seconds: float = 0.0


class ReconciliationReport(BaseModel):
    tenant_id: str
    reclaimed_stale_leases: int = 0
    repaired_deleted_chunks: int = 0
    repaired_deleted_neo4j_nodes: int = 0
    purged_superseded_chunks: int = 0
    active_documents: int = 0
    dlq_length: int = 0


class IncrementalIngestionWorker:
    """Enterprise Redis Streams + PostgreSQL 16 + Neo4j 5.26 idempotent incremental worker."""

    def __init__(
        self,
        pg_store: PgVectorStore,
        embedder: BaseEmbeddingClient,
        *,
        neo4j_store: Neo4jGraphStore | None = None,
        redis_url: str = "redis://127.0.0.1:26379/0",
        stream_key: str = "graphrag:doc_events",
        group_name: str = "graphrag_workers",
        dlq_key: str = "graphrag:doc_events:dlq",
        consumer_name: str | None = None,
        max_retries: int = 2,
        redis_client: Any = None,
    ) -> None:
        self.pg_store = pg_store
        self.embedder = embedder
        self.neo4j_store = neo4j_store
        self.redis_url = redis_url
        self.stream_key = stream_key
        self.group_name = group_name
        self.dlq_key = dlq_key
        self.consumer_name = consumer_name or f"worker_{uuid.uuid4().hex[:6]}"
        self.max_retries = max(1, int(max_retries))
        self._redis = redis_client

    def _get_redis(self) -> Any:
        if self._redis is not None:
            return self._redis
        import redis  # type: ignore[import-not-found]

        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def ensure_stream_group(self) -> None:
        r = self._get_redis()
        try:
            r.xgroup_create(name=self.stream_key, groupname=self.group_name, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001
            if "BUSYGROUP" not in str(exc):
                raise

    def enqueue_event(self, event: DocumentChangeEvent) -> str:
        """Publish a document change event into Redis Streams (`XADD`)."""
        r = self._get_redis()
        msg_id = r.xadd(
            self.stream_key,
            {"payload": event.model_dump_json()},
        )
        return str(msg_id)

    def list_dlq(self, limit: int = 20) -> list[dict[str, Any]]:
        """Inspect recent Dead Letter Queue entries."""
        r = self._get_redis()
        entries = r.xrevrange(self.dlq_key, count=limit)
        out: list[dict[str, Any]] = []
        for msg_id, fields in entries:
            payload_raw = fields.get("payload", "{}")
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                payload = {"raw": payload_raw}
            out.append(
                {
                    "stream_id": str(msg_id),
                    "error": str(fields.get("error", "")),
                    "failed_at": str(fields.get("failed_at", "")),
                    "event": payload,
                }
            )
        return out

    def _invalidate_tenant_cache(self, tenant_id: str) -> int:
        """Best-effort旁路 cache invalidation for tenant query/rerank caches in Redis."""
        try:
            r = self._get_redis()
            keys = list(r.scan_iter(match=f"graphrag:cache:{tenant_id}:*", count=200))
            if keys:
                return int(r.delete(*keys))
            return 0
        except Exception:  # noqa: BLE001
            return 0

    def _log_audit(
        self,
        cur: Any,
        *,
        tenant_id: str,
        doc_id: str,
        change_type: str,
        operator: str,
        result: str,
        error_message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        cur.execute(
            """
            INSERT INTO audit_logs (tenant_id, doc_id, change_type, operator, result, error_message, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                tenant_id,
                doc_id,
                change_type,
                operator,
                result,
                error_message[:1000],
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )

    def _handle_delete(self, event: DocumentChangeEvent, t0: float) -> ProcessEventResult:
        tid = event.tenant_id
        # Collect chunk_ids before soft-deleting so Neo4j can drop exact provenance
        chunk_ids: list[str] = []
        with self.pg_store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM child_chunks WHERE tenant_id = %s AND doc_id = %s",
                (tid, event.doc_id),
            )
            chunk_ids = [str(r[0]) for r in cur.fetchall()]

        pg_affected = self.pg_store.soft_delete_document(
            event.doc_id,
            tenant_id=tid,
            operator=event.operator,
        )
        neo_stats: dict[str, int] = {}
        if self.neo4j_store is not None:
            neo_stats = self.neo4j_store.soft_delete_document(
                event.doc_id,
                chunk_ids=chunk_ids,
                tenant_id=tid,
            )
        self._invalidate_tenant_cache(tid)
        return ProcessEventResult(
            event_id=event.event_id,
            tenant_id=tid,
            doc_id=event.doc_id,
            version_id=event.version_id,
            op="delete",
            outcome="soft_deleted",
            child_chunks=pg_affected,
            neo4j_soft_deleted=neo_stats,
            detail=f"Soft-deleted {pg_affected} PG chunk(s) and synchronized Neo4j ({neo_stats}).",
            elapsed_seconds=round(time.perf_counter() - t0, 3),
        )

    def process_event(self, event: DocumentChangeEvent) -> ProcessEventResult:
        """Execute idempotent, version-guarded incremental upsert or synchronized soft-delete."""
        t0 = time.perf_counter()
        if event.op.strip().lower() == "delete":
            return self._handle_delete(event, t0)

        tid = event.tenant_id
        fp = Path(event.file_path) if event.file_path else None
        gov = govern_document(
            doc_id=event.doc_id,
            file_path=fp,
            raw_text=event.raw_text,
            title=event.title or None,
            source_uri=event.source_uri or None,
            version_id=event.version_id,
            acl=event.acl,
        )
        if gov.status == "rejected":
            with self.pg_store._connect() as conn, conn.cursor() as cur:
                self._log_audit(
                    cur,
                    tenant_id=tid,
                    doc_id=event.doc_id,
                    change_type="quality_gate_rejected",
                    operator=event.operator,
                    result="rejected",
                    error_message=gov.rejection_reason or "Quality gate failed",
                    payload={"version_id": event.version_id, "gates": [g.model_dump() for g in gov.gates]},
                )
                conn.commit()
            return ProcessEventResult(
                event_id=event.event_id,
                tenant_id=tid,
                doc_id=event.doc_id,
                version_id=event.version_id,
                op="upsert",
                outcome="rejected_quality_gate",
                detail=gov.rejection_reason or "Rejected by document quality gates.",
                elapsed_seconds=round(time.perf_counter() - t0, 3),
            )

        dim = self.pg_store.embedding_dim
        eff_model = self.pg_store.embedding_model
        self.pg_store.register_or_verify_index(embedding_model=eff_model, embedding_dim=dim)

        # Phase A: Monotonic version guard, idempotency check, existing hash lookup & invisible staging
        claim_token = f"lease_{uuid.uuid4().hex[:12]}"
        hash_to_vec_lit: dict[str, str] = {}
        with self.pg_store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT active_version, status, is_deleted
                FROM documents
                WHERE tenant_id = %s AND doc_id = %s
                FOR UPDATE
                """,
                (tid, event.doc_id),
            )
            doc_row = cur.fetchone()
            if doc_row is not None:
                cur_active_ver, cur_status, cur_deleted = int(doc_row[0]), str(doc_row[1]), bool(doc_row[2])
                if not cur_deleted and event.version_id < cur_active_ver:
                    self._log_audit(
                        cur,
                        tenant_id=tid,
                        doc_id=event.doc_id,
                        change_type="stale_version_ignored",
                        operator=event.operator,
                        result="skipped",
                        payload={"incoming_version": event.version_id, "active_version": cur_active_ver},
                    )
                    conn.commit()
                    return ProcessEventResult(
                        event_id=event.event_id,
                        tenant_id=tid,
                        doc_id=event.doc_id,
                        version_id=event.version_id,
                        op="upsert",
                        outcome="stale_version_ignored",
                        detail=f"Ignored stale version v{event.version_id} (< active v{cur_active_ver}).",
                        elapsed_seconds=round(time.perf_counter() - t0, 3),
                    )
                if (
                    not cur_deleted
                    and event.version_id == cur_active_ver
                    and cur_status in {"ready", "partial_available"}
                ):
                    cur.execute(
                        """
                        SELECT content_hash
                        FROM child_chunks
                        WHERE tenant_id = %s AND doc_id = %s AND version_id = %s
                          AND index_status = 'ready' AND is_deleted = FALSE
                        ORDER BY chunk_index
                        """,
                        (tid, event.doc_id, event.version_id),
                    )
                    existing_hashes = [str(r[0]) for r in cur.fetchall()]
                    incoming_hashes = [c.content_hash for c in gov.child_chunks]
                    if existing_hashes == incoming_hashes:
                        self._log_audit(
                            cur,
                            tenant_id=tid,
                            doc_id=event.doc_id,
                            change_type="idempotent_noop",
                            operator=event.operator,
                            result="skipped",
                            payload={"version_id": event.version_id, "chunks": len(incoming_hashes)},
                        )
                        conn.commit()
                        return ProcessEventResult(
                            event_id=event.event_id,
                            tenant_id=tid,
                            doc_id=event.doc_id,
                            version_id=event.version_id,
                            op="upsert",
                            outcome="idempotent_noop",
                            parent_chunks=len(gov.parent_chunks),
                            child_chunks=len(gov.child_chunks),
                            reused_embeddings=len(gov.child_chunks),
                            computed_embeddings=0,
                            detail=f"Duplicate event for v{event.version_id} with identical hashes; skipped.",
                            elapsed_seconds=round(time.perf_counter() - t0, 3),
                        )
            else:
                # Brand new document: keep active_version=0 ('building') so queries don't see partial state
                cur.execute(
                    """
                    INSERT INTO documents (
                        doc_id, tenant_id, acl, source_type, source_uri, title,
                        active_version, status, is_deleted, missing_pages
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, 0, 'building', FALSE, %s)
                    """,
                    (
                        event.doc_id,
                        tid,
                        event.acl,
                        gov.source_type,
                        gov.source_uri,
                        gov.title,
                        gov.missing_pages,
                    ),
                )

            # Pull reusable SHA-256 content_hash -> embedding::text for unchanged chunks (same model & dim!)
            cur.execute(
                """
                SELECT content_hash, embedding::text
                FROM child_chunks
                WHERE tenant_id = %s
                  AND doc_id = %s
                  AND embedding IS NOT NULL
                  AND index_status = 'ready'
                  AND embedding_model = %s
                  AND embedding_dim = %s
                """,
                (tid, event.doc_id, eff_model, dim),
            )
            for h_val, vec_txt in cur.fetchall():
                if h_val and vec_txt:
                    hash_to_vec_lit[str(h_val)] = str(vec_txt)

            # Stage Parent chunks (invisible until active_version flips)
            for parent in gov.parent_chunks:
                cur.execute(
                    """
                    INSERT INTO parent_chunks (
                        parent_id, tenant_id, doc_id, version_id, acl,
                        section_path, page_no, element_type, content, token_count, is_deleted, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s::jsonb)
                    ON CONFLICT (parent_id) DO UPDATE SET
                        acl = EXCLUDED.acl,
                        section_path = EXCLUDED.section_path,
                        page_no = EXCLUDED.page_no,
                        element_type = EXCLUDED.element_type,
                        content = EXCLUDED.content,
                        token_count = EXCLUDED.token_count,
                        is_deleted = FALSE,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        parent.parent_id,
                        tid,
                        parent.doc_id,
                        event.version_id,
                        parent.acl,
                        parent.section_path,
                        parent.page_no,
                        parent.element_type,
                        parent.content,
                        parent.token_count,
                        json.dumps(parent.metadata, ensure_ascii=False),
                    ),
                )

            # Stage Child chunks as 'pending' and atomically claim with optimistic lease
            for ch in gov.child_chunks:
                cur.execute(
                    """
                    INSERT INTO child_chunks (
                        chunk_id, parent_id, tenant_id, doc_id, version_id, chunk_index,
                        start_char, end_char, source, section_path, acl,
                        content_hash, text, breadcrumb_text,
                        embedding_model, embedding_dim,
                        index_status, claim_token, claimed_at, is_deleted, metadata
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        'pending', NULL, NULL, FALSE, %s::jsonb
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        parent_id = EXCLUDED.parent_id,
                        acl = EXCLUDED.acl,
                        section_path = EXCLUDED.section_path,
                        content_hash = EXCLUDED.content_hash,
                        text = EXCLUDED.text,
                        breadcrumb_text = EXCLUDED.breadcrumb_text,
                        index_status = 'pending',
                        claim_token = NULL,
                        claimed_at = NULL,
                        is_deleted = FALSE,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    (
                        ch.chunk_id,
                        ch.parent_id,
                        tid,
                        ch.doc_id,
                        event.version_id,
                        ch.chunk_index,
                        ch.start_char,
                        ch.end_char,
                        ch.source,
                        ch.section_path,
                        ch.acl,
                        ch.content_hash,
                        ch.text,
                        ch.breadcrumb_text,
                        eff_model,
                        dim,
                        json.dumps(ch.metadata, ensure_ascii=False),
                    ),
                )

            # Optimistic lease claim: pending -> processing
            cur.execute(
                """
                UPDATE child_chunks
                SET index_status = 'processing', claim_token = %s, claimed_at = NOW(), updated_at = NOW()
                WHERE tenant_id = %s AND doc_id = %s AND version_id = %s AND index_status = 'pending'
                """,
                (claim_token, tid, event.doc_id, event.version_id),
            )
            conn.commit()

        # Phase B: Compute embeddings ONLY for genuinely changed/new content_hashes!
        to_compute: list[ChildChunkRecord] = []
        reused_map: dict[str, str] = {}
        for ch in gov.child_chunks:
            cached = hash_to_vec_lit.get(ch.content_hash)
            if cached is not None:
                reused_map[ch.chunk_id] = cached
            else:
                to_compute.append(ch)

        computed_map: dict[str, str] = {}
        if to_compute:
            vecs = self.embedder.embed([c.breadcrumb_text for c in to_compute])
            if len(vecs) != len(to_compute) or (vecs and len(vecs[0]) != dim):
                raise ValueError(
                    f"Embedding count/dim mismatch: got {len(vecs)} vectors (dim={len(vecs[0]) if vecs else 0}), "
                    f"expected {len(to_compute)} x {dim}."
                )
            for ch, vec in zip(to_compute, vecs, strict=True):
                computed_map[ch.chunk_id] = _vec_to_pg_literal(vec)

        # Phase C: Write vectors for leased chunks and atomically flip `documents.active_version`
        with self.pg_store._connect() as conn, conn.cursor() as cur:
            for ch in gov.child_chunks:
                vec_lit = reused_map.get(ch.chunk_id) or computed_map[ch.chunk_id]
                cur.execute(
                    f"""
                    UPDATE child_chunks
                    SET embedding = %s::halfvec({dim}),
                        embedding_model = %s,
                        embedding_dim = %s,
                        index_status = 'ready',
                        claim_token = NULL,
                        claimed_at = NULL,
                        updated_at = NOW()
                    WHERE chunk_id = %s AND tenant_id = %s AND claim_token = %s
                    """,
                    (vec_lit, eff_model, dim, ch.chunk_id, tid, claim_token),
                )

            # Verify every chunk in this version is ready before atomic pointer flip
            cur.execute(
                """
                SELECT COUNT(*) FROM child_chunks
                WHERE tenant_id = %s AND doc_id = %s AND version_id = %s
                  AND index_status != 'ready'
                """,
                (tid, event.doc_id, event.version_id),
            )
            unready = int((cur.fetchone() or (0,))[0])
            if unready > 0:
                raise RuntimeError(
                    f"Cannot flip active_version for {event.doc_id} v{event.version_id}: "
                    f"{unready} chunk(s) not ready (lease conflict or partial failure)."
                )

            cur.execute(
                """
                UPDATE documents
                SET acl = %s,
                    source_type = %s,
                    source_uri = %s,
                    title = %s,
                    active_version = %s,
                    status = %s,
                    is_deleted = FALSE,
                    missing_pages = %s,
                    updated_at = NOW()
                WHERE tenant_id = %s AND doc_id = %s
                """,
                (
                    event.acl,
                    gov.source_type,
                    gov.source_uri,
                    gov.title,
                    event.version_id,
                    gov.status,
                    gov.missing_pages,
                    tid,
                    event.doc_id,
                ),
            )
            self._log_audit(
                cur,
                tenant_id=tid,
                doc_id=event.doc_id,
                change_type="upsert_version",
                operator=event.operator,
                result="success",
                payload={
                    "version_id": event.version_id,
                    "parser_used": gov.parser_used,
                    "status": gov.status,
                    "parent_chunks": len(gov.parent_chunks),
                    "child_chunks": len(gov.child_chunks),
                    "reused_embeddings": len(reused_map),
                    "computed_embeddings": len(computed_map),
                    "dropped_chunks": gov.dropped_chunks,
                },
            )
            conn.commit()

        self._invalidate_tenant_cache(tid)
        return ProcessEventResult(
            event_id=event.event_id,
            tenant_id=tid,
            doc_id=event.doc_id,
            version_id=event.version_id,
            op="upsert",
            outcome="applied",
            parent_chunks=len(gov.parent_chunks),
            child_chunks=len(gov.child_chunks),
            reused_embeddings=len(reused_map),
            computed_embeddings=len(computed_map),
            detail=(
                f"Activated {event.doc_id} v{event.version_id} ({gov.parser_used}, {gov.status}): "
                f"{len(gov.parent_chunks)} parents / {len(gov.child_chunks)} children "
                f"(reused={len(reused_map)}, computed={len(computed_map)})."
            ),
            elapsed_seconds=round(time.perf_counter() - t0, 3),
        )

    def consume_batch(self, *, max_messages: int = 10, block_ms: int = 500) -> list[ProcessEventResult]:
        """Consume up to `max_messages` from Redis Streams Consumer Group with retry & DLQ routing."""
        self.ensure_stream_group()
        r = self._get_redis()
        results: list[ProcessEventResult] = []
        batches = r.xreadgroup(
            groupname=self.group_name,
            consumername=self.consumer_name,
            streams={self.stream_key: ">"},
            count=max_messages,
            block=block_ms,
        )
        if not batches:
            return results

        for _stream, messages in batches:
            for msg_id, fields in messages:
                raw_payload = fields.get("payload", "{}")
                try:
                    event = DocumentChangeEvent.model_validate_json(raw_payload)
                except Exception as exc:  # noqa: BLE001
                    r.xadd(
                        self.dlq_key,
                        {
                            "payload": raw_payload,
                            "error": f"Malformed event payload: {exc}",
                            "failed_at": datetime.now(UTC).isoformat(),
                        },
                    )
                    r.xack(self.stream_key, self.group_name, msg_id)
                    continue

                try:
                    res = self.process_event(event)
                    r.xack(self.stream_key, self.group_name, msg_id)
                    results.append(res)
                except Exception as exc:  # noqa: BLE001
                    r.xack(self.stream_key, self.group_name, msg_id)
                    if event.attempt + 1 < self.max_retries:
                        retry_evt = event.model_copy(update={"attempt": event.attempt + 1})
                        self.enqueue_event(retry_evt)
                    else:
                        r.xadd(
                            self.dlq_key,
                            {
                                "payload": event.model_dump_json(),
                                "error": str(exc)[:1000],
                                "failed_at": datetime.now(UTC).isoformat(),
                            },
                        )
                        with self.pg_store._connect() as conn, conn.cursor() as cur:
                            self._log_audit(
                                cur,
                                tenant_id=event.tenant_id,
                                doc_id=event.doc_id,
                                change_type="dlq_routed",
                                operator=self.consumer_name,
                                result="failed",
                                error_message=str(exc),
                                payload={"event_id": event.event_id, "attempt": event.attempt + 1},
                            )
                            conn.commit()
                        results.append(
                            ProcessEventResult(
                                event_id=event.event_id,
                                tenant_id=event.tenant_id,
                                doc_id=event.doc_id,
                                version_id=event.version_id,
                                op=event.op,
                                outcome="failed_dlq",
                                detail=f"Routed to DLQ after {event.attempt + 1} attempts: {exc}",
                            )
                        )
        return results

    def reconcile(
        self,
        *,
        tenant_id: str | None = None,
        stale_lease_seconds: int = 300,
        purge_superseded: bool = False,
    ) -> ReconciliationReport:
        """Periodic reconciliation: reclaim expired leases, fix missed soft-deletes, purge superseded versions."""
        tid = tenant_id or self.pg_store.default_tenant_id
        reclaimed = 0
        repaired_pg = 0
        repaired_neo = 0
        purged = 0
        active_docs = 0

        with self.pg_store._connect() as conn, conn.cursor() as cur:
            # 1. Reclaim expired 'processing' leases back to 'pending'
            cur.execute(
                """
                UPDATE child_chunks
                SET index_status = 'pending', claim_token = NULL, claimed_at = NULL, updated_at = NOW()
                WHERE tenant_id = %s
                  AND index_status = 'processing'
                  AND claimed_at < NOW() - make_interval(secs => %s)
                """,
                (tid, max(0, int(stale_lease_seconds))),
            )
            reclaimed = int(cur.rowcount)

            # 2. Repair any soft-deleted document whose chunks are still marked is_deleted = FALSE
            cur.execute(
                """
                UPDATE child_chunks c
                SET is_deleted = TRUE, updated_at = NOW()
                FROM documents d
                WHERE c.tenant_id = d.tenant_id
                  AND c.doc_id = d.doc_id
                  AND d.tenant_id = %s
                  AND d.is_deleted = TRUE
                  AND c.is_deleted = FALSE
                """,
                (tid,),
            )
            repaired_pg = int(cur.rowcount)

            # Collect deleted doc_ids for Neo4j reconciliation
            cur.execute(
                "SELECT doc_id FROM documents WHERE tenant_id = %s AND is_deleted = TRUE",
                (tid,),
            )
            deleted_doc_ids = [str(r[0]) for r in cur.fetchall()]

            # 3. Optional delayed purge of superseded historical versions
            if purge_superseded:
                cur.execute(
                    """
                    DELETE FROM child_chunks c
                    USING documents d
                    WHERE c.tenant_id = d.tenant_id
                      AND c.doc_id = d.doc_id
                      AND d.tenant_id = %s
                      AND c.version_id < d.active_version
                    """,
                    (tid,),
                )
                purged = int(cur.rowcount)
                cur.execute(
                    """
                    DELETE FROM parent_chunks p
                    USING documents d
                    WHERE p.tenant_id = d.tenant_id
                      AND p.doc_id = d.doc_id
                      AND d.tenant_id = %s
                      AND p.version_id < d.active_version
                    """,
                    (tid,),
                )

            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE tenant_id = %s AND is_deleted = FALSE",
                (tid,),
            )
            active_docs = int((cur.fetchone() or (0,))[0])

            # 4. Repair any leaked Neo4j nodes for deleted documents
            if self.neo4j_store is not None:
                for did in deleted_doc_ids:
                    stats = self.neo4j_store.soft_delete_document(did, tenant_id=tid)
                    repaired_neo += stats.get("entities_soft_deleted", 0)

            self._log_audit(
                cur,
                tenant_id=tid,
                doc_id="*",
                change_type="reconciliation",
                operator=self.consumer_name,
                result="success",
                payload={
                    "reclaimed_stale_leases": reclaimed,
                    "repaired_deleted_chunks": repaired_pg,
                    "repaired_deleted_neo4j_nodes": repaired_neo,
                    "purged_superseded_chunks": purged,
                },
            )
            conn.commit()

        dlq_len = 0
        try:
            dlq_len = int(self._get_redis().xlen(self.dlq_key))
        except Exception:  # noqa: BLE001
            dlq_len = 0

        return ReconciliationReport(
            tenant_id=tid,
            reclaimed_stale_leases=reclaimed,
            repaired_deleted_chunks=repaired_pg,
            repaired_deleted_neo4j_nodes=repaired_neo,
            purged_superseded_chunks=purged,
            active_documents=active_docs,
            dlq_length=dlq_len,
        )

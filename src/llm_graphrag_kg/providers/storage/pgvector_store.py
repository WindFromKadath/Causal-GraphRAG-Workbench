"""PostgreSQL 16 + pgvector 0.8 enterprise vector, metadata & A-MEM store (Stage 4.1).

Implements `BaseVectorStore` plus enterprise production guarantees:
- `halfvec(dim)` HNSW cosine indexing (-50% RAM vs float32) + `hnsw.iterative_scan`
- Pre-Filter multi-tenant (`tenant_id`) + RBAC (`acl && $roles`) + soft-delete (`is_deleted = FALSE`)
- Hybrid Dense + Lexical (`tsvector` + Chinese token fallback) RRF fusion
- Hard Embedding Model Fingerprint guard (`index_registry`) preventing silent vector-space mixing
- `PgVerifiedMemoryStore` persisting Stage 3.4 A-MEM records in PostgreSQL `verified_memories`
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_graphrag_kg.core.interfaces import BaseGraphStore
from llm_graphrag_kg.core.schema import TextChunk, VerifiedMemoryRecord
from llm_graphrag_kg.eval.llm_judge import detect_refusal
from llm_graphrag_kg.pipelines.hypergraph import create_hyperedge
from llm_graphrag_kg.pipelines.memory_store import (
    WriteBackGateDecision,
    check_memory_conflict,
)
from llm_graphrag_kg.providers.storage.local_vector import tokenize_bm25


def _import_psycopg() -> Any:
    try:
        import psycopg  # type: ignore[import-not-found]

        return psycopg
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL backend requires enterprise dependencies. "
            "Run: `uv sync --extra enterprise`."
        ) from exc


def _vec_to_pg_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.6g}" for v in vec) + "]"


def _parse_roles(roles: list[str] | str | None, default_roles: list[str]) -> list[str]:
    if roles is None:
        return list(default_roles)
    if isinstance(roles, str):
        parsed = [r.strip() for r in roles.split(",") if r.strip()]
        return parsed or list(default_roles)
    return [r.strip() for r in roles if r.strip()] or list(default_roles)


class PgVectorStore:
    """Enterprise PostgreSQL 16 + pgvector 0.8 vector & hybrid store implementing `BaseVectorStore`."""

    def __init__(
        self,
        dsn: str,
        *,
        embedding_dim: int = 1024,
        embedding_model: str = "qwen3-embedding:0.6b",
        ef_search: int = 80,
        iterative_scan: str = "strict_order",
        default_tenant_id: str = "default",
        default_acl_roles: list[str] | str = "public,internal",
        auto_init_schema: bool = True,
    ) -> None:
        self.dsn = dsn
        self.embedding_dim = int(embedding_dim)
        self.embedding_model = embedding_model
        self.ef_search = max(10, int(ef_search))
        self.iterative_scan = iterative_scan
        self.default_tenant_id = default_tenant_id
        self.default_acl_roles = _parse_roles(default_acl_roles, ["public", "internal"])
        self._psycopg = _import_psycopg()
        if auto_init_schema:
            self.ensure_schema()

    def _connect(self) -> Any:
        return self._psycopg.connect(self.dsn)

    def health_check(self) -> tuple[bool, str]:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT version(), extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cur.fetchone()
                if not row:
                    return False, "PostgreSQL reachable, but `vector` extension is not installed."
                pg_ver = str(row[0]).split(",")[0]
                vec_ver = str(row[1])
                return True, f"PostgreSQL ready ({pg_ver} | pgvector {vec_ver} | dim={self.embedding_dim})"
        except Exception as exc:  # noqa: BLE001
            return False, f"PostgreSQL health check failed: {exc}"

    def ensure_schema(self) -> None:
        """Idempotently bootstrap PostgreSQL extensions, tables, and HNSW/GIN indexes."""
        dim = self.embedding_dim
        ddl_statements = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
            """
            CREATE TABLE IF NOT EXISTS index_registry (
                index_name TEXT PRIMARY KEY,
                active_table TEXT NOT NULL DEFAULT 'child_chunks',
                embedding_model TEXT NOT NULL,
                embedding_version TEXT NOT NULL DEFAULT 'v1',
                embedding_dim INTEGER NOT NULL,
                chunk_strategy_version TEXT NOT NULL DEFAULT 'parent_child_v1',
                status TEXT NOT NULL DEFAULT 'active',
                switched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                acl TEXT[] NOT NULL DEFAULT ARRAY['public']::TEXT[],
                source_type TEXT NOT NULL DEFAULT 'markdown',
                source_uri TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                active_version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'ready',
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                missing_pages INTEGER[] NOT NULL DEFAULT ARRAY[]::INTEGER[],
                metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (tenant_id, doc_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                doc_id TEXT NOT NULL,
                version_id INTEGER NOT NULL DEFAULT 1,
                acl TEXT[] NOT NULL DEFAULT ARRAY['public']::TEXT[],
                section_path TEXT NOT NULL DEFAULT '',
                page_no INTEGER,
                element_type TEXT NOT NULL DEFAULT 'text',
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS child_chunks (
                chunk_id TEXT PRIMARY KEY,
                parent_id TEXT REFERENCES parent_chunks(parent_id) ON DELETE SET NULL,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                doc_id TEXT NOT NULL,
                version_id INTEGER NOT NULL DEFAULT 1,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                start_char INTEGER NOT NULL DEFAULT 0,
                end_char INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT '',
                section_path TEXT NOT NULL DEFAULT '',
                acl TEXT[] NOT NULL DEFAULT ARRAY['public']::TEXT[],
                content_hash TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                breadcrumb_text TEXT NOT NULL DEFAULT '',
                embedding halfvec({dim}),
                embedding_model TEXT NOT NULL DEFAULT 'qwen3-embedding:0.6b',
                embedding_dim INTEGER NOT NULL DEFAULT {dim},
                tsv TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('simple', coalesce(section_path, '') || ' ' || coalesce(text, ''))
                ) STORED,
                index_status TEXT NOT NULL DEFAULT 'ready',
                claim_token TEXT,
                claimed_at TIMESTAMPTZ,
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                metadata JSONB NOT NULL DEFAULT '{{}}'::JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_child_chunks_hnsw_cosine
                ON child_chunks USING hnsw (embedding halfvec_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """,
            "CREATE INDEX IF NOT EXISTS idx_child_chunks_tsv_gin ON child_chunks USING GIN (tsv);",
            "CREATE INDEX IF NOT EXISTS idx_child_chunks_acl_gin ON child_chunks USING GIN (acl);",
            """
            CREATE INDEX IF NOT EXISTS idx_child_chunks_filter
                ON child_chunks (tenant_id, is_deleted, index_status, doc_id, version_id);
            """,
            """
            CREATE TABLE IF NOT EXISTS verified_memories (
                memory_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                acl TEXT[] NOT NULL DEFAULT ARRAY['public']::TEXT[],
                question TEXT NOT NULL,
                conclusion TEXT NOT NULL,
                condition_entity_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                target_entity_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                source_chunk_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                grounding_score DOUBLE PRECISION NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                reasoning_paradigm TEXT NOT NULL DEFAULT 'auto',
                valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                provenance_run_id TEXT NOT NULL,
                conflict_checked BOOLEAN NOT NULL DEFAULT TRUE,
                status TEXT NOT NULL DEFAULT 'verified',
                metadata JSONB NOT NULL DEFAULT '{}'::JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id BIGSERIAL PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                doc_id TEXT NOT NULL DEFAULT '',
                change_type TEXT NOT NULL,
                operator TEXT NOT NULL DEFAULT 'system',
                result TEXT NOT NULL DEFAULT 'success',
                error_message TEXT NOT NULL DEFAULT '',
                payload JSONB NOT NULL DEFAULT '{}'::JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
        ]
        with self._connect() as conn, conn.cursor() as cur:
            for stmt in ddl_statements:
                cur.execute(stmt)
            conn.commit()

    def register_or_verify_index(
        self,
        *,
        index_name: str = "primary",
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        allow_update: bool = False,
    ) -> dict[str, Any]:
        """Enforce embedding model & dimension consistency against `index_registry`."""
        model = embedding_model or self.embedding_model
        dim = int(embedding_dim or self.embedding_dim)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT embedding_model, embedding_dim, active_table, status "
                "FROM index_registry WHERE index_name = %s",
                (index_name,),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """
                    INSERT INTO index_registry (index_name, embedding_model, embedding_dim, status)
                    VALUES (%s, %s, %s, 'active')
                    """,
                    (index_name, model, dim),
                )
                conn.commit()
                return {
                    "index_name": index_name,
                    "embedding_model": model,
                    "embedding_dim": dim,
                    "status": "registered",
                }
            reg_model, reg_dim, active_tbl, status = row
            if (reg_model != model or int(reg_dim) != dim) and not allow_update:
                raise ValueError(
                    f"Embedding model mismatch on index `{index_name}`: "
                    f"registry has ({reg_model}, dim={reg_dim}) vs runtime ({model}, dim={dim}). "
                    "Use blue-green full rebuild or pass allow_update=True after re-indexing."
                )
            if allow_update and (reg_model != model or int(reg_dim) != dim):
                cur.execute(
                    """
                    UPDATE index_registry
                    SET embedding_model = %s, embedding_dim = %s, switched_at = NOW()
                    WHERE index_name = %s
                    """,
                    (model, dim, index_name),
                )
                conn.commit()
            return {
                "index_name": index_name,
                "active_table": active_tbl,
                "embedding_model": model,
                "embedding_dim": dim,
                "status": status,
            }

    def add_chunks(
        self,
        chunks: list[TextChunk],
        vectors: list[list[float]],
        *,
        tenant_id: str | None = None,
        acl: list[str] | None = None,
        version_id: int = 1,
        embedding_model: str | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")
        if not chunks:
            return

        tid = tenant_id or self.default_tenant_id
        eff_model = embedding_model or self.embedding_model
        eff_dim = len(vectors[0])
        if eff_dim != self.embedding_dim:
            raise ValueError(
                f"Vector dim ({eff_dim}) != configured PgVectorStore dim ({self.embedding_dim})."
            )
        self.register_or_verify_index(embedding_model=eff_model, embedding_dim=eff_dim)

        with self._connect() as conn, conn.cursor() as cur:
            seen_docs: set[str] = set()
            for ch, vec in zip(chunks, vectors, strict=True):
                ch_acl = _parse_roles(
                    acl if acl is not None else ch.metadata.get("acl"),
                    self.default_acl_roles,
                )
                if ch.doc_id not in seen_docs:
                    cur.execute(
                        """
                        INSERT INTO documents (
                            doc_id, tenant_id, acl, source_uri, title, active_version, status, is_deleted
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, 'ready', FALSE)
                        ON CONFLICT (tenant_id, doc_id) DO UPDATE SET
                            acl = EXCLUDED.acl,
                            source_uri = EXCLUDED.source_uri,
                            active_version = EXCLUDED.active_version,
                            status = 'ready',
                            is_deleted = FALSE,
                            updated_at = NOW()
                        """,
                        (ch.doc_id, tid, ch_acl, ch.source, Path(ch.source).name, version_id),
                    )
                    seen_docs.add(ch.doc_id)

                content_hash = hashlib.sha256(ch.text.encode()).hexdigest()
                section_path = str(ch.metadata.get("section_path", ""))
                breadcrumb_text = str(ch.metadata.get("breadcrumb_text", ch.text))
                vec_lit = _vec_to_pg_literal(vec)
                meta_json = json.dumps(ch.metadata, ensure_ascii=False)

                cur.execute(
                    f"""
                    INSERT INTO child_chunks (
                        chunk_id, tenant_id, doc_id, version_id, chunk_index,
                        start_char, end_char, source, section_path, acl,
                        content_hash, text, breadcrumb_text,
                        embedding, embedding_model, embedding_dim,
                        index_status, is_deleted, metadata
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s::halfvec({eff_dim}), %s, %s,
                        'ready', FALSE, %s::jsonb
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        tenant_id = EXCLUDED.tenant_id,
                        version_id = EXCLUDED.version_id,
                        acl = EXCLUDED.acl,
                        section_path = EXCLUDED.section_path,
                        content_hash = EXCLUDED.content_hash,
                        text = EXCLUDED.text,
                        breadcrumb_text = EXCLUDED.breadcrumb_text,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dim = EXCLUDED.embedding_dim,
                        index_status = 'ready',
                        is_deleted = FALSE,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    (
                        ch.chunk_id,
                        tid,
                        ch.doc_id,
                        version_id,
                        ch.index,
                        ch.start_char,
                        ch.end_char,
                        ch.source,
                        section_path,
                        ch_acl,
                        content_hash,
                        ch.text,
                        breadcrumb_text,
                        vec_lit,
                        eff_model,
                        eff_dim,
                        meta_json,
                    ),
                )
            conn.commit()

    def _apply_hnsw_session_params(self, cur: Any) -> None:
        cur.execute(f"SET LOCAL hnsw.ef_search = {self.ef_search}")
        if self.iterative_scan in {"strict_order", "relaxed_order"}:
            cur.execute("SAVEPOINT sp_iter_scan")
            try:
                cur.execute(f"SET LOCAL hnsw.iterative_scan = {self.iterative_scan}")
                cur.execute("RELEASE SAVEPOINT sp_iter_scan")
            except Exception:  # noqa: BLE001
                cur.execute("ROLLBACK TO SAVEPOINT sp_iter_scan")

    @staticmethod
    def _row_to_chunk(row: tuple[Any, ...]) -> TextChunk:
        chunk_id, doc_id, source, idx, start_c, end_c, text, meta, acl, sec_path = row
        meta_dict = dict(meta) if isinstance(meta, dict) else {}
        if acl:
            meta_dict.setdefault("acl", list(acl))
        if sec_path:
            meta_dict.setdefault("section_path", sec_path)
        return TextChunk(
            chunk_id=str(chunk_id),
            doc_id=str(doc_id),
            source=str(source),
            index=int(idx),
            text=str(text),
            start_char=int(start_c),
            end_char=int(end_c),
            metadata=meta_dict,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[tuple[TextChunk, float]]:
        if not query_vector:
            return []
        if len(query_vector) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dim ({len(query_vector)}) != PgVectorStore dim ({self.embedding_dim})."
            )
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        vec_lit = _vec_to_pg_literal(query_vector)
        dim = self.embedding_dim

        with self._connect() as conn, conn.cursor() as cur:
            self._apply_hnsw_session_params(cur)
            cur.execute(
                f"""
                SELECT c.chunk_id, c.doc_id, c.source, c.chunk_index, c.start_char, c.end_char,
                       c.text, c.metadata, c.acl, c.section_path,
                       (1.0 - (c.embedding <=> %s::halfvec({dim})))::float8 AS sim
                FROM child_chunks c
                JOIN documents d
                  ON d.tenant_id = c.tenant_id
                 AND d.doc_id = c.doc_id
                 AND d.active_version = c.version_id
                 AND d.is_deleted = FALSE
                WHERE c.tenant_id = %s
                  AND c.is_deleted = FALSE
                  AND c.index_status = 'ready'
                  AND c.acl && %s::text[]
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::halfvec({dim})
                LIMIT %s
                """,
                (vec_lit, tid, roles, vec_lit, top_k),
            )
            rows = cur.fetchall()
            return [(self._row_to_chunk(r[:10]), round(float(r[10]), 6)) for r in rows]

    def search_bm25(
        self,
        query_text: str,
        top_k: int = 10,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[tuple[TextChunk, float]]:
        """Lexical search combining PostgreSQL `tsvector` with bilingual token fallback."""
        tokens = tokenize_bm25(query_text)
        if not tokens:
            return []
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)

        with self._connect() as conn, conn.cursor() as cur:
            # 1. PostgreSQL native tsvector full-text ranking (fast GIN index)
            cur.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.source, c.chunk_index, c.start_char, c.end_char,
                       c.text, c.metadata, c.acl, c.section_path,
                       ts_rank_cd(c.tsv, plainto_tsquery('simple', %s))::float8 AS rank_score
                FROM child_chunks c
                JOIN documents d
                  ON d.tenant_id = c.tenant_id
                 AND d.doc_id = c.doc_id
                 AND d.active_version = c.version_id
                 AND d.is_deleted = FALSE
                WHERE c.tenant_id = %s
                  AND c.is_deleted = FALSE
                  AND c.index_status = 'ready'
                  AND c.acl && %s::text[]
                  AND c.tsv @@ plainto_tsquery('simple', %s)
                ORDER BY rank_score DESC
                LIMIT %s
                """,
                (query_text, tid, roles, query_text, top_k),
            )
            rows = cur.fetchall()
            if rows:
                return [(self._row_to_chunk(r[:10]), round(float(r[10]), 6)) for r in rows]

            # 2. Bilingual CJK/phrase fallback via top informative tokens + ILIKE overlap
            top_toks = sorted(set(tokens), key=lambda t: -len(t))[:6]
            patterns = [f"%{t}%" for t in top_toks]
            cur.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.source, c.chunk_index, c.start_char, c.end_char,
                       c.text, c.metadata, c.acl, c.section_path
                FROM child_chunks c
                JOIN documents d
                  ON d.tenant_id = c.tenant_id
                 AND d.doc_id = c.doc_id
                 AND d.active_version = c.version_id
                 AND d.is_deleted = FALSE
                WHERE c.tenant_id = %s
                  AND c.is_deleted = FALSE
                  AND c.index_status = 'ready'
                  AND c.acl && %s::text[]
                  AND c.text ILIKE ANY(%s::text[])
                LIMIT %s
                """,
                (tid, roles, patterns, max(top_k * 3, 20)),
            )
            cand_rows = cur.fetchall()
            scored: list[tuple[TextChunk, float]] = []
            for r in cand_rows:
                ch = self._row_to_chunk(r)
                low = ch.text.lower()
                hits = sum(1 for t in top_toks if t in low)
                if hits > 0:
                    scored.append((ch, round(hits / len(top_toks), 6)))
            scored.sort(key=lambda item: -item[1])
            return scored[:top_k]

    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 4,
        *,
        bm25_weight: float = 0.35,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[tuple[TextChunk, float]]:
        """Weighted Reciprocal Rank Fusion (RRF) over PG HNSW Dense + PG Lexical hits."""
        pool_k = max(top_k * candidate_multiplier, top_k + 12)
        dense_hits = self.search(
            query_vector, top_k=pool_k, tenant_id=tenant_id, acl_roles=acl_roles
        )
        bm25_hits = self.search_bm25(
            query_text, top_k=pool_k, tenant_id=tenant_id, acl_roles=acl_roles
        )
        if not bm25_hits:
            return dense_hits[:top_k]

        w_lex = max(0.0, min(1.0, bm25_weight))
        w_dense = 1.0 - w_lex
        fused: dict[str, float] = {}
        chunk_map: dict[str, TextChunk] = {}

        for rank, (ch, _) in enumerate(dense_hits, start=1):
            chunk_map[ch.chunk_id] = ch
            fused[ch.chunk_id] = fused.get(ch.chunk_id, 0.0) + w_dense / (rrf_k + rank)
        for rank, (ch, _) in enumerate(bm25_hits, start=1):
            chunk_map[ch.chunk_id] = ch
            fused[ch.chunk_id] = fused.get(ch.chunk_id, 0.0) + w_lex / (rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [(chunk_map[cid], round(score, 6)) for cid, score in ranked]

    def explain_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
        force_index_scan: bool = False,
    ) -> str:
        """Run EXPLAIN ANALYZE on the vector HNSW query to verify real index usage."""
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        vec_lit = _vec_to_pg_literal(query_vector)
        dim = self.embedding_dim
        with self._connect() as conn, conn.cursor() as cur:
            self._apply_hnsw_session_params(cur)
            if force_index_scan:
                cur.execute("SET LOCAL enable_seqscan = off")
                cur.execute("SET LOCAL enable_sort = off")
            cur.execute(
                f"""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT chunk_id, (1.0 - (embedding <=> %s::halfvec({dim})))::float8 AS sim
                FROM child_chunks
                WHERE tenant_id = %s
                  AND is_deleted = FALSE
                  AND index_status = 'ready'
                  AND acl && %s::text[]
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::halfvec({dim})
                LIMIT %s
                """,
                (vec_lit, tid, roles, vec_lit, top_k),
            )
            return "\n".join(str(r[0]) for r in cur.fetchall())

    def soft_delete_document(
        self,
        doc_id: str,
        *,
        tenant_id: str | None = None,
        operator: str = "system",
    ) -> int:
        """Atomically soft-delete a document and its parent/child chunks with audit logging."""
        tid = tenant_id or self.default_tenant_id
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET is_deleted = TRUE, status = 'deleted', updated_at = NOW() "
                "WHERE tenant_id = %s AND doc_id = %s",
                (tid, doc_id),
            )
            cur.execute(
                "UPDATE parent_chunks SET is_deleted = TRUE WHERE tenant_id = %s AND doc_id = %s",
                (tid, doc_id),
            )
            cur.execute(
                "UPDATE child_chunks SET is_deleted = TRUE, updated_at = NOW() "
                "WHERE tenant_id = %s AND doc_id = %s",
                (tid, doc_id),
            )
            affected = int(cur.rowcount)
            cur.execute(
                """
                INSERT INTO audit_logs (tenant_id, doc_id, change_type, operator, result, payload)
                VALUES (%s, %s, 'soft_delete', %s, 'success', %s::jsonb)
                """,
                (tid, doc_id, operator, json.dumps({"affected_chunks": affected})),
            )
            conn.commit()
            return affected

    def clear_tenant(self, tenant_id: str | None = None) -> None:
        tid = tenant_id or self.default_tenant_id
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM child_chunks WHERE tenant_id = %s", (tid,))
            cur.execute("DELETE FROM parent_chunks WHERE tenant_id = %s", (tid,))
            cur.execute("DELETE FROM documents WHERE tenant_id = %s", (tid,))
            conn.commit()

    def get_chunk(
        self,
        chunk_id: str,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> TextChunk | None:
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.source, c.chunk_index, c.start_char, c.end_char,
                       c.text, c.metadata, c.acl, c.section_path
                FROM child_chunks c
                JOIN documents d
                  ON d.tenant_id = c.tenant_id
                 AND d.doc_id = c.doc_id
                 AND d.active_version = c.version_id
                 AND d.is_deleted = FALSE
                WHERE c.chunk_id = %s
                  AND c.tenant_id = %s
                  AND c.is_deleted = FALSE
                  AND c.index_status = 'ready'
                  AND c.acl && %s::text[]
                """,
                (chunk_id, tid, roles),
            )
            row = cur.fetchone()
            return self._row_to_chunk(row) if row else None

    def get_all_chunks(
        self,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[TextChunk]:
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.source, c.chunk_index, c.start_char, c.end_char,
                       c.text, c.metadata, c.acl, c.section_path
                FROM child_chunks c
                JOIN documents d
                  ON d.tenant_id = c.tenant_id
                 AND d.doc_id = c.doc_id
                 AND d.active_version = c.version_id
                 AND d.is_deleted = FALSE
                WHERE c.tenant_id = %s
                  AND c.is_deleted = FALSE
                  AND c.index_status = 'ready'
                  AND c.acl && %s::text[]
                ORDER BY c.doc_id, c.chunk_index
                """,
                (tid, roles),
            )
            return [self._row_to_chunk(r) for r in cur.fetchall()]

    def get_parent_chunks(
        self,
        parent_ids: list[str],
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fetch Parent reading-grain windows for retrieved child chunks (Parent-Child expansion)."""
        if not parent_ids:
            return {}
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.parent_id, p.doc_id, p.version_id, p.section_path, p.page_no,
                       p.element_type, p.content, p.token_count, p.metadata
                FROM parent_chunks p
                JOIN documents d
                  ON d.tenant_id = p.tenant_id
                 AND d.doc_id = p.doc_id
                 AND d.active_version = p.version_id
                 AND d.is_deleted = FALSE
                WHERE p.parent_id = ANY(%s::text[])
                  AND p.tenant_id = %s
                  AND p.is_deleted = FALSE
                  AND p.acl && %s::text[]
                """,
                (list(dict.fromkeys(parent_ids)), tid, roles),
            )
            out: dict[str, dict[str, Any]] = {}
            for r in cur.fetchall():
                out[str(r[0])] = {
                    "parent_id": str(r[0]),
                    "doc_id": str(r[1]),
                    "version_id": int(r[2]),
                    "section_path": str(r[3]),
                    "page_no": r[4],
                    "element_type": str(r[5]),
                    "content": str(r[6]),
                    "token_count": int(r[7]),
                    "metadata": dict(r[8]) if isinstance(r[8], dict) else {},
                }
            return out

    def rollback_document_version(
        self,
        doc_id: str,
        target_version: int,
        *,
        tenant_id: str | None = None,
        operator: str = "system",
    ) -> bool:
        """Atomically roll back `documents.active_version` to an earlier ready version."""
        tid = tenant_id or self.default_tenant_id
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM child_chunks
                WHERE tenant_id = %s AND doc_id = %s AND version_id = %s
                  AND index_status = 'ready' AND is_deleted = FALSE
                """,
                (tid, doc_id, target_version),
            )
            ready_cnt = int((cur.fetchone() or (0,))[0])
            if ready_cnt == 0:
                return False
            cur.execute(
                """
                UPDATE documents
                SET active_version = %s, status = 'ready', is_deleted = FALSE, updated_at = NOW()
                WHERE tenant_id = %s AND doc_id = %s
                """,
                (target_version, tid, doc_id),
            )
            cur.execute(
                """
                INSERT INTO audit_logs (tenant_id, doc_id, change_type, operator, result, payload)
                VALUES (%s, %s, 'rollback_version', %s, 'success', %s::jsonb)
                """,
                (tid, doc_id, operator, json.dumps({"target_version": target_version, "ready_chunks": ready_cnt})),
            )
            conn.commit()
            return True

    def save(self, directory: Path) -> None:
        """No-op: PostgreSQL persists transactions directly."""
        _ = directory

    def load(self, directory: Path) -> None:
        """No-op: PostgreSQL reads live state directly."""
        _ = directory


class PgVerifiedMemoryStore:
    """PostgreSQL-backed Stage 3.4/4.1 A-MEM verified episodic memory ledger."""

    def __init__(
        self,
        dsn: str,
        *,
        default_tenant_id: str = "default",
        default_acl_roles: list[str] | str = "public,internal",
    ) -> None:
        self.dsn = dsn
        self.default_tenant_id = default_tenant_id
        self.default_acl_roles = _parse_roles(default_acl_roles, ["public", "internal"])
        self._psycopg = _import_psycopg()
        self.storage_dir: Path | None = None

    def _connect(self) -> Any:
        return self._psycopg.connect(self.dsn)

    def load(self, storage_dir: Path | None = None) -> list[VerifiedMemoryRecord]:
        self.storage_dir = storage_dir
        return self.get_all()

    def get_all(
        self,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[VerifiedMemoryRecord]:
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT memory_id, question, conclusion, condition_entity_ids, target_entity_ids,
                       source_chunk_ids, grounding_score, confidence, reasoning_paradigm,
                       valid_from, provenance_run_id, conflict_checked, status, metadata
                FROM verified_memories
                WHERE tenant_id = %s AND acl && %s::text[]
                ORDER BY valid_from ASC
                """,
                (tid, roles),
            )
            out: list[VerifiedMemoryRecord] = []
            for r in cur.fetchall():
                vf = r[9].isoformat() if hasattr(r[9], "isoformat") else str(r[9])
                out.append(
                    VerifiedMemoryRecord(
                        memory_id=str(r[0]),
                        question=str(r[1]),
                        conclusion=str(r[2]),
                        condition_entity_ids=list(r[3] or []),
                        target_entity_ids=list(r[4] or []),
                        source_chunk_ids=list(r[5] or []),
                        grounding_score=float(r[6]),
                        confidence=float(r[7]),
                        reasoning_paradigm=str(r[8]),
                        valid_from=vf,
                        provenance_run_id=str(r[10]),
                        conflict_checked=bool(r[11]),
                        status=str(r[12]),
                        metadata=dict(r[13]) if isinstance(r[13], dict) else {},
                    )
                )
            return out

    def evaluate_and_write_back(
        self,
        *,
        question: str,
        answer: str,
        grounding_score: float | None,
        ungrounded_citations: list[str] | None = None,
        confidence: float = 0.0,
        condition_entity_ids: list[str] | None = None,
        target_entity_ids: list[str] | None = None,
        source_chunk_ids: list[str] | None = None,
        reasoning_paradigm: str = "auto",
        min_grounding: float = 1.0,
        min_confidence: float = 0.75,
        enable_conflict_check: bool = True,
        promote_to_hyperedge: bool = True,
        graph_store: BaseGraphStore | None = None,
        provenance_run_id: str | None = None,
        tenant_id: str | None = None,
        acl: list[str] | None = None,
    ) -> WriteBackGateDecision:
        """Enforce the exact 4-stage anti-poisoning gate and persist into PG `verified_memories`."""
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl, self.default_acl_roles)
        ungrounded = list(ungrounded_citations or [])
        cids = list(dict.fromkeys(source_chunk_ids or []))
        cond_ids = list(dict.fromkeys(condition_entity_ids or []))
        tgt_ids = list(dict.fromkeys(target_entity_ids or []))

        if grounding_score is None or grounding_score < min_grounding or ungrounded or not cids:
            return WriteBackGateDecision(
                accepted=False,
                gate_failed="GATE_1_GROUNDING",
                reason="Failed GroundingAudit or missing verifiable source_chunk_ids.",
            )
        clean_ans = answer.strip()
        if len(clean_ans) < 20 or detect_refusal(clean_ans):
            return WriteBackGateDecision(
                accepted=False,
                gate_failed="GATE_2_SUBSTANCE",
                reason="Refusal or too-short answer rejected.",
            )
        if confidence < min_confidence:
            return WriteBackGateDecision(
                accepted=False,
                gate_failed="GATE_2_CONFIDENCE",
                reason=f"Confidence ({confidence:.2f}) < threshold ({min_confidence:.2f}).",
            )

        existing = self.get_all(tenant_id=tid, acl_roles=roles)
        if enable_conflict_check:
            has_conflict, conflict_reason = check_memory_conflict(
                conclusion=clean_ans,
                condition_entity_ids=cond_ids,
                target_entity_ids=tgt_ids,
                existing_records=existing,
            )
            if has_conflict:
                return WriteBackGateDecision(
                    accepted=False,
                    gate_failed="GATE_3_CONFLICT_NLI",
                    reason=conflict_reason,
                )

        sig_hash = hashlib.sha1(
            f"{tid}|{question.strip().lower()}|{clean_ans[:240].lower()}".encode()
        ).hexdigest()[:12]
        mem_id = f"mem_{sig_hash}"
        now_iso = datetime.now(UTC).replace(microsecond=0).isoformat()
        run_id = provenance_run_id or f"run_{uuid.uuid4().hex[:10]}"

        record = VerifiedMemoryRecord(
            memory_id=mem_id,
            question=question.strip()[:400],
            conclusion=clean_ans[:800],
            condition_entity_ids=cond_ids,
            target_entity_ids=tgt_ids,
            source_chunk_ids=cids[:8],
            grounding_score=round(float(grounding_score), 4),
            confidence=round(float(confidence), 4),
            reasoning_paradigm=reasoning_paradigm,
            valid_from=now_iso,
            provenance_run_id=run_id,
            conflict_checked=enable_conflict_check,
            status="verified",
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO verified_memories (
                    memory_id, tenant_id, acl, question, conclusion,
                    condition_entity_ids, target_entity_ids, source_chunk_ids,
                    grounding_score, confidence, reasoning_paradigm,
                    valid_from, provenance_run_id, conflict_checked, status, metadata
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s::timestamptz, %s, %s, %s, %s::jsonb
                )
                ON CONFLICT (memory_id) DO NOTHING
                """,
                (
                    record.memory_id,
                    tid,
                    roles,
                    record.question,
                    record.conclusion,
                    record.condition_entity_ids,
                    record.target_entity_ids,
                    record.source_chunk_ids,
                    record.grounding_score,
                    record.confidence,
                    record.reasoning_paradigm,
                    record.valid_from,
                    record.provenance_run_id,
                    record.conflict_checked,
                    record.status,
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()

        promoted_hid: str | None = None
        if (
            promote_to_hyperedge
            and graph_store is not None
            and len(cond_ids) >= 2
            and tgt_ids
            and hasattr(graph_store, "upsert_hyperedge")
        ):
            he = create_hyperedge(
                cond_ids[:5],
                tgt_ids[0],
                relation_type="VERIFIED_JOINT_DIAGNOSIS",
                description=f"[Verified Memory {mem_id}] {clean_ans[:200]}",
                confidence=record.confidence,
                source_chunk_ids=cids[:6],
                metadata={"from_verified_memory": mem_id, "valid_from": now_iso},
            )
            graph_store.upsert_hyperedge(he)
            promoted_hid = he.hyperedge_id

        return WriteBackGateDecision(
            accepted=True,
            reason=f"Verified memory `{mem_id}` persisted to PostgreSQL (valid_from={now_iso}).",
            record=record,
            promoted_hyperedge_id=promoted_hid,
        )

    def retrieve_relevant(
        self,
        question: str,
        seed_entity_ids: list[str],
        *,
        top_k: int = 3,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[VerifiedMemoryRecord]:
        records = self.get_all(tenant_id=tenant_id, acl_roles=acl_roles)
        if not records:
            return []
        q_toks = set(tokenize_bm25(question))
        seed_set = set(seed_entity_ids)
        scored: list[tuple[float, VerifiedMemoryRecord]] = []
        for rec in records:
            if rec.status != "verified":
                continue
            ent_pool = set(rec.condition_entity_ids) | set(rec.target_entity_ids)
            ent_hits = len(seed_set & ent_pool)
            rec_toks = set(tokenize_bm25(f"{rec.question} {rec.conclusion}"))
            tok_hits = len(q_toks & rec_toks) / max(1, len(q_toks)) if q_toks else 0.0
            score = ent_hits * 0.65 + tok_hits
            if score >= 0.35:
                scored.append((score, rec))
        scored.sort(key=lambda item: (-item[0], item[1].valid_from))
        return [r for _, r in scored[:top_k]]

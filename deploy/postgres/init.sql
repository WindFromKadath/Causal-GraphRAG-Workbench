-- LLM-GraphRAG-KG Enterprise PostgreSQL 16 + pgvector 0.8 Schema (Stage 4.1)
-- Single ACID Source of Truth for:
--   1. Embedding index registry & blue-green alias tracking (`index_registry`)
--   2. Multi-tenant documents, active version switching & soft-delete (`documents`)
--   3. Parent-Child chunk storage (`parent_chunks`, `child_chunks` with halfvec(1024) HNSW + tsvector)
--   4. Verified episodic memory audit ledger (`verified_memories`)
--   5. Change audit logs & request observability traces (`audit_logs`, `request_traces`)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Embedding Model Fingerprint & Blue-Green Index Registry
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

-- 2. Documents Master Table (ACID Source of Truth)
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

CREATE INDEX IF NOT EXISTS idx_documents_acl_gin ON documents USING GIN (acl);
CREATE INDEX IF NOT EXISTS idx_documents_active ON documents (tenant_id, is_deleted, status);

-- 3. Parent Chunks (Reading Granularity DocStore: 1000-1500 tokens / complete tables)
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

CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc ON parent_chunks (tenant_id, doc_id, version_id, is_deleted);

-- 4. Child Chunks (Retrieval Granularity: 256-384 tokens + halfvec(1024) HNSW + tsvector)
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
    embedding halfvec(1024),
    embedding_model TEXT NOT NULL DEFAULT 'qwen3-embedding:0.6b',
    embedding_dim INTEGER NOT NULL DEFAULT 1024,
    tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(section_path, '') || ' ' || coalesce(text, ''))
    ) STORED,
    index_status TEXT NOT NULL DEFAULT 'ready',
    claim_token TEXT,
    claimed_at TIMESTAMPTZ,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_child_chunks_hnsw_cosine
    ON child_chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_child_chunks_tsv_gin ON child_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_child_chunks_acl_gin ON child_chunks USING GIN (acl);
CREATE INDEX IF NOT EXISTS idx_child_chunks_filter
    ON child_chunks (tenant_id, is_deleted, index_status, doc_id, version_id);
CREATE INDEX IF NOT EXISTS idx_child_chunks_hash
    ON child_chunks (tenant_id, doc_id, content_hash);

-- 5. Verified Episodic Memories (Stage 3.4 A-MEM Audit Ledger in PostgreSQL)
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

CREATE INDEX IF NOT EXISTS idx_verified_memories_tenant
    ON verified_memories (tenant_id, status, valid_from DESC);
CREATE INDEX IF NOT EXISTS idx_verified_memories_acl_gin
    ON verified_memories USING GIN (acl);

-- 6. Audit Logs & Request Observability Traces (Ready for Stage 4.2 & 4.3)
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

CREATE TABLE IF NOT EXISTS request_traces (
    request_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    rewritten_queries TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    retrieval_mode TEXT NOT NULL DEFAULT 'ppr',
    reasoning_paradigm TEXT NOT NULL DEFAULT 'auto',
    final_status TEXT NOT NULL DEFAULT 'answered',
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    stage_timings JSONB NOT NULL DEFAULT '{}'::JSONB,
    version_snapshot JSONB NOT NULL DEFAULT '{}'::JSONB,
    trace_payload JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

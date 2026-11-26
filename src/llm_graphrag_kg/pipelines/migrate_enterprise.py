"""Stage 4.1 enterprise infrastructure diagnostics and local-to-PG/Neo4j migration helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.pipelines.memory_store import VerifiedMemoryStore
from llm_graphrag_kg.providers.factory import create_embedding_client
from llm_graphrag_kg.providers.storage.local_vector import LocalVectorStore
from llm_graphrag_kg.providers.storage.neo4j_store import Neo4jGraphStore
from llm_graphrag_kg.providers.storage.networkx_store import NetworkXGraphStore
from llm_graphrag_kg.providers.storage.pgvector_store import PgVectorStore, PgVerifiedMemoryStore


@dataclass
class InfraCheckItem:
    service: str
    ok: bool
    endpoint: str
    detail: str


@dataclass
class MigrationReport:
    source_storage: str
    tenant_id: str
    acl_roles: list[str]
    embedding_model: str
    embedding_dim: int
    chunks_migrated: int
    entities_migrated: int
    relations_migrated: int
    hyperedges_migrated: int
    communities_migrated: int
    memories_migrated: int
    elapsed_seconds: float


def _parse_acl_list(acl_roles: list[str] | str | None, fallback: str) -> list[str]:
    raw = fallback if acl_roles is None else acl_roles
    if isinstance(raw, str):
        parsed = [r.strip() for r in raw.split(",") if r.strip()]
        return parsed or ["public"]
    return [r.strip() for r in raw if r.strip()] or ["public"]


def check_redis_health(redis_url: str) -> tuple[bool, str]:
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        return False, "Redis Python client not installed (`uv sync --extra enterprise`)."

    try:
        client = redis.from_url(redis_url, socket_connect_timeout=3, socket_timeout=3)
        pong = bool(client.ping())
        info = client.info("server")
        ver = str(info.get("redis_version", "unknown"))
        client.close()
        return pong, f"Redis ready (v{ver} @ {redis_url})"
    except Exception as exc:  # noqa: BLE001
        return False, f"Redis health check failed ({redis_url}): {exc}"


def check_enterprise_infra(config: AppConfig) -> list[InfraCheckItem]:
    """Verify PostgreSQL 16 + pgvector, Neo4j 5.26, and Redis 7 reachability."""
    results: list[InfraCheckItem] = []

    # 1. PostgreSQL + pgvector
    pg_endpoint = config.env.postgres_dsn.split("@")[-1]
    try:
        pg = PgVectorStore(
            dsn=config.env.postgres_dsn,
            embedding_dim=int(config.env.embedding_dimensions or config.env.pgvector_dim),
            embedding_model=config.env.embedding_model,
            ef_search=config.env.pgvector_ef_search,
            iterative_scan=config.env.pgvector_iterative_scan,
            default_tenant_id=config.env.default_tenant_id,
            default_acl_roles=config.env.default_acl_roles,
            auto_init_schema=False,
        )
        ok, detail = pg.health_check()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"PostgreSQL init failed: {exc}"
    results.append(InfraCheckItem("PostgreSQL + pgvector", ok, pg_endpoint, detail))

    # 2. Neo4j
    try:
        neo = Neo4jGraphStore(
            uri=config.env.neo4j_uri,
            user=config.env.neo4j_user,
            password=config.env.neo4j_password,
            database=config.env.neo4j_database,
            default_tenant_id=config.env.default_tenant_id,
            default_acl_roles=config.env.default_acl_roles,
            auto_init_constraints=False,
        )
        ok, detail = neo.health_check()
        neo.close()
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"Neo4j init failed: {exc}"
    results.append(InfraCheckItem("Neo4j", ok, config.env.neo4j_uri, detail))

    # 3. Redis
    ok, detail = check_redis_health(config.env.redis_url)
    results.append(InfraCheckItem("Redis", ok, config.env.redis_url, detail))
    return results


def migrate_local_storage_to_enterprise(
    storage_dir: Path,
    config: AppConfig,
    *,
    tenant_id: str | None = None,
    acl_roles: list[str] | str | None = None,
    clear_target: bool = False,
) -> MigrationReport:
    """Migrate a local NetworkX + LocalVectorStore + A-MEM directory into PG16 + Neo4j."""
    t0 = time.perf_counter()
    tid = (tenant_id or config.env.default_tenant_id).strip() or "default"
    roles = _parse_acl_list(acl_roles, config.env.default_acl_roles)
    eff_dim = int(config.env.embedding_dimensions or config.env.pgvector_dim)

    # 1. Load local stores
    local_graph = NetworkXGraphStore()
    local_vec = LocalVectorStore()
    local_mem = VerifiedMemoryStore()
    local_graph.load(storage_dir)
    local_vec.load(storage_dir)
    local_mem.load(storage_dir)

    chunks = local_vec.get_all_chunks()
    entities = local_graph.get_all_entities()
    relations = local_graph.get_all_relations()
    hyperedges = local_graph.get_all_hyperedges()
    communities = local_graph.get_all_communities()
    memories = local_mem.get_all()

    # 2. Initialize enterprise stores
    pg_store = PgVectorStore(
        dsn=config.env.postgres_dsn,
        embedding_dim=eff_dim,
        embedding_model=config.env.embedding_model,
        ef_search=config.env.pgvector_ef_search,
        iterative_scan=config.env.pgvector_iterative_scan,
        default_tenant_id=tid,
        default_acl_roles=roles,
        auto_init_schema=True,
    )
    neo_store = Neo4jGraphStore(
        uri=config.env.neo4j_uri,
        user=config.env.neo4j_user,
        password=config.env.neo4j_password,
        database=config.env.neo4j_database,
        default_tenant_id=tid,
        default_acl_roles=roles,
        auto_init_constraints=True,
    )
    pg_mem = PgVerifiedMemoryStore(
        dsn=config.env.postgres_dsn,
        default_tenant_id=tid,
        default_acl_roles=roles,
    )

    try:
        if clear_target:
            pg_store.clear_tenant(tid)
            neo_store.clear_tenant(tid)

        # 3. Re-embed chunks with configured enterprise embedder (e.g. qwen3-embedding:0.6b @ 1024d)
        embedder = create_embedding_client(config)
        vectors: list[list[float]] = []
        if chunks:
            vectors = embedder.embed([c.text for c in chunks])
            if not vectors or len(vectors[0]) != eff_dim:
                got_dim = len(vectors[0]) if vectors else 0
                raise ValueError(
                    f"Embedder `{config.env.embedding_model}` produced dim={got_dim}, "
                    f"expected pgvector_dim={eff_dim}. Pass `--embedder ollama_native` or align dimensions."
                )
            pg_store.add_chunks(
                chunks,
                vectors,
                tenant_id=tid,
                acl=roles,
                embedding_model=config.env.embedding_model,
            )

        # 4. Migrate Entities, Relations, HyperEdges, ACL-partitioned Communities to Neo4j
        for ent in entities:
            neo_store.upsert_entity(ent, tenant_id=tid, acl=roles)
        for rel in relations:
            neo_store.upsert_relation(rel, tenant_id=tid, acl=roles)
        for he in hyperedges:
            neo_store.upsert_hyperedge(he, tenant_id=tid, acl=roles)
        if communities:
            neo_store.set_communities(communities, tenant_id=tid, acl_partition=roles[0])

        # 5. Migrate existing VerifiedMemoryRecords (if any) into PG `verified_memories`
        mem_migrated = 0
        if memories:
            with pg_mem._connect() as conn, conn.cursor() as cur:
                for rec in memories:
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
                            rec.memory_id,
                            tid,
                            roles,
                            rec.question,
                            rec.conclusion,
                            rec.condition_entity_ids,
                            rec.target_entity_ids,
                            rec.source_chunk_ids,
                            rec.grounding_score,
                            rec.confidence,
                            rec.reasoning_paradigm,
                            rec.valid_from,
                            rec.provenance_run_id,
                            rec.conflict_checked,
                            rec.status,
                            json.dumps(rec.metadata, ensure_ascii=False),
                        ),
                    )
                    mem_migrated += 1
                conn.commit()

        return MigrationReport(
            source_storage=str(storage_dir),
            tenant_id=tid,
            acl_roles=roles,
            embedding_model=config.env.embedding_model,
            embedding_dim=eff_dim,
            chunks_migrated=len(chunks),
            entities_migrated=len(entities),
            relations_migrated=len(relations),
            hyperedges_migrated=len(hyperedges),
            communities_migrated=len(communities),
            memories_migrated=mem_migrated,
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        )
    finally:
        neo_store.close()

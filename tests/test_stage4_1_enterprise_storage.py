"""Stage 4.1 unit tests for PostgreSQL+pgvector, Neo4j 5.26, ACL pre-filtering, and factory routing.

All unit tests run completely offline without requiring Docker containers.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.schema import Community, Entity, HyperEdge, Relation, TextChunk
from llm_graphrag_kg.pipelines.memory_store import VerifiedMemoryStore
from llm_graphrag_kg.pipelines.migrate_enterprise import _parse_acl_list
from llm_graphrag_kg.providers.factory import (
    create_graph_store,
    create_memory_store,
    create_vector_store,
)
from llm_graphrag_kg.providers.storage.local_vector import LocalVectorStore
from llm_graphrag_kg.providers.storage.neo4j_store import Neo4jGraphStore
from llm_graphrag_kg.providers.storage.networkx_store import NetworkXGraphStore
from llm_graphrag_kg.providers.storage.pgvector_store import (
    PgVectorStore,
    PgVerifiedMemoryStore,
    _vec_to_pg_literal,
)


class _FakePgCursor:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._responses = list(responses or [])
        self.rowcount = 1

    def __enter__(self) -> _FakePgCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> Any:
        if self._responses:
            return self._responses.pop(0)
        return None

    def fetchall(self) -> list[Any]:
        if self._responses:
            nxt = self._responses.pop(0)
            return list(nxt) if isinstance(nxt, list) else [nxt]
        return []


class _FakePgConn:
    def __init__(self, cursor: _FakePgCursor) -> None:
        self.cursor_obj = cursor
        self.committed = False

    def __enter__(self) -> _FakePgConn:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def cursor(self) -> _FakePgCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True


class _FakePsycopg:
    def __init__(self, cursor: _FakePgCursor) -> None:
        self.conn = _FakePgConn(cursor)
        self.dsn_seen: str | None = None

    def connect(self, dsn: str) -> _FakePgConn:
        self.dsn_seen = dsn
        return self.conn


class _FakeNeoResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def single(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def __iter__(self) -> Any:
        return iter(self._rows)


class _FakeNeoSession:
    def __init__(self, responses: list[list[dict[str, Any]]] | None = None) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self._responses = list(responses or [])

    def __enter__(self) -> _FakeNeoSession:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def run(self, cypher: str, **params: Any) -> _FakeNeoResult:
        self.queries.append((cypher, params))
        rows = self._responses.pop(0) if self._responses else []
        return _FakeNeoResult(rows)


class _FakeNeoDriver:
    def __init__(self, session: _FakeNeoSession) -> None:
        self.session_obj = session
        self.closed = False

    def session(self, database: str = "neo4j") -> _FakeNeoSession:
        _ = database
        return self.session_obj

    def close(self) -> None:
        self.closed = True


def test_pgvector_literal_and_acl_parser() -> None:
    lit = _vec_to_pg_literal([0.1, -0.25, 1.0])
    assert lit == "[0.1,-0.25,1]"
    assert _parse_acl_list("public, engineer", "public") == ["public", "engineer"]
    assert _parse_acl_list(None, "public,internal") == ["public", "internal"]


def test_pgvector_index_registry_blocks_model_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Existing registry row has nomic-embed-text (768d), runtime attempts qwen3-embedding:0.6b (1024d)
    cur = _FakePgCursor(responses=[("nomic-embed-text", 768, "child_chunks", "active")])
    fake_pg = _FakePsycopg(cur)
    monkeypatch.setattr(
        "llm_graphrag_kg.providers.storage.pgvector_store._import_psycopg",
        lambda: fake_pg,
    )
    store = PgVectorStore(
        "postgresql://user:pass@127.0.0.1:25432/db",
        embedding_dim=1024,
        embedding_model="qwen3-embedding:0.6b",
        auto_init_schema=False,
    )
    with pytest.raises(ValueError, match="Embedding model mismatch"):
        store.register_or_verify_index()


def test_pgvector_dense_search_sets_iterative_scan_and_acl_prefilter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_row = (
        "c1",
        "d1",
        "doc.md",
        0,
        0,
        20,
        "bearing temperature",
        {"acl": ["public"]},
        ["public"],
        "1.1",
        0.92,
    )
    cur = _FakePgCursor(responses=[[chunk_row]])
    fake_pg = _FakePsycopg(cur)
    monkeypatch.setattr(
        "llm_graphrag_kg.providers.storage.pgvector_store._import_psycopg",
        lambda: fake_pg,
    )
    store = PgVectorStore(
        "postgresql://user:pass@127.0.0.1:25432/db",
        embedding_dim=4,
        default_tenant_id="tenant_a",
        default_acl_roles="public",
        iterative_scan="strict_order",
        auto_init_schema=False,
    )
    hits = store.search([0.5, 0.5, 0.5, 0.5], top_k=3)
    assert len(hits) == 1
    assert hits[0][0].chunk_id == "c1"
    assert pytest.approx(hits[0][1], rel=1e-3) == 0.92
    # Check SET LOCAL iterative_scan + ef_search + Pre-Filter WHERE clause
    sql_joined = "\n".join(sql for sql, _ in cur.executed)
    assert "hnsw.iterative_scan" in sql_joined
    assert "hnsw.ef_search" in sql_joined
    assert "tenant_id = %s" in sql_joined
    assert "is_deleted = FALSE" in sql_joined
    assert "acl && %s::text[]" in sql_joined


def test_pgvector_add_chunks_rejects_dim_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cur = _FakePgCursor()
    monkeypatch.setattr(
        "llm_graphrag_kg.providers.storage.pgvector_store._import_psycopg",
        lambda: _FakePsycopg(cur),
    )
    store = PgVectorStore(
        "postgresql://user:pass@127.0.0.1:25432/db",
        embedding_dim=1024,
        auto_init_schema=False,
    )
    chunk = TextChunk(
        chunk_id="c1",
        doc_id="d1",
        source="a.md",
        text="hello",
        index=0,
        start_char=0,
        end_char=5,
    )
    with pytest.raises(ValueError, match="Vector dim"):
        store.add_chunks([chunk], [[0.1, 0.2, 0.3]])


def test_pg_verified_memory_store_enforces_4_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    # First call: fail Gate 1 (ungrounded); Second call: pass all 4 gates and insert into PG
    cur = _FakePgCursor(responses=[[]])  # get_all() returns empty existing memories
    fake_pg = _FakePsycopg(cur)
    monkeypatch.setattr(
        "llm_graphrag_kg.providers.storage.pgvector_store._import_psycopg",
        lambda: fake_pg,
    )
    mem_store = PgVerifiedMemoryStore(
        "postgresql://user:pass@127.0.0.1:25432/db",
        default_tenant_id="tenant_a",
        default_acl_roles="public",
    )
    bad = mem_store.evaluate_and_write_back(
        question="test?",
        answer="A sufficiently long verified diagnostic conclusion for testing.",
        grounding_score=0.5,
        confidence=0.9,
        source_chunk_ids=["c1"],
    )
    assert not bad.accepted
    assert bad.gate_failed == "GATE_1_GROUNDING"

    good = mem_store.evaluate_and_write_back(
        question="What causes 2x vibration and high bearing temperature?",
        answer="Shaft misalignment causes 2x radial vibration and elevated bearing temperature.",
        grounding_score=1.0,
        confidence=0.92,
        condition_entity_ids=["e_vib", "e_temp"],
        target_entity_ids=["e_misalign"],
        source_chunk_ids=["c1"],
    )
    assert good.accepted
    assert good.record is not None
    assert any("INSERT INTO verified_memories" in sql for sql, _ in cur.executed)


def test_neo4j_store_enforces_acl_in_cypher_and_hyperedges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ent_props = {
        "entity_id": "e1",
        "name": "Centrifugal Pump",
        "entity_type": "Equipment",
        "description": "Pump",
        "aliases": [],
        "source_chunk_ids": ["c1"],
        "community_id": 0,
        "acl": ["public"],
        "metadata_json": "{}",
    }
    he_props = {
        "hyperedge_id": "he_1",
        "condition_entity_ids": ["e1", "e2"],
        "target_entity_id": "e3",
        "relation_type": "JOINT_INDICATES",
        "description": "joint rule",
        "confidence": 0.9,
        "weight": 1.0,
        "source_chunk_ids": ["c1"],
        "metadata_json": "{}",
    }
    sess = _FakeNeoSession(
        responses=[
            [{"props": ent_props}],  # upsert_entity
            [],  # upsert_hyperedge
            [{"props": he_props}],  # get_all_hyperedges
            [{"props": ent_props}],  # get_k_hop_subgraph nodes
            [],  # get_k_hop_subgraph rels
            [],  # set_communities delete
            [],  # set_communities create
            [],  # set_communities set entity community_id
        ]
    )
    fake_driver = _FakeNeoDriver(sess)

    class _FakeGraphDB:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> _FakeNeoDriver:
            _ = (uri, auth)
            return fake_driver

    monkeypatch.setattr(
        "llm_graphrag_kg.providers.storage.neo4j_store._import_neo4j",
        lambda: _FakeGraphDB,
    )

    store = Neo4jGraphStore(
        default_tenant_id="tenant_x",
        default_acl_roles="public",
        auto_init_constraints=False,
    )
    out_ent = store.upsert_entity(Entity(entity_id="e1", name="Centrifugal Pump", entity_type="Equipment"))
    assert out_ent.entity_id == "e1"

    store.upsert_hyperedge(
        HyperEdge(
            hyperedge_id="he_1",
            condition_entity_ids=["e1", "e2"],
            target_entity_id="e3",
            relation_type="JOINT_INDICATES",
            source_chunk_ids=["c1"],
        )
    )
    hes = store.get_all_hyperedges(acl_roles=["public"])
    assert len(hes) == 1
    assert hes[0].hyperedge_id == "he_1"

    nodes, rels = store.get_k_hop_subgraph(["e1"], k_hops=2, acl_roles=["public"])
    assert len(nodes) == 1 and rels == []

    store.set_communities(
        [Community(community_id=0, title="C0", summary="Summary", entity_ids=["e1"], relation_count=0)],
        acl_partition="public",
    )
    # Verify every read Cypher query includes tenant_id + is_deleted + acl pre-filtering
    k_hop_cypher, k_hop_params = sess.queries[3]
    assert "is_deleted" in k_hop_cypher
    assert "WHERE role IN $roles" in k_hop_cypher
    assert k_hop_params["tid"] == "tenant_x"
    assert k_hop_params["roles"] == ["public"]
    store.close()
    assert fake_driver.closed


def test_factory_defaults_and_enterprise_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig()
    assert isinstance(create_graph_store(cfg), NetworkXGraphStore)
    assert isinstance(create_vector_store(cfg), LocalVectorStore)
    assert isinstance(create_memory_store(cfg), VerifiedMemoryStore)

    # Enterprise routing (mocked drivers)
    cur = _FakePgCursor()
    monkeypatch.setattr(
        "llm_graphrag_kg.providers.storage.pgvector_store._import_psycopg",
        lambda: _FakePsycopg(cur),
    )
    sess = _FakeNeoSession()

    class _FakeGraphDB:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str]) -> _FakeNeoDriver:
            _ = (uri, auth)
            return _FakeNeoDriver(sess)

    monkeypatch.setattr(
        "llm_graphrag_kg.providers.storage.neo4j_store._import_neo4j",
        lambda: _FakeGraphDB,
    )

    cfg.env.vector_backend = "pgvector"
    cfg.env.graph_backend = "neo4j"
    pg_vec = create_vector_store(cfg)
    neo_g = create_graph_store(cfg)
    pg_mem = create_memory_store(cfg)
    assert isinstance(pg_vec, PgVectorStore)
    assert isinstance(neo_g, Neo4jGraphStore)
    assert isinstance(pg_mem, PgVerifiedMemoryStore)
    # Keep Relation import covered cleanly
    rel = Relation(relation_id="r1", source_id="e1", target_id="e2", relation_type="CAUSES")
    assert rel.relation_type == "CAUSES"

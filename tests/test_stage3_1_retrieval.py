"""Stage 3.1 unit tests: OllamaNativeEmbeddingClient, BM25+RRF, SYNONYM_OF, and Query-to-Triple."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from llm_graphrag_kg.config import AppConfig, EnvSettings
from llm_graphrag_kg.core.schema import Entity, Relation, TextChunk
from llm_graphrag_kg.pipelines.hipporag import SemanticGraphIndex, build_synonym_edges
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.embedding.ollama_native import (
    EmbeddingDiskCache,
    OllamaNativeEmbeddingClient,
)
from llm_graphrag_kg.providers.factory import create_embedding_client
from llm_graphrag_kg.providers.llm.mock import MockLLMClient
from llm_graphrag_kg.providers.storage.local_vector import LocalVectorStore, tokenize_bm25
from llm_graphrag_kg.providers.storage.networkx_store import NetworkXGraphStore


class _KeywordSemanticEmbedder:
    """Deterministic toy semantic embedder grouping synonyms into nearby 4D vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for txt in texts:
            t = txt.lower()
            v = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)
            if any(k in t for k in ("immunosuppression", "transplant", "rejection", "免疫抑制")):
                v += np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
            if any(k in t for k in ("carcinoma", "skin cancer", "basal cell", "皮肤癌")):
                v += np.array([0.0, 1.0, 0.1, 0.0], dtype=np.float32)
            if any(k in t for k in ("misalignment", "2x vibration", "轴系不对中", "2倍频")):
                v += np.array([0.0, 0.0, 1.0, 0.1], dtype=np.float32)
            if any(k in t for k in ("bearing", "spalling", "轴承", "剥落")):
                v += np.array([0.0, 0.0, 0.1, 1.0], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            out.append((v / norm).tolist())
        return out


def test_bm25_bilingual_and_hybrid_rrf_ranking() -> None:
    assert "iso20816" in tokenize_bm25("ISO20816 轴系不对中 2倍频振动")
    assert "轴系" in tokenize_bm25("ISO20816 轴系不对中")

    store = LocalVectorStore()
    chunks = [
        TextChunk(
            chunk_id="c1",
            doc_id="d1",
            source="s1.md",
            text="Pump unit model XZ-9040 experienced severe cavitation and bearing heat.",
            index=0,
            start_char=0,
            end_char=70,
        ),
        TextChunk(
            chunk_id="c2",
            doc_id="d1",
            source="s1.md",
            text="General centrifugal pump maintenance guide for routine lubrication.",
            index=1,
            start_char=71,
            end_char=137,
        ),
        TextChunk(
            chunk_id="c3",
            doc_id="d1",
            source="s1.md",
            text="离心泵机组发生轴系不对中时，出现明显的2倍频振动升高与轴承温升。",
            index=2,
            start_char=138,
            end_char=172,
        ),
    ]
    # Deliberately give c2 a slightly higher fake dense score than c1, while BM25 matches XZ-9040 on c1
    vectors = [
        [0.85, 0.52, 0.0],
        [0.98, 0.19, 0.0],
        [0.0, 0.0, 1.0],
    ]
    store.add_chunks(chunks, vectors)

    bm25_hits = store.search_bm25("XZ-9040 cavitation", top_k=2)
    assert bm25_hits and bm25_hits[0][0].chunk_id == "c1"

    zh_hits = store.search_bm25("轴系不对中 2倍频振动", top_k=2)
    assert zh_hits and zh_hits[0][0].chunk_id == "c3"

    hybrid_hits = store.search_hybrid(
        [1.0, 0.0, 0.0],
        "XZ-9040 cavitation",
        top_k=2,
        bm25_weight=0.5,
    )
    assert hybrid_hits[0][0].chunk_id == "c1"

    # Dimension mismatch gives an actionable error
    with pytest.raises(ValueError, match="Query embedding dim"):
        store.search([1.0, 0.0], top_k=2)


def test_ollama_native_embedder_batching_cache_and_factory(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class _FakeResp:
        def __init__(self, body: dict[str, object]) -> None:
            self._raw = json.dumps(body).encode("utf-8")

        def read(self) -> bytes:
            return self._raw

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def _fake_urlopen(req, timeout=120):  # noqa: ANN001
        payload = json.loads(req.data.decode("utf-8"))
        calls.append(payload)
        batch = payload["input"]
        return _FakeResp({"embeddings": [[float(len(t)), 1.0, 0.0] for t in batch]})

    client = OllamaNativeEmbeddingClient(
        base_url="http://127.0.0.1:11434/v1",
        model="nomic-embed-text",
        batch_size=2,
        num_gpu=0,
        cache_dir=tmp_path / "cache",
    )
    assert client.endpoint == "http://127.0.0.1:11434/api/embed"

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        vecs1 = client.embed(["alpha", "beta", "alpha", "gamma"])
        assert len(vecs1) == 4
        assert vecs1[0] == vecs1[2]
        # 3 unique texts with batch_size=2 -> 2 HTTP calls, passing num_gpu=0
        assert len(calls) == 2
        assert calls[0]["options"] == {"num_gpu": 0, "num_ctx": 2048}
        assert client.last_telemetry["cache_hits"] == 1

        # Second call for identical inputs must hit SQLite cache with 0 HTTP calls
        vecs2 = client.embed(["alpha", "beta", "gamma"])
        assert len(calls) == 2
        assert vecs2[0] == vecs1[0]
        assert client.last_telemetry["cache_hits"] == 3

    # Verify disk cache persistence across instances
    cache = EmbeddingDiskCache(tmp_path / "cache" / "embed_cache.sqlite3")
    found = cache.get_many("nomic-embed-text", ["alpha"])
    assert len(found) == 1

    # Verify factory routing
    cfg = AppConfig(
        project_root=tmp_path,
        env=EnvSettings(
            embedding_provider="ollama_native",
            embedding_base_url="http://127.0.0.1:11434",
            embedding_model="nomic-embed-text",
            embedding_num_gpu=0,
        ),
    )
    created = create_embedding_client(cfg)
    assert isinstance(created, OllamaNativeEmbeddingClient)
    assert created.num_gpu == 0

    cfg_auto = AppConfig(
        project_root=tmp_path,
        env=EnvSettings(
            embedding_provider="openai_compat",
            embedding_base_url="http://127.0.0.1:11434/v1",
            embedding_model="bge-m3",
            embedding_num_gpu=0,
        ),
    )
    assert isinstance(create_embedding_client(cfg_auto), OllamaNativeEmbeddingClient)


def test_hipporag2_synonym_edges_and_query_to_triple(tmp_path: Path) -> None:
    gs = NetworkXGraphStore()
    e1 = Entity(
        entity_id="ent_1",
        name="organ transplant",
        entity_type="RiskFactor",
        description="Corpus entity `organ transplant` extracted from GraphRAG-Bench (medical).",
        source_chunk_ids=["doc::c1"],
    )
    e2 = Entity(
        entity_id="ent_2",
        name="immunosuppression",
        entity_type="RiskFactor",
        description="Long-term transplant rejection therapy.",
        source_chunk_ids=["doc::c2"],
    )
    e3 = Entity(
        entity_id="ent_3",
        name="basal cell carcinoma",
        entity_type="Disease",
        description="Common non-melanoma skin cancer.",
        source_chunk_ids=["doc::c2"],
    )
    for e in (e1, e2, e3):
        gs.upsert_entity(e)
    gs.upsert_relation(
        Relation(
            relation_id="rel_1",
            source_id="ent_2",
            target_id="ent_3",
            relation_type="INCREASES_RISK_OF",
            description="Immunosuppression raises basal cell carcinoma risk.",
            weight=2.0,
            source_chunk_ids=["doc::c2"],
        )
    )

    embedder = _KeywordSemanticEmbedder()
    added, ent_ids, ent_vecs = build_synonym_edges(gs, embedder, cosine_threshold=0.90)
    assert added == 1
    syn_rels = [r for r in gs.get_all_relations() if r.relation_type == "SYNONYM_OF"]
    assert len(syn_rels) == 1
    assert syn_rels[0].source_chunk_ids  # SHACL-lite zero-ghost compliant

    sem_idx = SemanticGraphIndex()
    stats = sem_idx.build(gs, embedder, precomputed_entities=(ent_ids, ent_vecs))
    assert stats["indexed_entities"] == 3
    assert stats["indexed_triples"] == 1  # SYNONYM_OF excluded from Query-to-Triple list

    sem_idx.save(tmp_path)
    reloaded = SemanticGraphIndex()
    assert reloaded.load(tmp_path) is True

    q_vec = embedder.embed(["Does skin cancer risk rise after transplant?"])[0]
    triple_hits = reloaded.search_triples(q_vec, top_k=3)
    assert triple_hits and triple_hits[0]["relation_id"] == "rel_1"


def test_pipeline_end_to_end_stage3_1_persistence_and_ppr(tmp_path: Path) -> None:
    in_dir = tmp_path / "docs"
    st_dir = tmp_path / "storage"
    in_dir.mkdir()
    (in_dir / "pump.md").write_text(
        "# Centrifugal Pump Diagnosis\n\n"
        "Shaft misalignment causes elevated 2x vibration and motor current fluctuation. "
        "Bearing spalling causes high-frequency shock pulses.",
        encoding="utf-8",
    )

    cfg = AppConfig(project_root=tmp_path)
    pipeline = GraphRAGPipeline(
        cfg,
        llm=MockLLMClient(),
        embedder=_KeywordSemanticEmbedder(),
    )
    stats = pipeline.build(in_dir, storage_dir=st_dir)
    assert stats.entity_count >= 1
    assert (st_dir / "semantic_index.npz").is_file()
    assert (st_dir / "semantic_index_meta.json").is_file()

    # Reload into fresh pipeline and run PPR query
    fresh = GraphRAGPipeline(
        cfg,
        llm=MockLLMClient(),
        embedder=_KeywordSemanticEmbedder(),
    )
    fresh.load(st_dir)
    assert fresh.semantic_index.is_ready
    res = fresh.query("What causes 2x vibration in the pump?", mode="ppr", reasoning="direct")
    stage3_meta = res.metadata.get("retrieval_stage3_1", {})
    assert isinstance(stage3_meta, dict)
    assert stage3_meta.get("hybrid_bm25_rrf") is True
    assert any("hybrid_rrf" in ev.matched_via for ev in res.evidences)


def test_ollama_native_embedder_fix_a_num_ctx_fix_b_cpu_and_mrl(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    from llm_graphrag_kg.providers.embedding.ollama_native import OllamaNativeEmbeddingClient
    from llm_graphrag_kg.providers.factory import create_embedding_client

    captured_payloads: list[dict] = []

    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def _fake_urlopen(req, timeout=120):
        payload = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(payload)
        vecs = [[0.6, 0.8, 0.0, 0.0] for _ in payload["input"]]
        return _FakeResp(json.dumps({"embeddings": vecs}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    cfg = AppConfig(project_root=tmp_path)
    cfg.env.embedding_provider = "ollama_native"
    cfg.env.embedding_model = "qwen3-embedding:0.6b"
    cfg.env.embedding_dimensions = 2

    client = create_embedding_client(cfg)
    assert isinstance(client, OllamaNativeEmbeddingClient)
    # Default config must enforce BOTH Fix B (CPU num_gpu=0) and Fix A (num_ctx=2048)
    assert client.num_gpu == 0
    assert client.num_ctx == 2048

    vecs = client.embed(["离心泵轴系不对中"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 2  # MRL truncated + re-normalized
    assert abs((vecs[0][0] ** 2 + vecs[0][1] ** 2) - 1.0) < 1e-5
    assert captured_payloads[0]["options"] == {"num_gpu": 0, "num_ctx": 2048}

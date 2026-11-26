"""Unit tests for Stage 3.4: HyperGraph N-ary HyperEdges & Gated A-MEM Verified Write-Back."""

from __future__ import annotations

from pathlib import Path

from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.core.schema import Entity, Relation, TextChunk
from llm_graphrag_kg.pipelines.hypergraph import (
    create_hyperedge,
    format_hyperedge_context,
    match_hyperedges,
    mine_joint_hyperedges,
)
from llm_graphrag_kg.pipelines.memory_store import VerifiedMemoryStore, check_memory_conflict
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.embedding.local_hash import LocalHashEmbeddingClient
from llm_graphrag_kg.providers.llm.mock import MockLLMClient
from llm_graphrag_kg.providers.storage.local_vector import LocalVectorStore
from llm_graphrag_kg.providers.storage.networkx_store import NetworkXGraphStore


def _build_multi_symptom_store() -> tuple[NetworkXGraphStore, LocalVectorStore]:
    gs = NetworkXGraphStore()
    vs = LocalVectorStore()
    embedder = LocalHashEmbeddingClient()

    c1 = TextChunk(
        chunk_id="doc_pump::chunk_0000",
        doc_id="doc_pump",
        source="pump_diag.md",
        index=0,
        text="2倍频振动升高、轴承温升异常与电机定子电流波动共同指向轴系不对中，需执行激光对中 SOP。",
        start_char=0,
        end_char=52,
    )
    vs.add_chunks([c1], embedder.embed([c1.text]))

    ents = [
        Entity(entity_id="ent_s1", name="2倍频振动升高", entity_type="Symptom", source_chunk_ids=[c1.chunk_id]),
        Entity(entity_id="ent_s2", name="轴承温升异常", entity_type="Symptom", source_chunk_ids=[c1.chunk_id]),
        Entity(entity_id="ent_s3", name="电机定子电流波动", entity_type="Symptom", source_chunk_ids=[c1.chunk_id]),
        Entity(entity_id="ent_fc", name="轴系不对中", entity_type="FaultMode", source_chunk_ids=[c1.chunk_id]),
        Entity(entity_id="ent_other", name="转子动不平衡", entity_type="FaultMode", source_chunk_ids=[c1.chunk_id]),
    ]
    for e in ents:
        gs.upsert_entity(e)

    rels = [
        Relation(
            relation_id="rel_1",
            source_id="ent_s1",
            target_id="ent_fc",
            relation_type="INDICATES",
            source_chunk_ids=[c1.chunk_id],
        ),
        Relation(
            relation_id="rel_2",
            source_id="ent_s2",
            target_id="ent_fc",
            relation_type="INDICATES",
            source_chunk_ids=[c1.chunk_id],
        ),
        Relation(
            relation_id="rel_3",
            source_id="ent_s3",
            target_id="ent_fc",
            relation_type="INDICATES",
            source_chunk_ids=[c1.chunk_id],
        ),
        Relation(
            relation_id="rel_4",
            source_id="ent_s1",
            target_id="ent_other",
            relation_type="INDICATES",
            source_chunk_ids=[c1.chunk_id],
        ),
    ]
    for r in rels:
        gs.upsert_relation(r)
    return gs, vs


def test_hyperedge_mining_persistence_and_joint_matching(tmp_path: Path) -> None:
    gs, _ = _build_multi_symptom_store()
    mined = mine_joint_hyperedges(gs, min_conditions=2)

    # Only `ent_fc` (轴系不对中) has >=2 co-grounded symptoms; `ent_other` has only 1
    assert len(mined) == 1
    he = mined[0]
    assert he.target_entity_id == "ent_fc"
    assert set(he.condition_entity_ids) == {"ent_s1", "ent_s2", "ent_s3"}
    assert "doc_pump::chunk_0000" in he.source_chunk_ids

    # Roundtrip persistence in graph.json
    gs.save(tmp_path)
    gs2 = NetworkXGraphStore()
    gs2.load(tmp_path)
    assert len(gs2.get_all_hyperedges()) == 1
    assert gs2.get_all_hyperedges()[0].hyperedge_id == he.hyperedge_id

    ent_map = {e.entity_id: e for e in gs2.get_all_entities()}
    # Single symptom should NOT trigger 3-condition hyperedge (coverage=0.33 < 0.60)
    assert match_hyperedges(["ent_s1"], gs2.get_all_hyperedges(), ent_map, min_match_ratio=0.60) == []

    # Multi-symptom match (2/3 or 3/3) MUST fire cleanly
    matches = match_hyperedges(["ent_s1", "ent_s2", "ent_s3"], gs2.get_all_hyperedges(), ent_map)
    assert len(matches) == 1
    assert matches[0].target_entity_name == "轴系不对中"
    assert matches[0].coverage_ratio == 1.0
    ctx = format_hyperedge_context(matches)
    assert "JOINT_INDICATES" in ctx and "轴系不对中" in ctx


def test_verified_memory_anti_poisoning_gates_and_recall(tmp_path: Path) -> None:
    gs, _ = _build_multi_symptom_store()
    store = VerifiedMemoryStore(tmp_path)

    # 1. Ungrounded answer MUST be rejected by Gate 1
    dec_ungrounded = store.evaluate_and_write_back(
        question="轴系不对中怎么诊断？",
        answer="根据经验判断一定是轴系不对中。",
        grounding_score=0.5,
        ungrounded_citations=["ent_fake"],
        confidence=0.95,
        condition_entity_ids=["ent_s1", "ent_s2"],
        target_entity_ids=["ent_fc"],
        source_chunk_ids=["doc_pump::chunk_0000"],
    )
    assert dec_ungrounded.accepted is False
    assert dec_ungrounded.gate_failed == "GATE_1_GROUNDING"

    # 2. Refusal answer MUST be rejected by Gate 2
    dec_refusal = store.evaluate_and_write_back(
        question="轴系不对中怎么诊断？",
        answer="The provided context does not specify the root cause.",
        grounding_score=1.0,
        confidence=0.9,
        condition_entity_ids=["ent_s1", "ent_s2"],
        target_entity_ids=["ent_fc"],
        source_chunk_ids=["doc_pump::chunk_0000"],
    )
    assert dec_refusal.accepted is False
    assert dec_refusal.gate_failed == "GATE_2_SUBSTANCE"

    # 3. Low confidence MUST be rejected by Gate 2
    dec_lowconf = store.evaluate_and_write_back(
        question="轴系不对中怎么诊断？",
        answer="2倍频振动升高与轴承温升异常联合指向轴系不对中，应执行激光对中 SOP。",
        grounding_score=1.0,
        confidence=0.45,
        condition_entity_ids=["ent_s1", "ent_s2"],
        target_entity_ids=["ent_fc"],
        source_chunk_ids=["doc_pump::chunk_0000"],
    )
    assert dec_lowconf.accepted is False
    assert dec_lowconf.gate_failed == "GATE_2_CONFIDENCE"

    # 4. 100%-grounded, high-confidence, valid conclusion MUST be accepted & promote HyperEdge
    dec_ok = store.evaluate_and_write_back(
        question="2倍频振动升高且轴承温升异常的根因是什么？",
        answer="依据原文切片，2倍频振动升高与轴承温升异常联合指向轴系不对中，需执行激光对中 SOP。",
        grounding_score=1.0,
        confidence=0.92,
        condition_entity_ids=["ent_s1", "ent_s2"],
        target_entity_ids=["ent_fc"],
        source_chunk_ids=["doc_pump::chunk_0000"],
        reasoning_paradigm="tot_dfs",
        graph_store=gs,
    )
    assert dec_ok.accepted is True
    assert dec_ok.record is not None
    assert dec_ok.promoted_hyperedge_id is not None
    assert (tmp_path / VerifiedMemoryStore.FILENAME).is_file()

    # 5. Polarity contradiction poisoning MUST be blocked by Gate 3
    dec_poison_neg = store.evaluate_and_write_back(
        question="2倍频振动升高且轴承温升异常的根因是什么？",
        answer="2倍频振动升高与轴承温升异常并非轴系不对中引起，排除轴系不对中。",
        grounding_score=1.0,
        confidence=0.95,
        condition_entity_ids=["ent_s1", "ent_s2"],
        target_entity_ids=["ent_fc"],
        source_chunk_ids=["doc_pump::chunk_0000"],
    )
    assert dec_poison_neg.accepted is False
    assert dec_poison_neg.gate_failed == "GATE_3_CONFLICT_NLI"

    # 6. Competing exclusive target on identical conditions MUST also be blocked by Gate 3
    has_conflict, _ = check_memory_conflict(
        conclusion="2倍频振动升高与轴承温升异常指示转子动不平衡。",
        condition_entity_ids=["ent_s1", "ent_s2"],
        target_entity_ids=["ent_other"],
        existing_records=store.get_all(),
    )
    assert has_conflict is True

    # 7. Reload from disk and recall verified episodic memory
    store2 = VerifiedMemoryStore(tmp_path)
    loaded = store2.load()
    assert len(loaded) == 1
    recalled = store2.retrieve_relevant("2倍频振动升高和轴承温升异常原因", ["ent_s1", "ent_s2"])
    assert len(recalled) == 1
    assert recalled[0].memory_id == dec_ok.record.memory_id


def test_end_to_end_pipeline_hypergraph_and_writeback(tmp_path: Path) -> None:
    gs, vs = _build_multi_symptom_store()
    gs.upsert_hyperedge(
        create_hyperedge(
            ["ent_s1", "ent_s2", "ent_s3"],
            "ent_fc",
            description="2倍频振动升高 + 轴承温升异常 + 电机定子电流波动 => 轴系不对中",
            source_chunk_ids=["doc_pump::chunk_0000"],
        )
    )
    gs.save(tmp_path)
    vs.save(tmp_path)

    cfg = load_config(llm_provider_override="mock", embedding_provider_override="local_hash")
    pipeline = GraphRAGPipeline(
        cfg,
        llm=MockLLMClient(),
        embedder=LocalHashEmbeddingClient(),
        graph_store=gs,
        vector_store=vs,
    )
    pipeline.memory_store.storage_dir = tmp_path

    res = pipeline.query(
        "2倍频振动升高、轴承温升异常且电机定子电流波动，根因是什么？",
        mode="ppr",
        reasoning="cot",
        auto_load=False,
        enable_writeback=True,
    )
    assert "matched_hyperedges" in res.metadata
    assert res.metadata["matched_hyperedges"][0]["target_entity_name"] == "轴系不对中"
    assert "writeback_decision" in res.metadata
    assert isinstance(res.metadata["writeback_decision"]["accepted"], bool)

"""Unit tests for Stage 2.2: Explicit Graph-CoT, GroundingAudit, PPR retrieval, and Scope-Aware L0."""

from __future__ import annotations

import json
from pathlib import Path

from llm_graphrag_kg.adapters.failuresensoriq import (
    _build_graphrag_mcqa_context,
    _infer_relevancy_label,
    build_failuresensoriq_kg,
    evaluate_failuresensoriq,
)
from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.core.schema import Entity, EvidenceItem, Relation
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.factory import (
    create_embedding_client,
    create_graph_store,
    create_vector_store,
)
from llm_graphrag_kg.providers.llm.mock import MockLLMClient
from llm_graphrag_kg.reasoning.cot import GraphCoTEngine, audit_cot_grounding
from llm_graphrag_kg.reasoning.schema import ReasoningStep


class _RepairingCoTLLM:
    """Stub LLM: first response fabricates a relation ID; second response fixes it."""

    def __init__(self, real_eid: str, real_rid: str, real_cid: str) -> None:
        self.calls = 0
        self.real_eid = real_eid
        self.real_rid = real_rid
        self.real_cid = real_cid

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        _ = (prompt, system_prompt, temperature, json_mode, max_tokens)
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "anchor_entities": [self.real_eid],
                    "steps": [
                        {
                            "step_id": 1,
                            "stage": "anchor",
                            "claim": "定位设备节点",
                            "cited_entity_ids": [self.real_eid],
                            "cited_relation_ids": [],
                            "cited_chunk_ids": [],
                        },
                        {
                            "step_id": 2,
                            "stage": "graph_traverse",
                            "claim": "编造了一条不存在的边",
                            "cited_entity_ids": [self.real_eid],
                            "cited_relation_ids": ["rel_fabricated_999"],
                            "cited_chunk_ids": [],
                        },
                    ],
                    "final_answer": "初版回答",
                    "confidence": 0.7,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "anchor_entities": [self.real_eid],
                "steps": [
                    {
                        "step_id": 1,
                        "stage": "anchor",
                        "claim": "定位离心泵锚点",
                        "cited_entity_ids": [self.real_eid],
                        "cited_relation_ids": [],
                        "cited_chunk_ids": [],
                    },
                    {
                        "step_id": 2,
                        "stage": "graph_traverse",
                        "claim": "核对真实图谱因果边",
                        "cited_entity_ids": [self.real_eid],
                        "cited_relation_ids": [self.real_rid],
                        "cited_chunk_ids": [self.real_cid],
                    },
                ],
                "final_answer": f"修复后的扎根结论 [{self.real_cid}]",
                "confidence": 0.95,
            },
            ensure_ascii=False,
        )

    def health_check(self) -> tuple[bool, str]:
        return True, "ok"


def test_audit_cot_grounding_detects_fabricated_citations_and_passes_grounded() -> None:
    entities = [
        Entity(entity_id="ent_pump", name="离心泵", entity_type="Equipment", source_chunk_ids=["c1"]),
        Entity(entity_id="ent_vib", name="2倍频振动升高", entity_type="Symptom", source_chunk_ids=["c1"]),
    ]
    relations = [
        Relation(
            relation_id="rel_01",
            source_id="ent_pump",
            target_id="ent_vib",
            relation_type="EXHIBITS",
            source_chunk_ids=["c1"],
        )
    ]
    evidences = [
        EvidenceItem(
            chunk_id="c1",
            doc_id="d1",
            source="doc.md",
            text="离心泵 2倍频振动 7.8 mm/s",
            score=1.0,
            matched_via="graph",
        )
    ]

    bad_steps = [
        ReasoningStep(
            step_id=1,
            stage="graph_traverse",
            claim="命中真实实体但编造了边",
            cited_entity_ids=["离心泵"],
            cited_relation_ids=["rel_fake_404"],
            cited_chunk_ids=["c1"],
        )
    ]
    audited_bad, audit_bad = audit_cot_grounding(
        bad_steps, entities=entities, relations=relations, evidences=evidences
    )
    assert not audit_bad.passed
    assert "relation:rel_fake_404" in audit_bad.ungrounded_citations
    assert not audited_bad[0].grounded
    assert audited_bad[0].cited_entity_ids == ["ent_pump"]

    good_steps = [
        ReasoningStep(
            step_id=1,
            stage="anchor",
            claim="锚定离心泵",
            cited_entity_ids=["离心泵"],
        ),
        ReasoningStep(
            step_id=2,
            stage="graph_traverse",
            claim="确认振动关联",
            cited_relation_ids=["rel_01"],
            cited_chunk_ids=["c1"],
        ),
        ReasoningStep(
            step_id=3,
            stage="synthesis",
            claim="综合给出诊断结论",
        ),
    ]
    audited_good, audit_good = audit_cot_grounding(
        good_steps, entities=entities, relations=relations, evidences=evidences
    )
    assert audit_good.passed
    assert audit_good.grounding_score == 1.0
    assert all(s.grounded for s in audited_good)


def test_graph_cot_engine_self_corrects_ungrounded_citation() -> None:
    entities = [
        Entity(entity_id="ent_pump", name="离心泵", entity_type="Equipment", source_chunk_ids=["d1::chunk_0000"]),
        Entity(entity_id="ent_mis", name="轴系不对中", entity_type="FaultMode", source_chunk_ids=["d1::chunk_0000"]),
    ]
    relations = [
        Relation(
            relation_id="rel_real_1",
            source_id="ent_pump",
            target_id="ent_mis",
            relation_type="HAS_FAULT",
            source_chunk_ids=["d1::chunk_0000"],
        )
    ]
    evidences = [
        EvidenceItem(
            chunk_id="d1::chunk_0000",
            doc_id="d1",
            source="fault.md",
            text="离心泵轴系不对中时 2倍频振动显著升高。",
            score=1.0,
            matched_via="graph",
        )
    ]
    stub = _RepairingCoTLLM("ent_pump", "rel_real_1", "d1::chunk_0000")
    harness = LLMHarness(stub, max_calls=6)
    engine = GraphCoTEngine(harness, self_correct_grounding=True)

    res = engine.run(
        question="离心泵 2倍频升高原因？",
        seed_entity_ids=["ent_pump"],
        entities=entities,
        relations=relations,
        communities=[],
        evidences=evidences,
    )
    assert stub.calls == 2
    assert res.grounding_audit.repair_triggered is True
    assert res.grounding_audit.passed is True
    assert res.grounding_audit.grounding_score == 1.0
    assert "d1::chunk_0000" in res.final_answer


def test_ppr_retrieval_and_cot_pipeline_e2e(tmp_path: Path) -> None:
    cfg = load_config(
        config_path=Path("configs/fault_diagnosis.yaml"),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    storage_dir = tmp_path / "storage"
    pipeline = GraphRAGPipeline(cfg)
    stats = pipeline.build(Path("data/sample_fault_diagnosis"), storage_dir=storage_dir)
    assert stats.entity_count > 5

    result = pipeline.query(
        "离心泵机组 2倍频振动升高应如何排查和执行 SOP？",
        mode="ppr",
        reasoning="cot",
        storage_dir=storage_dir,
        enable_compact=True,
    )
    assert result.mode == "ppr"
    assert result.metadata["reasoning_paradigm"] == "cot"
    assert "ppr_top_nodes" in result.metadata
    reasoning_meta = result.metadata["reasoning"]
    assert isinstance(reasoning_meta, dict)
    audit = reasoning_meta["grounding_audit"]
    assert audit["passed"] is True
    assert audit["grounding_score"] == 1.0
    assert len(reasoning_meta["steps"]) == 4


def test_scope_aware_l0_and_negation_keywords_on_10_options(tmp_path: Path) -> None:
    # 1. Verify expanded negation keyword recognition
    for q_text in (
        "In steam turbine, which failure event is unimportant if the sensor power shows an abnormal reading?",
        "For pump, which sensor should be disregarded for monitoring misalignment?",
        "In electric motor, which sensor among the choices is least useful for detecting brush fault?",
        "For aero gas turbine, what is the non-relevant failure event when fuel flow is abnormal?",
    ):
        assert _infer_relevancy_label({"question": q_text}).startswith("irrelevant_")

    # 2. Build miniature FailureSensorIQ KG with 3 failure modes & 3 sensors for `pump`
    mapped_path = tmp_path / "mini.mapped.jsonl"
    pairs = [
        ("bearing damage", "vibration"),
        ("bearing damage", "temperature"),
        ("damaged seals", "fluid leakage"),
        ("damaged seals", "vibration"),
        ("misalignment", "vibration"),
    ]
    rows = [
        {
            "id": i,
            "asset": "pump",
            "anchor": fm,
            "anchor_class": "FailureMode",
            "option": sensor,
            "option_class": "SignalChannel",
            "relation": "monitoredBy",
            "gold": True,
            "relevant": True,
        }
        for i, (fm, sensor) in enumerate(pairs)
    ]
    mapped_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    cfg = load_config(
        config_path=Path("configs/failuresensoriq.yaml"),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    docs_dir = tmp_path / "docs"
    storage_dir = tmp_path / "storage"
    build_failuresensoriq_kg(mapped_path, cfg, docs_dir=docs_dir, storage_dir=storage_dir)

    gs = create_graph_store()
    gs.load(storage_dir)
    vs = create_vector_store()
    vs.load(storage_dir)
    emb = create_embedding_client(cfg)

    # Question asks which failure mode is unimportant when `temperature` is abnormal
    # Valid pump failure modes: bearing damage (monitored), damaged seals & misalignment (unmonitored)
    # Out-of-asset distractors: stuck, delayed operation, internal leakage
    q_row = {
        "id": 101,
        "question": "In pump, which failure event is unimportant if the sensor temperature shows an abnormal reading?",
        "options": ["stuck", "bearing damage", "damaged seals", "delayed operation", "internal leakage"],
        "option_ids": ["A", "B", "C", "D", "E"],
        "correct": [False, False, True, False, False],
    }
    ctx, _, _ = _build_graphrag_mcqa_context(
        q_row, "pump", gs, vs, emb, compact_anchor_only=True
    )
    assert "(L0 Scope-Aware ISO Complete List)" in ctx
    assert "monitors ONLY these FailureModes: [bearing damage]" in ctx
    assert "Other Valid ISO FailureModes NOT monitored by `temperature`" in ctx
    assert "damaged seals" in ctx
    assert "stuck" not in ctx  # proves zero option peeking!

    # 3. End-to-end MCQA CoT evaluation should nail option C (`damaged seals`)
    q_file = tmp_path / "q10.jsonl"
    q_file.write_text(json.dumps(q_row) + "\n", encoding="utf-8")
    summary, _ = evaluate_failuresensoriq(
        questions_path=q_file,
        config=cfg,
        storage_dir=storage_dir,
        output_dir=tmp_path / "out",
        limit=1,
        modes=["graphrag_kg"],
        reasoning_paradigm="cot",
        resume=False,
        llm=MockLLMClient(),
        embedder=emb,
    )
    assert summary.modes[0].accuracy == 1.0

"""Unit tests for Stage 2.3: Explicit Controllable Graph-ToT Engine & L2 Branch Folding."""

from __future__ import annotations

import json
from pathlib import Path

from llm_graphrag_kg.adapters.graphrag_bench import (
    compute_evidence_coverage,
    compute_rouge_l_f1,
    compute_token_f1,
    evaluate_graphrag_bench,
    ingest_graphrag_bench_corpus,
)
from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.core.schema import Entity, EvidenceItem, Relation
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.factory import create_graph_store, create_vector_store
from llm_graphrag_kg.reasoning.tot import GraphToTEngine


class _BudgetCountingToTLLM:
    """Stub LLM tracking exact call counts and returning structured ToT JSON."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        _ = (system_prompt, temperature, json_mode, max_tokens)
        if "GRAPH_TOT_BRANCH_TASK" in prompt:
            self.calls.append("branch")
            return json.dumps(
                {
                    "branches": [
                        {
                            "branch_id": "H1",
                            "hypothesis": "轴系不对中导致2倍频振动与电流波动",
                            "target_entity_id": "ent_misalign",
                            "target_entity_name": "轴系不对中",
                            "rationale": "典型2倍频主导特征",
                        },
                        {
                            "branch_id": "H2",
                            "hypothesis": "转子动不平衡导致振动升高",
                            "target_entity_id": "ent_unbal",
                            "target_entity_name": "转子动不平衡",
                            "rationale": "旋转机械常见振动原因",
                        },
                        {
                            "branch_id": "H3",
                            "hypothesis": "轴承外圈剥落导致振动升高",
                            "target_entity_id": "ent_bearing",
                            "target_entity_name": "轴承外圈剥落",
                            "rationale": "轴承高频冲击特征",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if "GRAPH_TOT_EVAL_TASK" in prompt:
            self.calls.append("eval")
            return json.dumps(
                {
                    "evaluations": [
                        {
                            "branch_id": "H1",
                            "semantic_score": 0.96,
                            "supporting_relations": ["rel_m1", "rel_m2"],
                            "supporting_chunks": ["doc_1::chunk_0000"],
                            "contradicting_evidences": [],
                            "prune_reason": "",
                        },
                        {
                            "branch_id": "H2",
                            "semantic_score": 0.30,
                            "supporting_relations": ["rel_u1"],
                            "supporting_chunks": ["doc_2::chunk_0000"],
                            "contradicting_evidences": ["转子动不平衡以1倍频为主，不支持2倍频 7.8 mm/s"],
                            "prune_reason": "缺少1倍频主导特征且不解释电机电流波动",
                        },
                        {
                            "branch_id": "H3",
                            "semantic_score": 0.25,
                            "supporting_relations": ["rel_b1"],
                            "supporting_chunks": ["doc_3::chunk_0000"],
                            "contradicting_evidences": ["未见高频冲击脉冲 BPFO 12.8 kHz"],
                            "prune_reason": "缺少高频冲击特征 BPFO",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        self.calls.append("synth")
        return json.dumps(
            {
                "selected_branch_id": "H1",
                "steps": [
                    {
                        "step_id": 1,
                        "stage": "anchor",
                        "claim": "锁定主假设轴系不对中及其关联症状。",
                        "cited_entity_ids": ["ent_misalign", "ent_2x"],
                        "cited_relation_ids": [],
                        "cited_chunk_ids": [],
                    },
                    {
                        "step_id": 2,
                        "stage": "graph_traverse",
                        "claim": "验证2倍频振动与电流波动同时指向轴系不对中。",
                        "cited_entity_ids": ["ent_misalign"],
                        "cited_relation_ids": ["rel_m1", "rel_m2"],
                        "cited_chunk_ids": ["doc_1::chunk_0000"],
                    },
                    {
                        "step_id": 3,
                        "stage": "synthesis",
                        "claim": "确诊轴系不对中并排除动不平衡与轴承剥落。",
                        "cited_entity_ids": ["ent_misalign"],
                        "cited_relation_ids": ["rel_m1"],
                        "cited_chunk_ids": ["doc_1::chunk_0000"],
                    },
                ],
                "excluded_summary": ["排除 H2（1倍频不符）", "排除 H3（无 BPFO 高频冲击）"],
                "final_answer": "确诊为轴系不对中 [doc_1::chunk_0000]，已剪枝转子动不平衡与轴承剥落。",
                "confidence": 0.94,
            },
            ensure_ascii=False,
        )

    def health_check(self) -> tuple[bool, str]:
        return True, "ok"


def test_graph_tot_branches_prunes_folds_l2_and_audits_grounding() -> None:
    gs = create_graph_store()
    vs = create_vector_store()

    entities = [
        Entity(
            entity_id="ent_2x",
            name="2倍频振动升高",
            entity_type="Symptom",
            source_chunk_ids=["doc_1::chunk_0000"],
        ),
        Entity(
            entity_id="ent_curr",
            name="电机电流波动",
            entity_type="Symptom",
            source_chunk_ids=["doc_1::chunk_0000"],
        ),
        Entity(
            entity_id="ent_misalign",
            name="轴系不对中",
            entity_type="FaultMode",
            source_chunk_ids=["doc_1::chunk_0000"],
        ),
        Entity(
            entity_id="ent_unbal",
            name="转子动不平衡",
            entity_type="FaultMode",
            source_chunk_ids=["doc_2::chunk_0000"],
        ),
        Entity(
            entity_id="ent_bearing",
            name="轴承外圈剥落",
            entity_type="FaultMode",
            source_chunk_ids=["doc_3::chunk_0000"],
        ),
    ]
    for e in entities:
        gs.upsert_entity(e)

    relations = [
        Relation(
            relation_id="rel_m1",
            source_id="ent_misalign",
            target_id="ent_2x",
            relation_type="CAUSES",
            source_chunk_ids=["doc_1::chunk_0000"],
        ),
        Relation(
            relation_id="rel_m2",
            source_id="ent_misalign",
            target_id="ent_curr",
            relation_type="CAUSES",
            source_chunk_ids=["doc_1::chunk_0000"],
        ),
        Relation(
            relation_id="rel_u1",
            source_id="ent_unbal",
            target_id="ent_2x",
            relation_type="WEAK_LINK",
            source_chunk_ids=["doc_2::chunk_0000"],
        ),
        Relation(
            relation_id="rel_b1",
            source_id="ent_bearing",
            target_id="ent_curr",
            relation_type="WEAK_LINK",
            source_chunk_ids=["doc_3::chunk_0000"],
        ),
    ]
    for r in relations:
        gs.upsert_relation(r)

    evidences = [
        EvidenceItem(
            chunk_id="doc_1::chunk_0000",
            doc_id="doc_1",
            source="misalign.md",
            text="轴系不对中会引发 2倍频振动 7.8 mm/s 升高及电机电流波动。",
            score=1.0,
            matched_via="graph",
        ),
        EvidenceItem(
            chunk_id="doc_2::chunk_0000",
            doc_id="doc_2",
            source="unbal.md",
            text="转子动不平衡以 1倍频为主，通常无电流波动。",
            score=0.7,
            matched_via="graph",
        ),
        EvidenceItem(
            chunk_id="doc_3::chunk_0000",
            doc_id="doc_3",
            source="bearing.md",
            text="轴承外圈剥落伴随高频冲击脉冲 12.8 kHz (BPFO)。",
            score=0.6,
            matched_via="graph",
        ),
    ]

    llm = _BudgetCountingToTLLM()
    harness = LLMHarness(llm, max_calls=6)
    engine = GraphToTEngine(
        harness,
        graph_store=gs,
        vector_store=vs,
        beam_width=3,
        max_depth=2,
        max_llm_calls=6,
    )

    result, _, _, _ = engine.run(
        question="离心泵同时出现2倍频振动升高和电机电流波动，根本原因是什么？",
        seed_entity_ids=["ent_2x", "ent_curr"],
        entities=entities,
        relations=relations,
        communities=[],
        evidences=evidences,
    )

    assert result.paradigm == "tot"
    assert llm.calls == ["branch", "eval", "synth"]
    assert len(result.branches) == 3
    selected = [b for b in result.branches if b.status == "selected"]
    pruned = [b for b in result.branches if b.status == "pruned"]
    assert len(selected) == 1 and selected[0].branch_id == "H1"
    assert len(pruned) == 2
    # L2 folding must preserve counter-evidence numeric measurements and chunk IDs
    assert len(result.pruned_memories) == 2
    assert any("12.8 kHz" in mem for mem in result.pruned_memories)
    assert any("doc_3::chunk_0000" in mem for mem in result.pruned_memories)
    assert result.grounding_audit.passed is True
    assert result.grounding_audit.grounding_score == 1.0


def test_pipeline_query_tot_e2e_with_mock(tmp_path: Path) -> None:
    cfg = load_config(
        config_path=Path("configs/fault_diagnosis.yaml"),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    storage_dir = tmp_path / "storage"
    pipeline = GraphRAGPipeline(cfg)
    pipeline.build(Path("data/sample_fault_diagnosis"), storage_dir=storage_dir)

    result = pipeline.query(
        "离心泵机组出现2倍频振动升高且电机电流波动，应如何定位根因并排除其他故障？",
        mode="ppr",
        reasoning="tot",
        storage_dir=storage_dir,
        enable_compact=True,
    )
    assert result.mode == "ppr"
    assert result.metadata["reasoning_paradigm"] == "tot"
    rm = result.metadata["reasoning"]
    assert isinstance(rm, dict)
    assert len(rm["branches"]) >= 2
    assert sum(1 for b in rm["branches"] if b["status"] == "selected") == 1
    assert sum(1 for b in rm["branches"] if b["status"] == "pruned") >= 1
    assert rm["grounding_audit"]["passed"] is True
    assert rm["trace_ledger"]["total_calls"] <= cfg.reasoning.tot_max_llm_calls


def test_graphrag_bench_ingest_and_multi_paradigm_eval_offline(tmp_path: Path) -> None:
    corpus_file = tmp_path / "medical.json"
    corpus_text = (
        "Basal cell carcinoma (BCC) is the most common skin cancer. "
        "Fair skin and immune suppression from organ transplant are independent risk factors for BCC. "
        "A shiny bump on sun-exposed skin warrants a medical history, skin exam, and biopsy for diagnosis. "
        "Surgery is the primary treatment for BCC; recurrent BCC management depends on risk and site. "
    ) * 4
    corpus_file.write_text(
        json.dumps({"corpus_name": "MedicalMini", "context": corpus_text}),
        encoding="utf-8",
    )

    qs_file = tmp_path / "medical_questions.json"
    qs_file.write_text(
        json.dumps(
            [
                {
                    "id": "med-1",
                    "question": "Why are fair skin and organ transplant high risk factors for basal cell carcinoma?",
                    "answer": (
                        "Fair skin and immune suppression from organ transplant "
                        "are independent risk factors for BCC."
                    ),
                    "question_type": "Complex Reasoning",
                    "evidence": (
                        "Fair skin is an independent risk factor for BCC.; "
                        "Immune suppression from organ transplant is a risk factor."
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(
        config_path=Path("configs/graphrag_bench.yaml"),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    stats = ingest_graphrag_bench_corpus(
        corpus_file,
        cfg,
        docs_dir=tmp_path / "docs",
        storage_dir=tmp_path / "storage",
        max_chars=0,
        fast_lexical=True,
    )
    assert stats.entity_count >= 3 and stats.relation_count >= 2

    report, report_md = evaluate_graphrag_bench(
        qs_file,
        cfg,
        storage_dir=tmp_path / "storage",
        output_dir=tmp_path / "out",
        question_type="Complex Reasoning",
        limit=1,
        modes=["naive_rag", "hybrid_direct", "ppr_cot", "ppr_tot"],
    )
    assert report_md.is_file()
    assert len(report.modes) == 4
    assert compute_token_f1("fair skin organ transplant", "fair skin and organ transplant") == 1.0
    assert compute_rouge_l_f1("fair skin organ transplant", "fair skin organ transplant") == 1.0
    assert compute_evidence_coverage("fair skin risk factor", "fair skin is a risk factor") > 0.8

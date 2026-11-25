"""Stage 3.2 unit tests: Adaptive Query Router, Multi-Branch Non-Pruning Synthesis, and ToT fix."""

from __future__ import annotations

from pathlib import Path

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.schema import Entity, EvidenceItem, Relation
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.embedding.local_hash import LocalHashEmbeddingClient
from llm_graphrag_kg.providers.llm.mock import MockLLMClient
from llm_graphrag_kg.reasoning.router import route_query


def _sample_evidences(text: str) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id=f"doc_1::chunk_000{i}",
            doc_id="doc_1",
            source="doc.md",
            text=text,
            score=0.9,
            matched_via="hybrid_rrf",
        )
        for i in range(3)
    ]


def test_adaptive_router_distinguishes_exclusive_multi_facet_and_early_stop() -> None:
    # 1. Exclusive diagnosis -> Graph-ToT (competitive pruning)
    dec_diag = route_query(
        "离心泵同时出现2倍频振动升高和电流波动，应优先排查什么根因并排除无关故障？"
    )
    assert dec_diag.route == "exclusive_diagnosis"
    assert dec_diag.target_paradigm == "tot"
    assert dec_diag.early_stop is False

    dec_en_diag = route_query(
        "Which root cause is most likely and which sensor should we rule out?"
    )
    assert dec_en_diag.route == "exclusive_diagnosis"
    assert dec_en_diag.target_paradigm == "tot"

    # 2. Multi-facet summary / comparison -> Multi-Branch additive synthesis (non-pruning)
    dec_sum = route_query(
        "Summarize how fair skin, UV exposure, and organ transplant immunosuppression "
        "collectively contribute to basal cell carcinoma risk across patients."
    )
    assert dec_sum.route == "multi_facet_synthesis"
    assert dec_sum.target_paradigm == "multi_branch"
    assert dec_sum.early_stop is False

    # 3. Direct factoid -> Fast Path CoT with early stop
    dec_fact = route_query("What is Mohs surgery?")
    assert dec_fact.route == "fact_direct"
    assert dec_fact.target_paradigm == "cot"
    assert dec_fact.early_stop is True

    # 4. High first-hop coverage triggers Dynamic Early Stopping
    seed = Entity(
        entity_id="ent_bcc",
        name="basal cell carcinoma",
        entity_type="Disease",
        description="Slow-growing skin tumor treated via surgical excision.",
        source_chunk_ids=["doc_1::chunk_0000"],
    )
    rel = Relation(
        relation_id="rel_1",
        source_id="ent_bcc",
        target_id="ent_bcc",
        relation_type="TREATED_BY",
        description="excision biopsy recurrence margin",
        source_chunk_ids=["doc_1::chunk_0000"],
    )
    evs = _sample_evidences(
        "Basal cell carcinoma excision margin biopsy recurrence rates remain very low."
    )
    dec_early = route_query(
        "Basal cell carcinoma excision margin recurrence biopsy?",
        seed_entities=[seed],
        relations=[rel],
        evidences=evs,
        enable_early_stop=True,
        early_stop_coverage_threshold=0.80,
    )
    assert dec_early.route == "fact_direct"
    assert dec_early.target_paradigm == "cot"
    assert dec_early.early_stop is True
    assert dec_early.coverage_score >= 0.80


def test_multi_branch_non_pruning_preserves_complementary_branches_and_full_passages(
    tmp_path: Path,
) -> None:
    in_dir = tmp_path / "docs"
    st_dir = tmp_path / "storage"
    in_dir.mkdir()
    long_passage = (
        "Basal cell carcinoma (BCC) develops primarily after chronic UV exposure. "
        "Patients with fair skin and history of organ transplant require long-term immunosuppression, "
        "which significantly multiplies skin cancer incidence by 10x to 65x. "
        "Mohs micrographic surgery offers the highest cure rate with tissue preservation, "
        "and regular dermatological screening is mandatory for early lesion detection."
    )
    assert len(long_passage) > 300  # Exceeds old Stage 2.3 220-char truncation cap
    (in_dir / "medical.md").write_text(f"# BCC Clinical Guide\n\n{long_passage}\n", encoding="utf-8")

    cfg = AppConfig(project_root=tmp_path)
    pipeline = GraphRAGPipeline(
        cfg,
        llm=MockLLMClient(),
        embedder=LocalHashEmbeddingClient(),
    )
    pipeline.build(in_dir, storage_dir=st_dir)

    # Multi-branch additive synthesis uses only 2 LLM calls (Branch + Additive Union Synthesis)
    res_mb = pipeline.query(
        "Summarize the risk factors and surgical treatments for basal cell carcinoma.",
        mode="ppr",
        reasoning="multi_branch",
        auto_load=False,
    )
    assert res_mb.metadata["reasoning_paradigm"] == "multi_branch"
    rm = res_mb.metadata["reasoning"]
    assert isinstance(rm, dict)
    assert rm["paradigm"] == "multi_branch"
    assert rm["grounding_audit"]["passed"] is True
    assert res_mb.metadata["harness_ledger"]["total_calls"] == 2
    selected_cnt = sum(1 for b in rm["branches"] if b["status"] == "selected")
    assert selected_cnt >= 1

    # Auto mode routes exclusive diagnosis -> tot, and summary -> multi_branch
    res_auto_sum = pipeline.query(
        "Summarize how UV exposure and immunosuppression influence basal cell carcinoma.",
        mode="ppr",
        reasoning="auto",
        auto_load=False,
    )
    assert res_auto_sum.metadata["route_decision"]["route"] == "multi_facet_synthesis"
    assert res_auto_sum.metadata["reasoning_paradigm"] == "multi_branch"

    res_auto_diag = pipeline.query(
        "Which root cause is most likely and which factor should we rule out for BCC?",
        mode="ppr",
        reasoning="auto",
        auto_load=False,
    )
    assert res_auto_diag.metadata["route_decision"]["route"] == "exclusive_diagnosis"
    assert res_auto_diag.metadata["reasoning_paradigm"] == "tot"

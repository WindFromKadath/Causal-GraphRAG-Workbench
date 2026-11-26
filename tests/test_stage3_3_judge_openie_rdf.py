"""Unit tests for Stage 3.3: 4D LLM-as-a-Judge, Checkpointed Semantic OpenIE, and RDF FMEA Importer."""

from __future__ import annotations

import json
from pathlib import Path

from llm_graphrag_kg.adapters.rdf_fmea import ingest_fmea_rdf_ttl, parse_turtle_triples
from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.core.schema import TextChunk
from llm_graphrag_kg.eval.llm_judge import (
    LLMJudgeEvaluator,
    detect_refusal,
    fbeta_score,
    judge_grb_checkpoints,
)
from llm_graphrag_kg.pipelines.extraction import (
    SemanticOpenIEExtractor,
    normalize_openie_predicate,
)
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.embedding.local_hash import LocalHashEmbeddingClient
from llm_graphrag_kg.providers.llm.mock import MockLLMClient
from llm_graphrag_kg.viz.schema_validator import validate_kg_shacl_lite


def test_detect_refusal_and_fbeta_guardrails() -> None:
    assert detect_refusal("The provided context does not specify the route of metastasis.") is True
    assert detect_refusal("信息不足，无法确定根本原因。") is True
    assert (
        detect_refusal(
            "Radioactive iodine therapy is recommended for iodine-avid metastatic differentiated "
            "thyroid cancer, while targeted therapy is used for refractory cases."
        )
        is False
    )
    assert abs(fbeta_score(tp=2, fp=0, fn=2, beta=1.0) - (2 / 3)) < 1e-4
    assert fbeta_score(tp=0, fp=2, fn=2, beta=1.0) == 0.0


def test_4d_llm_judge_scoring_and_checkpoint_resume(tmp_path: Path) -> None:
    llm = MockLLMClient()
    embedder = LocalHashEmbeddingClient()
    evaluator = LLMJudgeEvaluator(llm, embedder)

    good = evaluator.score_one(
        question_id="q1",
        question_type="Complex Reasoning",
        mode="ppr_cot_iter",
        question="How is iodine-avid thyroid cancer treated?",
        prediction=(
            "Radioactive iodine therapy is recommended for iodine-avid thyroid cancer. "
            "Targeted therapy is used when cancer is not iodine-avid."
        ),
        reference=(
            "Radioactive iodine therapy treats iodine-avid thyroid cancer. "
            "Targeted therapy treats non-iodine-avid thyroid cancer."
        ),
        contexts=[
            "Radioactive iodine therapy treats iodine-avid thyroid cancer. "
            "Targeted therapy may be used if the cancer is not iodine-avid."
        ],
        retrieval_evidence_recall=0.9,
        token_f1_legacy=0.75,
    )
    refusal = evaluator.score_one(
        question_id="q1",
        question_type="Complex Reasoning",
        mode="naive_rag",
        question="How is iodine-avid thyroid cancer treated?",
        prediction="The retrieved context does not mention how to treat this cancer.",
        reference=(
            "Radioactive iodine therapy treats iodine-avid thyroid cancer. "
            "Targeted therapy treats non-iodine-avid thyroid cancer."
        ),
        contexts=["Unrelated passage."],
        retrieval_evidence_recall=0.0,
        token_f1_legacy=0.1,
    )

    assert good.is_refusal is False
    assert good.answer_accuracy > refusal.answer_accuracy
    assert good.semantic_coverage > refusal.semantic_coverage
    assert refusal.is_refusal is True
    assert refusal.factuality_f1 == 0.0

    # End-to-end checkpoint judge + resume test
    qs_file = tmp_path / "questions.json"
    qs_file.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "How is iodine-avid thyroid cancer treated?",
                    "answer": "Radioactive iodine therapy treats iodine-avid thyroid cancer.",
                    "question_type": "Complex Reasoning",
                    "evidence": ["Radioactive iodine therapy treats iodine-avid thyroid cancer."],
                }
            ]
        ),
        encoding="utf-8",
    )
    ckpt_file = tmp_path / "grb_ppr_auto.checkpoint.jsonl"
    ckpt_file.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_type": "Complex Reasoning",
                "mode": "ppr_auto",
                "token_f1": 0.7,
                "retrieval_evidence_recall": 0.9,
                "prediction": "Radioactive iodine therapy treats iodine-avid thyroid cancer.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(llm_provider_override="mock", embedding_provider_override="local_hash")
    report, report_path = judge_grb_checkpoints(
        questions_path=qs_file,
        checkpoint_map={"ppr_auto": ckpt_file},
        config=cfg,
        output_dir=tmp_path / "judge_out",
        tag="unit_test",
        llm=llm,
        embedder=embedder,
    )
    assert report_path.is_file()
    assert len(report.modes) == 1
    assert report.modes[0].avg_answer_accuracy > 0.5
    assert report.modes[0].refusal_rate == 0.0

    # Resume should reuse cached JSONL with 0 errors
    report2, _ = judge_grb_checkpoints(
        questions_path=qs_file,
        checkpoint_map={"ppr_auto": ckpt_file},
        config=cfg,
        output_dir=tmp_path / "judge_out",
        tag="unit_test",
        resume=True,
        llm=llm,
        embedder=embedder,
    )
    assert report2.modes[0].avg_answer_accuracy == report.modes[0].avg_answer_accuracy


def test_checkpointed_semantic_openie_extractor(tmp_path: Path) -> None:
    assert normalize_openie_predicate("co_occurs_with") == "ASSOCIATED_WITH"
    assert normalize_openie_predicate("treats / mitigates") == "TREATS_MITIGATES"

    chunks = [
        TextChunk(
            chunk_id="doc_1::chunk_0000",
            doc_id="doc_1",
            source="sample.md",
            text="GraphRAG combines Knowledge Graph and LLM to solve multi-hop reasoning.",
            index=0,
            start_char=0,
            end_char=71,
        ),
        TextChunk(
            chunk_id="doc_1::chunk_0001",
            doc_id="doc_1",
            source="sample.md",
            text="滚动轴承外圈剥落 causes 高频冲击脉冲, diagnosed via 包络解调.",
            index=1,
            start_char=72,
            end_char=112,
        ),
    ]
    extractor = SemanticOpenIEExtractor(MockLLMClient(), domain="industrial")
    ckpt = tmp_path / "openie.checkpoint.jsonl"
    ents, rels, stats = extractor.extract_chunks_checkpointed(
        chunks, checkpoint_path=ckpt, resume=True, progress_every=0
    )
    assert len(ents) >= 3
    assert len(rels) >= 1
    assert all(r.relation_type != "CO_OCCURS_WITH" for r in rels)
    assert stats.total_chunks_processed == 2
    assert stats.resumed_from_checkpoint == 0

    # Second pass must resume 100% from checkpoint
    _, _, stats2 = extractor.extract_chunks_checkpointed(
        chunks, checkpoint_path=ckpt, resume=True, progress_every=0
    )
    assert stats2.resumed_from_checkpoint == 2


def test_industrial_rdf_turtle_importer_passes_shacl_100(tmp_path: Path) -> None:
    sample_ttl = tmp_path / "sample_fmea.ttl"
    sample_ttl.write_text(
        """@prefix : <http://empwr.ai/entity/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:entity1 rdf:type :Entity ;
  rdfs:label "SHH" .

:entity2 rdf:type :Concept ;
  rdfs:label "Spurious high alarm level" .

:entity3 rdf:type :Organization ;
  rdfs:label "safety_and_control" .

:entity4 rdf:type :Organization ;
  rdfs:label "Gas detection" .

:entity5 rdf:type :Concept ;
  rdfs:label "Gas detection" .

:entity6 rdf:type :Concept ;
  rdfs:label "Isolated orphan node" .

:entity1 :description :entity2 .
:entity1 :asset_name :entity3 .
:entity1 :component :entity4 .
:entity1 :component :entity5 .
:entity4 :related_entity :entity5 .
""",
        encoding="utf-8",
    )

    prefixes, triples = parse_turtle_triples(sample_ttl.read_text(encoding="utf-8"))
    assert prefixes[""] == "http://empwr.ai/entity/"
    assert len(triples) == 17

    cfg = load_config(
        config_path=Path("configs/fmea_kg_rdf.yaml"),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    storage = tmp_path / "fmea_storage"
    report = ingest_fmea_rdf_ttl(
        sample_ttl, cfg, storage_dir=storage, embedder=LocalHashEmbeddingClient()
    )
    assert report.merged_duplicate_labels == 1  # entity4 & entity5 ("Gas detection") merged
    assert report.skipped_self_loops == 1  # entity4 -> entity5 became self-loop and skipped
    assert report.skipped_isolated == 1  # entity6 dropped cleanly
    assert report.audit_stats.entity_count == 4
    assert report.audit_stats.relation_count >= 3

    # Verify 100% SHACL-lite PASS (0 errors, 0 warnings) + live PPR query
    pipeline = GraphRAGPipeline(cfg, llm=MockLLMClient(), embedder=LocalHashEmbeddingClient())
    pipeline.load(storage)
    shacl = validate_kg_shacl_lite(pipeline.graph_store, pipeline.vector_store, cfg)
    assert shacl.overall_status == "PASS"
    assert shacl.compliance_score == 1.0
    assert shacl.error_count == 0
    assert shacl.warning_count == 0

    res = pipeline.query("What does OREDA failure code SHH affect?", mode="ppr", reasoning="auto")
    assert res.evidences
    assert res.answer

"""End-to-end and unit tests for the Phase 1 LLM + GraphRAG + KG framework."""

from __future__ import annotations

from pathlib import Path

from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.pipelines.extraction import parse_json_payload
from llm_graphrag_kg.pipelines.ingestion import DocumentIngestor
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.viz.audit_report import export_audit_artifacts
from llm_graphrag_kg.viz.graphml_exporter import export_graphml
from llm_graphrag_kg.viz.html_exporter import export_pyvis_html


def test_parse_json_payload_handles_fences() -> None:
    raw = 'Here is the result:\n```json\n{"entities": [{"name": "GraphRAG"}], "relations": []}\n```'
    parsed = parse_json_payload(raw)
    assert len(parsed["entities"]) == 1
    assert parsed["entities"][0]["name"] == "GraphRAG"


def test_document_ingestor_preserves_provenance() -> None:
    sample_dir = Path("data/sample")
    ingestor = DocumentIngestor(chunk_size=180, chunk_overlap=30)
    docs = ingestor.load_directory(sample_dir, base_dir=Path.cwd())
    assert len(docs) >= 3
    for doc in docs:
        assert not Path(doc.source).is_absolute()

    chunks = ingestor.chunk_documents(docs)
    assert len(chunks) >= len(docs)
    assert all("::chunk_" in c.chunk_id for c in chunks)


def test_end_to_end_build_query_and_visual_exports(tmp_path: Path) -> None:
    cfg = load_config(
        project_root=Path.cwd(),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    storage_dir = tmp_path / "storage"
    output_dir = tmp_path / "outputs"

    pipeline = GraphRAGPipeline(cfg)
    stats = pipeline.build(
        input_dir=Path("data/sample"),
        storage_dir=storage_dir,
        summarize_communities=True,
    )

    assert stats.doc_count == 3
    assert stats.chunk_count >= 3
    assert stats.entity_count >= 10
    assert stats.relation_count >= 5
    assert stats.community_count >= 1

    # Reload from disk into a fresh pipeline instance to verify persistence portability
    reloaded = GraphRAGPipeline(cfg)
    loaded_docs = reloaded.load(storage_dir)
    assert loaded_docs == 3

    for mode in ("local", "global", "hybrid"):
        result = reloaded.query("GraphRAG 如何结合 NetworkX 和 PyVis？", mode=mode, auto_load=False)
        assert result.mode == mode
        assert result.answer
        assert len(result.trace.expanded_entity_ids) >= 1

    hybrid_res = reloaded.query("GraphRAG 如何支持迁移至 Neo4j？", mode="hybrid", auto_load=False)
    assert len(hybrid_res.evidences) >= 1

    # Verify visual verification artifacts
    _, audit_paths = export_audit_artifacts(
        reloaded.graph_store, reloaded.vector_store, output_dir, doc_count=loaded_docs
    )
    assert audit_paths["audit_report"].is_file()
    assert audit_paths["entities_csv"].is_file()
    assert audit_paths["relations_csv"].is_file()

    html_path = export_pyvis_html(
        reloaded.graph_store,
        output_dir / "trace.html",
        vector_store=reloaded.vector_store,
        color_by="community",
        trace=hybrid_res.trace,
    )
    assert html_path.is_file()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Trace Highlight:" in html_text

    graphml_path = export_graphml(reloaded.graph_store, output_dir / "kg.graphml")
    assert graphml_path.is_file()
    assert "<graphml" in graphml_path.read_text(encoding="utf-8")


def test_fault_diagnosis_domain_config_and_corpus(tmp_path: Path) -> None:
    cfg = load_config(
        config_path=Path("configs/fault_diagnosis.yaml"),
        project_root=Path.cwd(),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    pipeline = GraphRAGPipeline(cfg)
    stats = pipeline.build(storage_dir=tmp_path / "fd_storage")

    assert stats.doc_count == 6
    assert stats.chunk_count >= 10
    assert stats.entity_count >= 30
    assert stats.relation_count >= 30

    res = pipeline.query(
        "离心泵机组出现2倍频振动升高和电机定子电流波动时，可能根因与处置SOP是什么？",
        mode="hybrid",
        auto_load=False,
    )
    assert res.answer
    assert len(res.trace.seed_entity_ids) >= 2
    assert len(res.trace.expanded_entity_ids) >= len(res.trace.seed_entity_ids)
    assert len(res.evidences) >= 2


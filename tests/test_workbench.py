"""Unit tests for the Phase 2 Streamlit visual verification workbench helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.viz.html_exporter import render_pyvis_html
from llm_graphrag_kg.viz.workbench_data import (
    build_chunk_records,
    build_community_records,
    build_entity_records,
    build_relation_records,
    filter_entities,
    filter_relations,
    resolve_provenance_chunks,
)


def test_workbench_data_builders_filters_and_provenance(tmp_path: Path) -> None:
    cfg = load_config(
        project_root=Path.cwd(),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    pipeline = GraphRAGPipeline(cfg)
    pipeline.build(input_dir=Path("data/sample"), storage_dir=tmp_path / "storage")

    chunk_rows = build_chunk_records(pipeline.vector_store)
    entity_rows = build_entity_records(pipeline.graph_store)
    relation_rows = build_relation_records(pipeline.graph_store)
    community_rows = build_community_records(pipeline.graph_store)

    assert len(chunk_rows) >= 3
    assert len(entity_rows) >= 10
    assert len(relation_rows) >= 5
    assert len(community_rows) >= 1

    # Filter entities by keyword and type
    graphrag_hits = filter_entities(entity_rows, keyword="GraphRAG")
    assert any(row["name"] == "GraphRAG" for row in graphrag_hits)

    method_only = filter_entities(entity_rows, entity_types=["Method"])
    assert method_only and all(row["entity_type"] == "Method" for row in method_only)

    # Filter relations linked to filtered entity subset
    allowed_ids = {row["entity_id"] for row in graphrag_hits}
    linked_rels = filter_relations(relation_rows, allowed_entity_ids=allowed_ids)
    assert len(linked_rels) >= 1
    assert all(
        row["source_id"] in allowed_ids or row["target_id"] in allowed_ids
        for row in linked_rels
    )

    # Provenance lookup resolves chunk IDs back to full source text
    sample_entity = graphrag_hits[0]
    resolved_chunks = resolve_provenance_chunks(
        sample_entity["source_chunk_ids"], pipeline.vector_store
    )
    assert len(resolved_chunks) >= 1
    assert resolved_chunks[0].text

    # Filtered PyVis HTML rendering + HTML escaping safety
    html_str = render_pyvis_html(
        pipeline.graph_store,
        vector_store=pipeline.vector_store,
        allowed_entity_ids=allowed_ids,
        title="Test <Filtered> Graph",
    )
    assert "Test &lt;Filtered&gt; Graph" in html_str
    assert f"Nodes={len(allowed_ids)}" in html_str


def test_streamlit_workbench_headless_render() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "src" / "llm_graphrag_kg" / "viz" / "app.py"
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) == 1


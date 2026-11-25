"""Unit tests for the Phase 2 Streamlit visual verification workbench helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_graphrag_kg.adapters.failuresensoriq import build_failuresensoriq_kg
from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.core.schema import Entity, Relation
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.factory import create_graph_store, create_vector_store
from llm_graphrag_kg.viz.html_exporter import render_pyvis_html
from llm_graphrag_kg.viz.schema_validator import validate_kg_shacl_lite
from llm_graphrag_kg.viz.workbench_data import (
    build_chunk_records,
    build_community_records,
    build_entity_records,
    build_readable_citation_maps,
    build_relation_records,
    extract_ego_subgraph,
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


def test_ego_graph_readable_citations_and_shacl_lite_validator(tmp_path: Path) -> None:
    import json

    mapped_path = tmp_path / "iso_mini.mapped.jsonl"
    rows = [
        {
            "id": 1,
            "asset": "pump",
            "anchor": "damaged seals",
            "anchor_class": "FailureMode",
            "option": "vibration",
            "option_class": "SignalChannel",
            "relation": "monitoredBy",
            "gold": True,
            "relevant": True,
        },
        {
            "id": 2,
            "asset": "pump",
            "anchor": "bearing wear",
            "anchor_class": "FailureMode",
            "option": "temperature",
            "option_class": "SignalChannel",
            "relation": "monitoredBy",
            "gold": True,
            "relevant": True,
        },
        {
            "id": 3,
            "asset": "electric motor",
            "anchor": "rotor windings fault",
            "anchor_class": "FailureMode",
            "option": "cooling gas",
            "option_class": "SignalChannel",
            "relation": "monitoredBy",
            "gold": True,
            "relevant": True,
        },
    ]
    mapped_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    cfg = load_config(
        config_path=Path("configs/failuresensoriq.yaml"),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    storage_dir = tmp_path / "storage"
    build_failuresensoriq_kg(
        mapped_path, cfg, docs_dir=tmp_path / "docs", storage_dir=storage_dir
    )

    gs = create_graph_store()
    gs.load(storage_dir)
    vs = create_vector_store()
    vs.load(storage_dir)

    # 1. Clean 4-layer ISO KG must pass all 8 SHACL-lite checks with 100% compliance
    clean_report = validate_kg_shacl_lite(gs, vs, cfg)
    assert clean_report.profile == "iso_failuresensoriq"
    assert clean_report.overall_status == "PASS"
    assert clean_report.compliance_score == 1.0
    assert clean_report.error_count == 0
    assert "SHACL-lite 本体合规质检报告" in clean_report.to_markdown()

    # 2. Ego-Graph centered on `pump` isolates pump nodes from `electric motor`
    pump_ent = next(e for e in gs.get_all_entities() if e.name == "pump")
    allowed, trace, summary = extract_ego_subgraph(gs, pump_ent.entity_id, k_hops=2, max_nodes=20)
    assert pump_ent.entity_id in allowed
    assert summary["ego_nodes"] < summary["total_nodes"]
    assert summary["reduction_pct"] > 0.0
    assert trace.seed_entity_ids == [pump_ent.entity_id]

    ent_labels, rel_labels = build_readable_citation_maps(gs)
    assert ent_labels[pump_ent.entity_id] == "pump [Equipment]"
    assert any("--[MONITORED_BY]-->" in lbl for lbl in rel_labels.values())

    # 3. Inject defects: ghost entity + self-loop + raw typo leak + domain/range violation
    gs.upsert_entity(
        Entity(
            entity_id="ent_bad_typo",
            name="temparature",
            entity_type="SignalChannel",
            source_chunk_ids=[],
        )
    )
    gs.upsert_relation(
        Relation(
            relation_id="rel_bad_loop",
            source_id=pump_ent.entity_id,
            target_id=pump_ent.entity_id,
            relation_type="MONITORED_BY",
            source_chunk_ids=[],
        )
    )
    bad_report = validate_kg_shacl_lite(gs, vs, cfg)
    assert bad_report.overall_status == "FAIL"
    assert bad_report.error_count >= 3
    failed_ids = {c.check_id for c in bad_report.checks if not c.passed}
    assert "SHACL_04_PROVENANCE_COMPLETENESS" in failed_ids
    assert "ISO_06_DOMAIN_RANGE_CONSTRAINTS" in failed_ids
    assert "ISO_08_SIGNAL_ER_HYGIENE" in failed_ids


def test_streamlit_workbench_headless_render() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "src" / "llm_graphrag_kg" / "viz" / "app.py"
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) == 1
    # Verify Stage 2.4 defaults to locked primary dataset `configs/failuresensoriq.yaml`
    assert at.sidebar.selectbox[0].value == "configs/failuresensoriq.yaml"
    assert len(at.tabs) == 6


def test_tot_tree_exporter_builds_interactive_html_for_tot_and_cot(tmp_path: Path) -> None:
    from llm_graphrag_kg.config import AppConfig
    from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
    from llm_graphrag_kg.providers.embedding.local_hash import LocalHashEmbeddingClient
    from llm_graphrag_kg.providers.llm.mock import MockLLMClient
    from llm_graphrag_kg.viz.tot_tree_exporter import build_tot_tree_payload, export_tot_tree_html

    in_dir = tmp_path / "docs"
    st_dir = tmp_path / "storage"
    in_dir.mkdir()
    (in_dir / "pump_case.md").write_text(
        "# 离心泵故障诊断\n\n"
        "离心泵机组出现2倍频振动升高、轴承温升异常与电机定子电流波动时，优先排查轴系不对中并执行激光对中SOP。\n"
        "滚动轴承外圈剥落通常伴随高频冲击信号。\n",
        encoding="utf-8",
    )

    cfg = AppConfig(project_root=tmp_path)
    pipeline = GraphRAGPipeline(
        cfg,
        llm=MockLLMClient(),
        embedder=LocalHashEmbeddingClient(),
    )
    pipeline.build(in_dir, storage_dir=st_dir)

    tot_res = pipeline.query(
        "离心泵2倍频振动升高和电流波动优先排查什么？",
        mode="ppr",
        reasoning="tot",
        auto_load=False,
    )
    payload = build_tot_tree_payload(tot_res, pipeline.graph_store, pipeline.vector_store)
    assert payload["paradigm"] == "tot"
    assert len(payload["branches"]) >= 1
    assert any(b["status"] == "selected" for b in payload["branches"])
    assert len(payload["chunks"]) >= 1
    # Verify Node -> Edge -> Node traversal waves and final core convergence stats
    trav = payload["traversal"]
    assert len(trav["nodes"]) >= 1
    assert trav["nodes"][0]["read_order"] == 1
    assert trav["stats"]["used_nodes"] >= 1
    assert trav["stats"]["used_nodes"] <= trav["stats"]["total_read_nodes"]

    out_html = export_tot_tree_html(
        tot_res,
        pipeline.graph_store,
        tmp_path / "tot_reasoning_tree.html",
        vector_store=pipeline.vector_store,
    )
    html_text = out_html.read_text(encoding="utf-8")
    assert "基于思维树的透明推理与因果链条解析" in html_text
    assert "drawTreeLines" in html_text
    assert "drawHopEdges" in html_text
    assert "仅保留最终用到的一片核心 Node" in html_text
    assert "https://" not in html_text  # Strictly zero external CDN dependencies

    # Verify graceful single-branch fallback on CoT mode
    cot_res = pipeline.query(
        "离心泵2倍频振动升高原因？",
        mode="ppr",
        reasoning="cot",
        auto_load=False,
    )
    cot_payload = build_tot_tree_payload(cot_res, pipeline.graph_store, pipeline.vector_store)
    assert cot_payload["paradigm"] == "cot"
    assert len(cot_payload["branches"]) == 1
    assert cot_payload["branches"][0]["status"] == "selected"


def test_workbench_discovers_medical_and_novel_configs_and_auto_detects_embedder(
    tmp_path: Path,
) -> None:
    pytest.importorskip("streamlit")
    import json

    import numpy as np

    from llm_graphrag_kg.viz.app import (
        _detect_storage_embedding_profile,
        _discover_config_files,
    )

    cfgs = _discover_config_files()
    assert cfgs[0] == "configs/failuresensoriq.yaml"
    assert "configs/graphrag_bench_medical.yaml" in cfgs
    assert "configs/graphrag_bench_novel.yaml" in cfgs

    # 1. Stage 3.1 storage with manifest embedding metadata
    s31 = tmp_path / "s31"
    s31.mkdir()
    (s31 / "manifest.json").write_text(
        json.dumps(
            {
                "embedding_provider": "ollama_native",
                "embedding_model": "nomic-embed-text",
                "synonym_edges_added": 236,
            }
        ),
        encoding="utf-8",
    )
    np.save(s31 / "embeddings.npy", np.zeros((3, 768), dtype=np.float32))
    prov, model, dim, _ = _detect_storage_embedding_profile(s31)
    assert (prov, model, dim) == ("ollama_native", "nomic-embed-text", 768)

    # 2. Legacy 256-dim storage without embedding fields in manifest
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    np.save(legacy / "embeddings.npy", np.zeros((2, 256), dtype=np.float32))
    prov2, model2, dim2, _ = _detect_storage_embedding_profile(legacy)
    assert (prov2, model2, dim2) == ("local_hash", "local_hash", 256)


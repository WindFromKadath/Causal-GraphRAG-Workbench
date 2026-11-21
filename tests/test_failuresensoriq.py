"""Unit and integration tests for the IBM FailureSensorIQ adapter & evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

from llm_graphrag_kg.adapters.failuresensoriq import (
    build_failuresensoriq_kg,
    evaluate_failuresensoriq,
    normalize_sensor_name,
)
from llm_graphrag_kg.config import load_config


def test_sensor_entity_resolution_fixes_real_typos() -> None:
    name, merged = normalize_sensor_name("temparature")
    assert name == "temperature" and merged is True

    name2, merged2 = normalize_sensor_name("coast down")
    assert name2 == "coast down time" and merged2 is True

    name3, merged3 = normalize_sensor_name("Vibration")
    assert name3 == "vibration" and merged3 is False


def test_failuresensoriq_ingest_and_eval_e2e(tmp_path: Path) -> None:
    # Build synthetic mini mapped JSONL mirroring real IBM FailureSensorIQ structure (with typo!)
    mapped_rows = [
        {
            "id": 0,
            "asset": "pump",
            "anchor": "bearing damage",
            "anchor_class": "FailureMode",
            "option": "vibration",
            "option_class": "SignalChannel",
            "relation": "monitoredBy",
            "gold": True,
            "relevant": True,
        },
        {
            "id": 1,
            "asset": "pump",
            "anchor": "bearing damage",
            "anchor_class": "FailureMode",
            "option": "temparature",  # deliberate dataset typo to verify ER merge
            "option_class": "SignalChannel",
            "relation": "monitoredBy",
            "gold": True,
            "relevant": True,
        },
        {
            "id": 2,
            "asset": "electric motor",
            "anchor": "bearing damage",
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
            "option": "current",
            "option_class": "SignalChannel",
            "relation": "monitoredBy",
            "gold": True,
            "relevant": True,
        },
        # Negative row should be ignored during gold KG construction
        {
            "id": 4,
            "asset": "pump",
            "anchor": "bearing damage",
            "anchor_class": "FailureMode",
            "option": "voltage",
            "option_class": "SignalChannel",
            "relation": "monitoredBy",
            "gold": False,
            "relevant": False,
        },
    ]
    mapped_file = tmp_path / "mini_mapped.jsonl"
    mapped_file.write_text(
        "\n".join(json.dumps(r) for r in mapped_rows) + "\n", encoding="utf-8"
    )

    qa_rows = [
        {
            "id": 10,
            "asset_name": "pump",
            "question": "For pump, if bearing damage occurs, which sensor is most relevant?",
            "options": ["voltage", "vibration", "current"],
            "option_ids": ["A", "B", "C"],
            "correct": [False, True, False],
            "relevancy": "relevant_sensors_for_failure_mode",
            "question_type": "mcp1_positive",
        },
        {
            "id": 11,
            "asset_name": "electric motor",
            "question": "In electric motor, for rotor windings fault, which sensor is relevant?",
            "options": ["voltage", "vibration", "current"],
            "option_ids": ["A", "B", "C"],
            "correct": [False, False, True],
            "relevancy": "relevant_sensors_for_failure_mode",
            "question_type": "mcp1_positive",
        },
    ]
    qa_file = tmp_path / "mini_qa.jsonl"
    qa_file.write_text("\n".join(json.dumps(r) for r in qa_rows) + "\n", encoding="utf-8")

    cfg = load_config(
        config_path=Path("configs/failuresensoriq.yaml"),
        project_root=Path.cwd(),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    docs_dir = tmp_path / "docs"
    storage_dir = tmp_path / "storage"
    output_dir = tmp_path / "outputs"

    ingest_stats = build_failuresensoriq_kg(
        mapped_file,
        cfg,
        docs_dir=docs_dir,
        storage_dir=storage_dir,
    )
    assert ingest_stats.asset_count == 2
    assert ingest_stats.failure_mode_count == 3  # asset-scoped (2 on motor + 1 on pump)
    assert ingest_stats.failure_category_count == 2  # shared `bearing damage` + `rotor windings fault`
    assert ingest_stats.sensor_count == 3  # `temparature` merged into `temperature`!
    assert ingest_stats.er_normalized_sensor_mentions == 1
    assert ingest_stats.audit.isolated_entities == []

    summary, report_path = evaluate_failuresensoriq(
        qa_file,
        cfg,
        storage_dir=storage_dir,
        output_dir=output_dir,
        limit=2,
        modes=["llm_only", "naive_rag", "graphrag_kg"],
    )
    assert report_path.is_file()
    assert summary.total_questions == 2
    by_mode = {m.mode: m for m in summary.modes}
    assert by_mode["graphrag_kg"].accuracy == 1.0

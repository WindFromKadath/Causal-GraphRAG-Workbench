"""Unit and integration tests for the IBM FailureSensorIQ adapter & evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

from llm_graphrag_kg.adapters.failuresensoriq import (
    build_failuresensoriq_kg,
    compute_pairwise_gaps,
    compute_set_f1,
    detect_asset_from_question,
    evaluate_failuresensoriq,
    normalize_option_id,
    normalize_sensor_name,
    parse_option_answers,
    plan_failuresensoriq_stress_suite,
    run_failuresensoriq_stress_suite,
)
from llm_graphrag_kg.config import load_config


def test_sensor_entity_resolution_fixes_real_typos() -> None:
    name, merged = normalize_sensor_name("temparature")
    assert name == "temperature" and merged is True

    name2, merged2 = normalize_sensor_name("coast down")
    assert name2 == "coast down time" and merged2 is True

    name3, merged3 = normalize_sensor_name("Vibration")
    assert name3 == "vibration" and merged3 is False


def test_metadata_free_asset_linking_and_perturbed_multi_answer_parsing() -> None:
    assets = ["electric motor", "industrial gas turbine", "pump", "aero gas turbine"]
    assert (
        detect_asset_from_question(
            "In industrial gas turbine, when blade crack occurs, which sensors apply?",
            assets,
        )
        == "industrial gas turbine"
    )
    assert normalize_option_id("P)") == "P"
    assert parse_option_answers('{"answers": ["P)", "S)"]}', ["P)", "Q)", "R)", "S)", "T)"], 2) == [
        "P",
        "S",
    ]
    assert parse_option_answers("Answer: B, E", ["A", "B", "C", "D", "E"], 2) == ["B", "E"]
    assert compute_set_f1(["A", "B"], ["A", "C"]) == 0.5
    assert compute_set_f1(["A", "B"], ["A", "B"]) == 1.0


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

    # Verify 4-suite stress runner with multi-answer + perturbed option labels + resume
    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    (suite_dir / "failuresensoriq_standard__all.jsonl").write_text(
        qa_file.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (suite_dir / "failuresensoriq_standard__all_10_options.jsonl").write_text(
        qa_file.read_text(encoding="utf-8"), encoding="utf-8"
    )
    multi_rows = [
        {
            "id": 100,
            "question": "For pump, if bearing damage occurs, which sensors are most relevant?",
            "options": ["voltage", "vibration", "current", "temperature", "resistance"],
            "option_ids": ["A", "B", "C", "D", "E"],
            "correct": [False, True, False, True, False],
        }
    ]
    (suite_dir / "failuresensoriq_standard__all_multi_answers.jsonl").write_text(
        "\n".join(json.dumps(r) for r in multi_rows) + "\n", encoding="utf-8"
    )
    perturbed_rows = [
        {
            "id": 200,
            "question": "In electric motor, for rotor windings fault, which sensor is relevant?",
            "options": ["voltage", "current", "vibration"],
            "option_ids": ["P)", "Q)", "R)"],
            "correct": [False, True, False],
        }
    ]
    (suite_dir / "failuresensoriq_perturbed__perturbed_simple.jsonl").write_text(
        "\n".join(json.dumps(r) for r in perturbed_rows) + "\n", encoding="utf-8"
    )

    stress_report, stress_md = run_failuresensoriq_stress_suite(
        suite_dir,
        cfg,
        storage_dir=storage_dir,
        output_dir=tmp_path / "stress_out",
        limit_per_suite=2,
        modes=["llm_only", "graphrag_kg"],
        concise_json=True,
        resume=True,
    )
    assert stress_md.is_file()
    assert len(stress_report.suites) == 4
    suite_map = {s.suite_name: {m.mode: m for m in s.modes} for s in stress_report.suites}
    assert suite_map["L3_multi_answers"]["graphrag_kg"].accuracy == 1.0
    assert suite_map["L4_perturbed_simple"]["graphrag_kg"].accuracy == 1.0
    assert suite_map["L4_perturbed_simple"]["graphrag_kg"].asset_linked_count == 1

    # Verify full-dataset mode (limit=0), graphrag_kg_raw vs graphrag_kg, dry-run plan & pairwise gaps
    plan = plan_failuresensoriq_stress_suite(
        suite_dir,
        cfg,
        output_dir=tmp_path / "stress_out",
        output_tag="full_check",
        limit_per_suite=0,
        modes=["llm_only", "graphrag_kg_raw", "graphrag_kg"],
    )
    assert len(plan) == 12
    assert all(row["remaining_calls"] == row["total_in_file"] for row in plan)

    full_report, full_md = run_failuresensoriq_stress_suite(
        suite_dir,
        cfg,
        storage_dir=storage_dir,
        output_dir=tmp_path / "stress_out",
        output_tag="full_check",
        limit_per_suite=0,
        modes=["llm_only", "graphrag_kg_raw", "graphrag_kg"],
        concise_json=True,
        resume=True,
        progress_every=1,
    )
    assert full_md.is_file()
    assert "Pairwise Win/Loss" in full_md.read_text(encoding="utf-8")
    gaps = compute_pairwise_gaps(full_report.suites[0].details, primary_mode="graphrag_kg")
    assert len(gaps) == 2

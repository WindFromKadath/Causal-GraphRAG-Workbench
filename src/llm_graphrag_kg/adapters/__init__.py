"""Structured dataset adapters for public industrial fault-diagnosis benchmarks."""

from llm_graphrag_kg.adapters.failuresensoriq import (
    DEFAULT_STRESS_SUITES,
    SENSOR_CANONICAL_MAP,
    EvaluationSummary,
    IngestStats,
    StressSuiteReport,
    build_failuresensoriq_kg,
    compute_set_f1,
    detect_asset_from_question,
    evaluate_failuresensoriq,
    normalize_option_id,
    normalize_sensor_name,
    parse_option_answers,
    run_failuresensoriq_stress_suite,
)

__all__ = [
    "DEFAULT_STRESS_SUITES",
    "EvaluationSummary",
    "IngestStats",
    "SENSOR_CANONICAL_MAP",
    "StressSuiteReport",
    "build_failuresensoriq_kg",
    "compute_set_f1",
    "detect_asset_from_question",
    "evaluate_failuresensoriq",
    "normalize_option_id",
    "normalize_sensor_name",
    "parse_option_answers",
    "run_failuresensoriq_stress_suite",
]

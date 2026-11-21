"""Structured dataset adapters for public industrial fault-diagnosis benchmarks."""

from llm_graphrag_kg.adapters.failuresensoriq import (
    SENSOR_CANONICAL_MAP,
    EvaluationSummary,
    IngestStats,
    build_failuresensoriq_kg,
    evaluate_failuresensoriq,
    normalize_sensor_name,
)

__all__ = [
    "EvaluationSummary",
    "IngestStats",
    "SENSOR_CANONICAL_MAP",
    "build_failuresensoriq_kg",
    "evaluate_failuresensoriq",
    "normalize_sensor_name",
]

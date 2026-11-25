"""Explicit structured reasoning paradigms (CoT, Iterative CoT v2, ToT, Deep ToT v2, Router)."""

from llm_graphrag_kg.reasoning.cot import GraphCoTEngine, audit_cot_grounding
from llm_graphrag_kg.reasoning.cot_iterative import IterativeGraphCoTEngine
from llm_graphrag_kg.reasoning.router import (
    RouteDecision,
    compute_first_hop_coverage,
    route_query,
)
from llm_graphrag_kg.reasoning.schema import (
    GraphCoTOutput,
    GroundingAudit,
    MCQACoTOutput,
    OptionGroundingItem,
    ReasoningResult,
    ReasoningStep,
    ThoughtBranch,
)
from llm_graphrag_kg.reasoning.tot import GraphToTEngine
from llm_graphrag_kg.reasoning.tot_deep import DeepGraphToTEngine

__all__ = [
    "DeepGraphToTEngine",
    "GraphCoTEngine",
    "GraphCoTOutput",
    "GraphToTEngine",
    "GroundingAudit",
    "IterativeGraphCoTEngine",
    "MCQACoTOutput",
    "OptionGroundingItem",
    "ReasoningResult",
    "ReasoningStep",
    "RouteDecision",
    "ThoughtBranch",
    "audit_cot_grounding",
    "compute_first_hop_coverage",
    "route_query",
]

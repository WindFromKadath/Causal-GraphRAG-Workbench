"""Structured schemas for Explicit Graph-CoT reasoning and deterministic GroundingAudit."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    """One explicit reasoning step bound to retrieved KG entities, relations, and chunks."""

    step_id: int
    stage: str = "graph_traverse"  # anchor | graph_traverse | evidence_check | synthesis
    claim: str
    cited_entity_ids: list[str] = Field(default_factory=list)
    cited_relation_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    grounded: bool = False
    grounding_notes: list[str] = Field(default_factory=list)


class GraphCoTOutput(BaseModel):
    """Structured LLM output contract for explicit 4-stage Graph-CoT."""

    anchor_entities: list[str] = Field(default_factory=list)
    steps: list[ReasoningStep] = Field(default_factory=list)
    final_answer: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class OptionGroundingItem(BaseModel):
    """Explicit tri-state KG grounding check for one MCQA candidate option."""

    option_id: str
    option_text: str = ""
    kg_status: Literal["LINKED", "IN_ASSET_UNLINKED", "OUT_OF_ASSET"] = "OUT_OF_ASSET"
    matched_kg_item: str = ""


class MCQACoTOutput(BaseModel):
    """Structured Graph-CoT output for MCQA preventing parametric-prior overrides."""

    target_anchor: str = ""
    polarity: Literal["relevant", "irrelevant"] = "relevant"
    option_checks: list[OptionGroundingItem] = Field(default_factory=list)
    reasoning_summary: str = ""
    answers: list[str] = Field(default_factory=list)


class GroundingAudit(BaseModel):
    """Deterministic faithfulness verification report over explicit reasoning steps."""

    total_steps: int = 0
    grounded_steps: int = 0
    grounding_score: float = 0.0
    valid_cited_entities: list[str] = Field(default_factory=list)
    valid_cited_relations: list[str] = Field(default_factory=list)
    valid_cited_chunks: list[str] = Field(default_factory=list)
    ungrounded_citations: list[str] = Field(default_factory=list)
    repair_triggered: bool = False
    passed: bool = False


class ReasoningResult(BaseModel):
    """Complete explicit reasoning trace returned by Graph-CoT / Graph-ToT engines."""

    question: str
    paradigm: str = "cot"  # direct | cot | tot
    anchor_entities: list[str] = Field(default_factory=list)
    final_answer: str
    confidence: float = 0.8
    steps: list[ReasoningStep] = Field(default_factory=list)
    grounding_audit: GroundingAudit = Field(default_factory=GroundingAudit)
    compaction: dict[str, object] | None = None
    trace_ledger: dict[str, object] | None = None

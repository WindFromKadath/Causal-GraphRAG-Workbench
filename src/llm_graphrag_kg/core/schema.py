"""Domain data models for LLM + GraphRAG + KG."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document loaded from disk or an external connector."""

    doc_id: str
    source: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    """A slice of a source document retaining character-level provenance."""

    chunk_id: str
    doc_id: str
    source: str
    index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A knowledge graph node representing an extracted entity."""

    entity_id: str
    name: str
    entity_type: str = "Concept"
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    community_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    """A directed edge connecting two entities with provenance and weight."""

    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HyperEdge(BaseModel):
    """N-ary multi-condition hyperedge (Stage 3.4 HyperGraphRAG).

    Represents joint multi-condition inference such as:
    `{Symptom_A, Symptom_B, Symptom_C} ===[JOINT_INDICATES]===> RootCause`
    """

    hyperedge_id: str
    condition_entity_ids: list[str] = Field(default_factory=list)
    target_entity_id: str
    relation_type: str = "JOINT_INDICATES"
    description: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    weight: float = 1.0
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifiedMemoryRecord(BaseModel):
    """Auditable episodic memory entry written back only after strict verification (Stage 3.4 A-MEM)."""

    memory_id: str
    question: str
    conclusion: str
    condition_entity_ids: list[str] = Field(default_factory=list)
    target_entity_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    grounding_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_paradigm: str = "auto"
    valid_from: str
    provenance_run_id: str
    conflict_checked: bool = True
    status: str = "verified"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Community(BaseModel):
    """A detected cluster of closely related entities and its summary."""

    community_id: int
    title: str
    summary: str
    entity_ids: list[str] = Field(default_factory=list)
    relation_count: int = 0


class EvidenceItem(BaseModel):
    """A retrieved chunk or provenance citation supporting an answer."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    score: float
    matched_via: str


class SubgraphTrace(BaseModel):
    """Trace of entities, relations, and communities used during GraphRAG retrieval."""

    seed_entity_ids: list[str] = Field(default_factory=list)
    expanded_entity_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    community_ids: list[int] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Structured answer returned by a GraphRAG query."""

    question: str
    mode: str
    answer: str
    trace: SubgraphTrace = Field(default_factory=SubgraphTrace)
    evidences: list[EvidenceItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphAuditStats(BaseModel):
    """Quality and structural audit metrics for visual/manual KG verification."""

    doc_count: int
    chunk_count: int
    entity_count: int
    relation_count: int
    community_count: int
    hyperedge_count: int = 0
    verified_memory_count: int = 0
    isolated_entities: list[str] = Field(default_factory=list)
    connected_components: int = 0
    entity_type_counts: dict[str, int] = Field(default_factory=dict)
    top_degree_entities: list[tuple[str, int]] = Field(default_factory=list)

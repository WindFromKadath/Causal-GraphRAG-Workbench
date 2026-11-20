"""Core domain schemas and abstract provider protocols."""

from llm_graphrag_kg.core.interfaces import (
    BaseEmbeddingClient,
    BaseGraphStore,
    BaseLLMClient,
    BaseVectorStore,
)
from llm_graphrag_kg.core.schema import (
    Community,
    Document,
    Entity,
    EvidenceItem,
    GraphAuditStats,
    QueryResult,
    Relation,
    SubgraphTrace,
    TextChunk,
)

__all__ = [
    "BaseEmbeddingClient",
    "BaseGraphStore",
    "BaseLLMClient",
    "BaseVectorStore",
    "Community",
    "Document",
    "Entity",
    "EvidenceItem",
    "GraphAuditStats",
    "QueryResult",
    "Relation",
    "SubgraphTrace",
    "TextChunk",
]

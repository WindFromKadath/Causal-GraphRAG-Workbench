"""Abstract interfaces (Ports) enabling smooth migration across LLM and storage backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import networkx as nx

from llm_graphrag_kg.core.schema import Community, Entity, Relation, TextChunk


@runtime_checkable
class BaseLLMClient(Protocol):
    """Protocol for any LLM backend (OpenAI-compatible API, local Ollama/vLLM, or Mock)."""

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """Generate a completion string from the model."""
        ...

    def health_check(self) -> tuple[bool, str]:
        """Verify connectivity and return (ok, human_readable_message)."""
        ...


@runtime_checkable
class BaseEmbeddingClient(Protocol):
    """Protocol for embedding backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into dense float vectors."""
        ...


@runtime_checkable
class BaseGraphStore(Protocol):
    """Protocol for knowledge graph persistence and traversal (NetworkX now, Neo4j/Kuzu later)."""

    def upsert_entity(self, entity: Entity) -> Entity: ...

    def upsert_relation(self, relation: Relation) -> Relation: ...

    def get_entity(self, entity_id: str) -> Entity | None: ...

    def find_entities_by_query(self, query: str, top_k: int = 5) -> list[Entity]: ...

    def get_all_entities(self) -> list[Entity]: ...

    def get_all_relations(self) -> list[Relation]: ...

    def get_k_hop_subgraph(
        self, seed_entity_ids: list[str], k_hops: int = 1
    ) -> tuple[list[Entity], list[Relation]]: ...

    def set_communities(self, communities: list[Community]) -> None: ...

    def get_all_communities(self) -> list[Community]: ...

    def to_networkx(self) -> nx.DiGraph: ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...


@runtime_checkable
class BaseVectorStore(Protocol):
    """Protocol for chunk vector storage and similarity search."""

    def add_chunks(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None: ...

    def search(self, query_vector: list[float], top_k: int = 4) -> list[tuple[TextChunk, float]]: ...

    def get_chunk(self, chunk_id: str) -> TextChunk | None: ...

    def get_all_chunks(self) -> list[TextChunk]: ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...

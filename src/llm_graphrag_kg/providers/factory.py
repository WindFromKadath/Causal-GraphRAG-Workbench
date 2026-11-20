"""Dependency injection factory creating concrete providers from AppConfig."""

from __future__ import annotations

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import (
    BaseEmbeddingClient,
    BaseGraphStore,
    BaseLLMClient,
    BaseVectorStore,
)
from llm_graphrag_kg.providers.embedding.local_hash import LocalHashEmbeddingClient
from llm_graphrag_kg.providers.embedding.openai_compat import OpenAICompatEmbeddingClient
from llm_graphrag_kg.providers.llm.mock import MockLLMClient
from llm_graphrag_kg.providers.llm.openai_compat import OpenAICompatLLMClient
from llm_graphrag_kg.providers.storage.local_vector import LocalVectorStore
from llm_graphrag_kg.providers.storage.networkx_store import NetworkXGraphStore


def create_llm_client(config: AppConfig) -> BaseLLMClient:
    provider = config.env.llm_provider.strip().lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai_compat":
        return OpenAICompatLLMClient(
            base_url=config.env.llm_base_url,
            api_key=config.env.llm_api_key,
            model=config.env.llm_model,
            timeout=config.env.llm_timeout,
        )
    raise ValueError(f"Unsupported llm_provider: {config.env.llm_provider!r}")


def create_embedding_client(config: AppConfig) -> BaseEmbeddingClient:
    provider = config.env.embedding_provider.strip().lower()
    if provider == "local_hash":
        return LocalHashEmbeddingClient()
    if provider == "openai_compat":
        return OpenAICompatEmbeddingClient(
            base_url=config.env.embedding_base_url,
            api_key=config.env.embedding_api_key,
            model=config.env.embedding_model,
            timeout=config.env.llm_timeout,
        )
    raise ValueError(f"Unsupported embedding_provider: {config.env.embedding_provider!r}")


def create_graph_store() -> BaseGraphStore:
    return NetworkXGraphStore()


def create_vector_store() -> BaseVectorStore:
    return LocalVectorStore()

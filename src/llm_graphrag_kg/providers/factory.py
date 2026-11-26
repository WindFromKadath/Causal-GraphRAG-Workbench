"""Dependency injection factory creating concrete providers from AppConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import (
    BaseEmbeddingClient,
    BaseGraphStore,
    BaseLLMClient,
    BaseVectorStore,
)
from llm_graphrag_kg.providers.embedding.local_hash import LocalHashEmbeddingClient
from llm_graphrag_kg.providers.embedding.ollama_native import OllamaNativeEmbeddingClient
from llm_graphrag_kg.providers.embedding.openai_compat import OpenAICompatEmbeddingClient
from llm_graphrag_kg.providers.llm.mock import MockLLMClient
from llm_graphrag_kg.providers.llm.ollama_native import OllamaNativeLLMClient
from llm_graphrag_kg.providers.llm.openai_compat import OpenAICompatLLMClient
from llm_graphrag_kg.providers.storage.local_vector import LocalVectorStore
from llm_graphrag_kg.providers.storage.networkx_store import NetworkXGraphStore

if TYPE_CHECKING:
    from llm_graphrag_kg.pipelines.memory_store import VerifiedMemoryStore
    from llm_graphrag_kg.providers.storage.pgvector_store import PgVerifiedMemoryStore


def create_llm_client(config: AppConfig) -> BaseLLMClient:
    provider = config.env.llm_provider.strip().lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "ollama_native":
        return OllamaNativeLLMClient(
            base_url=config.env.llm_base_url,
            model=config.env.llm_model,
            timeout=config.env.llm_timeout,
            think=config.env.llm_think,
            num_ctx=config.env.llm_num_ctx,
            num_gpu=config.env.llm_num_gpu,
            max_tokens=config.env.llm_max_tokens,
        )
    if provider == "openai_compat":
        # If targeting local Ollama with explicit native overrides, use `/api/chat`.
        if (
            config.env.llm_think is not None
            or config.env.llm_num_ctx is not None
            or config.env.llm_num_gpu is not None
        ) and (":11434" in config.env.llm_base_url):
            return OllamaNativeLLMClient(
                base_url=config.env.llm_base_url,
                model=config.env.llm_model,
                timeout=config.env.llm_timeout,
                think=config.env.llm_think,
                num_ctx=config.env.llm_num_ctx,
                num_gpu=config.env.llm_num_gpu,
                max_tokens=config.env.llm_max_tokens,
            )
        return OpenAICompatLLMClient(
            base_url=config.env.llm_base_url,
            api_key=config.env.llm_api_key,
            model=config.env.llm_model,
            timeout=config.env.llm_timeout,
        )
    raise ValueError(f"Unsupported llm_provider: {config.env.llm_provider!r}")


def create_fast_llm_client(config: AppConfig) -> BaseLLMClient | None:
    """Create optional fast-tier LLM client for intermediate CoT/ToT planning & pruning."""
    fast_model = (config.env.llm_fast_model or "").strip()
    if not fast_model or fast_model == config.env.llm_model:
        return None
    provider = config.env.llm_provider.strip().lower()
    if provider == "mock":
        return MockLLMClient()
    fast_ctx = config.env.llm_fast_num_ctx or config.env.llm_num_ctx or 8192
    if provider == "ollama_native" or (":11434" in config.env.llm_base_url):
        return OllamaNativeLLMClient(
            base_url=config.env.llm_base_url,
            model=fast_model,
            timeout=config.env.llm_timeout,
            think=False if config.env.llm_think is None else config.env.llm_think,
            num_ctx=fast_ctx,
            num_gpu=config.env.llm_num_gpu,
            max_tokens=config.env.llm_max_tokens,
        )
    if provider == "openai_compat":
        return OpenAICompatLLMClient(
            base_url=config.env.llm_base_url,
            api_key=config.env.llm_api_key,
            model=fast_model,
            timeout=config.env.llm_timeout,
        )
    return None


def create_embedding_client(config: AppConfig) -> BaseEmbeddingClient:
    provider = config.env.embedding_provider.strip().lower()
    if provider == "local_hash":
        return LocalHashEmbeddingClient()

    cache_dir = (
        config.resolve_path(config.env.embedding_cache_dir)
        if config.env.embedding_cache_dir
        else (config.storage_path / ".embed_cache")
    )
    if provider == "ollama_native":
        return OllamaNativeEmbeddingClient(
            base_url=config.env.embedding_base_url,
            model=config.env.embedding_model,
            timeout=config.env.embedding_timeout,
            batch_size=config.env.embedding_batch_size,
            num_gpu=config.env.embedding_num_gpu,
            num_ctx=config.env.embedding_num_ctx,
            dimensions=config.env.embedding_dimensions,
            cache_dir=cache_dir,
        )
    if provider == "openai_compat":
        # If targeting local Ollama with explicit num_gpu/num_ctx control, route via native `/api/embed`.
        if (
            config.env.embedding_num_gpu is not None or config.env.embedding_num_ctx is not None
        ) and (":11434" in config.env.embedding_base_url):
            return OllamaNativeEmbeddingClient(
                base_url=config.env.embedding_base_url,
                model=config.env.embedding_model,
                timeout=config.env.embedding_timeout,
                batch_size=config.env.embedding_batch_size,
                num_gpu=config.env.embedding_num_gpu,
                num_ctx=config.env.embedding_num_ctx,
                dimensions=config.env.embedding_dimensions,
                cache_dir=cache_dir,
            )
        return OpenAICompatEmbeddingClient(
            base_url=config.env.embedding_base_url,
            api_key=config.env.embedding_api_key,
            model=config.env.embedding_model,
            timeout=config.env.embedding_timeout,
            batch_size=config.env.embedding_batch_size,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unsupported embedding_provider: {config.env.embedding_provider!r}")


def create_graph_store(config: AppConfig | None = None) -> BaseGraphStore:
    if config is not None and config.env.graph_backend.strip().lower() == "neo4j":
        from llm_graphrag_kg.providers.storage.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(
            uri=config.env.neo4j_uri,
            user=config.env.neo4j_user,
            password=config.env.neo4j_password,
            database=config.env.neo4j_database,
            default_tenant_id=config.env.default_tenant_id,
            default_acl_roles=config.env.default_acl_roles,
        )
    return NetworkXGraphStore()


def create_vector_store(config: AppConfig | None = None) -> BaseVectorStore:
    if config is not None and config.env.vector_backend.strip().lower() == "pgvector":
        from llm_graphrag_kg.providers.storage.pgvector_store import PgVectorStore

        eff_dim = int(config.env.embedding_dimensions or config.env.pgvector_dim)
        return PgVectorStore(
            dsn=config.env.postgres_dsn,
            embedding_dim=eff_dim,
            embedding_model=config.env.embedding_model,
            ef_search=config.env.pgvector_ef_search,
            iterative_scan=config.env.pgvector_iterative_scan,
            default_tenant_id=config.env.default_tenant_id,
            default_acl_roles=config.env.default_acl_roles,
        )
    return LocalVectorStore()


def create_memory_store(
    config: AppConfig | None = None,
) -> VerifiedMemoryStore | PgVerifiedMemoryStore:
    if config is not None and config.env.vector_backend.strip().lower() == "pgvector":
        from llm_graphrag_kg.providers.storage.pgvector_store import PgVerifiedMemoryStore

        return PgVerifiedMemoryStore(
            dsn=config.env.postgres_dsn,
            default_tenant_id=config.env.default_tenant_id,
            default_acl_roles=config.env.default_acl_roles,
        )
    from llm_graphrag_kg.pipelines.memory_store import VerifiedMemoryStore

    return VerifiedMemoryStore()

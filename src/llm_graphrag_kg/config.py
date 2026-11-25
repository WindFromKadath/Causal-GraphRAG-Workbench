"""Configuration management combining non-sensitive YAML defaults with `.env` secrets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    input_dir: str = "data/sample"
    storage_dir: str = "storage"
    output_dir: str = "outputs"


class ChunkingConfig(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 60


class ExtractionConfig(BaseModel):
    temperature: float = 0.1
    max_entities_per_chunk: int = 15
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "Concept",
            "Technology",
            "Method",
            "Component",
            "Organization",
            "Person",
            "Document",
            "Metric",
        ]
    )


class CommunityConfig(BaseModel):
    resolution: float = 1.0
    seed: int = 42
    summarize_with_llm: bool = False


class RetrievalConfig(BaseModel):
    default_mode: str = "hybrid"
    top_k_chunks: int = 4
    top_k_entities: int = 5
    k_hops: int = 1
    top_k_communities: int = 3
    temperature: float = 0.2
    # Stage 3.1: Dense + BM25 RRF hybrid & HippoRAG 2 extensions (backward-compatible defaults)
    enable_hybrid_bm25: bool = True
    hybrid_bm25_weight: float = 0.35
    rrf_k: int = 60
    enable_query_to_triple: bool = True
    top_k_triples: int = 8
    enable_synonym_edges: bool = True
    synonym_cosine_threshold: float = 0.85
    max_synonyms_per_entity: int = 5
    max_indexed_triples: int = 3000


class VisualizationConfig(BaseModel):
    default_color_by: str = "type"
    physics_enabled: bool = True


class HarnessConfig(BaseModel):
    enable_compact: bool = True
    max_context_tokens: int = 4000
    max_entities: int = 24
    max_subgraph_relations: int = 30
    max_evidence_chunks: int = 6
    json_repair_retries: int = 1
    max_calls: int = 12
    max_wall_seconds: float = 180.0


class ReasoningConfig(BaseModel):
    default_paradigm: str = "direct"  # auto | direct | cot | tot | multi_branch
    tot_beam_width: int = 3
    tot_max_depth: int = 2
    tot_max_llm_calls: int = 6
    # Stage 3.2: Adaptive Query Router & Dynamic Early Stopping
    enable_early_stop: bool = True
    early_stop_coverage_threshold: float = 0.80
    multi_branch_min_score: float = 0.25


class EnvSettings(BaseSettings):
    """Sensitive and endpoint-specific settings loaded from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="GRAPHRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "openai_compat"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "local-no-key"
    llm_model: str = "qwen3.5:9b"
    llm_timeout: int = 180
    llm_think: bool | None = None
    llm_num_ctx: int | None = None
    llm_num_gpu: int | None = None
    llm_max_tokens: int | None = None
    # Stage 3.2+ dual-model tier: optional fast intermediate planner/evaluator (e.g. qwen3.5:4b)
    llm_fast_model: str | None = None
    llm_fast_num_ctx: int | None = 8192

    embedding_provider: str = "local_hash"
    embedding_base_url: str = "http://127.0.0.1:11434/v1"
    embedding_api_key: str = "local-no-key"
    embedding_model: str = "nomic-embed-text"
    embedding_timeout: int = 120
    embedding_batch_size: int = 32
    embedding_num_gpu: int | None = None
    embedding_cache_dir: str | None = None


class AppConfig(BaseModel):
    """Unified application configuration with safe redaction helpers."""

    project_root: Path = Field(default_factory=Path.cwd)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    env: EnvSettings = Field(default_factory=EnvSettings)

    def resolve_path(self, rel_or_abs: str | Path) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def input_path(self) -> Path:
        return self.resolve_path(self.paths.input_dir)

    @property
    def storage_path(self) -> Path:
        return self.resolve_path(self.paths.storage_dir)

    @property
    def output_path(self) -> Path:
        return self.resolve_path(self.paths.output_dir)

    def redacted_summary(self) -> dict[str, Any]:
        """Return a safe dictionary for diagnostics/logging with API keys masked."""
        return {
            "llm_provider": self.env.llm_provider,
            "llm_base_url": self.env.llm_base_url,
            "llm_model": self.env.llm_model,
            "llm_think": self.env.llm_think,
            "llm_num_ctx": self.env.llm_num_ctx,
            "llm_num_gpu": self.env.llm_num_gpu,
            "llm_fast_model": self.env.llm_fast_model,
            "llm_api_key": mask_secret(self.env.llm_api_key),
            "embedding_provider": self.env.embedding_provider,
            "embedding_base_url": self.env.embedding_base_url,
            "embedding_model": self.env.embedding_model,
            "embedding_num_gpu": self.env.embedding_num_gpu,
            "embedding_batch_size": self.env.embedding_batch_size,
            "embedding_api_key": mask_secret(self.env.embedding_api_key),
            "paths": self.paths.model_dump(),
        }


def mask_secret(value: str) -> str:
    """Mask secret tokens so they never appear in logs or terminal output."""
    if not value or value in {"your-api-key-here", "local-no-key"}:
        return "<unset/placeholder>"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def load_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
    llm_provider_override: str | None = None,
    embedding_provider_override: str | None = None,
) -> AppConfig:
    """Load YAML defaults and merge `.env` settings."""
    root = (project_root or Path.cwd()).resolve()
    yaml_path = config_path or (root / "configs" / "default.yaml")

    raw: dict[str, Any] = {}
    if yaml_path.is_file():
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded

    env_settings = EnvSettings(_env_file=root / ".env")
    if llm_provider_override:
        env_settings.llm_provider = llm_provider_override
    if embedding_provider_override:
        env_settings.embedding_provider = embedding_provider_override

    return AppConfig(
        project_root=root,
        paths=PathsConfig(**raw.get("paths", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        extraction=ExtractionConfig(**raw.get("extraction", {})),
        community=CommunityConfig(**raw.get("community", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        visualization=VisualizationConfig(**raw.get("visualization", {})),
        harness=HarnessConfig(**raw.get("harness", {})),
        reasoning=ReasoningConfig(**raw.get("reasoning", {})),
        env=env_settings,
    )

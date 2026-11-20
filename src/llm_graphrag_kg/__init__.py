"""LLM-GraphRAG-KG: Modular Python framework for LLM + GraphRAG + Knowledge Graph."""

from llm_graphrag_kg.config import AppConfig, load_config
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline

__all__ = ["AppConfig", "GraphRAGPipeline", "load_config"]

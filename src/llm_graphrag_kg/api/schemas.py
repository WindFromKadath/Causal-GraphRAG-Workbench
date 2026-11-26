"""Pydantic v2 OpenAPI request/response contracts for Stage 4.4 FastAPI endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuthContext(BaseModel):
    tenant_id: str = "default"
    acl_roles: list[str] = Field(default_factory=lambda: ["public"])
    operator: str = "anonymous"
    is_admin: bool = False


class ServiceHealthResponse(BaseModel):
    status: str = "ok"
    vector_backend: str
    graph_backend: str
    checkpointer: str
    llm_model: str
    embedding_model: str
    reranker_backend: str


class InfraServiceItem(BaseModel):
    service: str
    ok: bool
    endpoint: str
    detail: str


class InfraHealthResponse(BaseModel):
    all_healthy: bool
    services: list[InfraServiceItem] = Field(default_factory=list)


class DocumentIngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=128)
    title: str = ""
    source_uri: str = ""
    raw_text: str = Field(..., min_length=1)
    version_id: int = Field(default=1, ge=1)
    acl: list[str] | None = None
    via_redis: bool = False


class DocumentRollbackRequest(BaseModel):
    target_version: int = Field(..., ge=1)


class WorkerConsumeRequest(BaseModel):
    max_messages: int = Field(default=10, ge=1, le=200)
    block_ms: int = Field(default=300, ge=50, le=5000)


class ReconcileRequest(BaseModel):
    stale_lease_seconds: int = Field(default=300, ge=0)
    purge_superseded: bool = False


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = None
    mode: str = "ppr"
    reasoning: str = "auto"
    enable_writeback: bool = False
    require_hitl_approval: bool = False
    use_llm_rewrite: bool = False


class HitlResumeRequest(BaseModel):
    approved: bool
    reviewer: str = "api_reviewer"


class EvalMetricItemInput(BaseModel):
    question_id: str
    question_type: str = "General"
    prediction: str = ""
    answer_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    factuality_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieved_ids: list[str] = Field(default_factory=list)
    gold_ids: list[str] = Field(default_factory=list)
    execution_error: bool = False


class EvalSummaryRequest(BaseModel):
    mode: str = "ppr_auto"
    items: list[EvalMetricItemInput] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

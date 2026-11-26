"""FastAPI dependency injection, Tenant/RBAC Auth Guard, and AppServices runtime container."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status

from llm_graphrag_kg.agent.graph import GraphRAGAgentRunner
from llm_graphrag_kg.api.schemas import AuthContext
from llm_graphrag_kg.config import AppConfig, load_config
from llm_graphrag_kg.pipelines.incremental_worker import IncrementalIngestionWorker
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.storage.neo4j_store import Neo4jGraphStore
from llm_graphrag_kg.providers.storage.pgvector_store import PgVectorStore


class AppServices:
    """Lifecycle-managed enterprise services injected into FastAPI routes."""

    def __init__(
        self,
        config: AppConfig,
        *,
        pipeline: GraphRAGPipeline | None = None,
        runner: GraphRAGAgentRunner | None = None,
        worker: IncrementalIngestionWorker | None = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline or GraphRAGPipeline(config)
        self.runner = runner or GraphRAGAgentRunner(self.pipeline)
        self._worker = worker

    @property
    def worker(self) -> IncrementalIngestionWorker | None:
        if self._worker is not None:
            return self._worker
        if isinstance(self.pipeline.vector_store, PgVectorStore):
            neo = self.pipeline.graph_store if isinstance(self.pipeline.graph_store, Neo4jGraphStore) else None
            self._worker = IncrementalIngestionWorker(
                pg_store=self.pipeline.vector_store,
                embedder=self.pipeline.embedder,
                neo4j_store=neo,
                redis_url=self.config.env.redis_url,
            )
            return self._worker
        return None

    def close(self) -> None:
        self.runner.close()
        if hasattr(self.pipeline.graph_store, "close"):
            self.pipeline.graph_store.close()


def get_services(request: Request) -> AppServices:
    svc: Any = getattr(request.app.state, "services", None)
    if svc is None:
        cfg = load_config()
        svc = AppServices(cfg)
        request.app.state.services = svc
    return svc


def get_auth_context(
    services: AppServices = Depends(get_services),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_acl_roles: str | None = Header(default=None, alias="X-ACL-Roles"),
    x_operator: str | None = Header(default=None, alias="X-Operator"),
) -> AuthContext:
    """Enforce optional API key authentication and extract strict Tenant + RBAC ACL context."""
    expected_token = (services.config.env.api_auth_token or "").strip()
    if expected_token:
        supplied = (x_api_key or "").strip()
        if not supplied or not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid `X-API-Key` header.",
            )

    tenant = (x_tenant_id or services.config.env.default_tenant_id).strip() or "default"
    raw_roles = x_acl_roles if x_acl_roles is not None else services.config.env.default_acl_roles
    roles = [r.strip() for r in raw_roles.split(",") if r.strip()] or ["public"]
    operator = (x_operator or "api_user").strip() or "api_user"
    return AuthContext(
        tenant_id=tenant,
        acl_roles=roles,
        operator=operator,
        is_admin=("admin" in roles),
    )


def require_admin(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    """Guard privileged governance/reconciliation/HITL approval operations behind `admin` ACL role."""
    if not auth.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation requires `admin` role in `X-ACL-Roles`.",
        )
    return auth

import type {
  AgentRunOutput,
  DocumentItem,
  EvalMetricItemInput,
  EvalSummaryResponse,
  GraphOverviewResponse,
  InfraHealthResponse,
  RequestTraceItem,
  ServiceHealth,
  ShaclReportResponse,
  SseEventPacket,
  TenantSession,
  VerifiedMemoryItem,
} from "../types/api";

function buildHeaders(session: TenantSession, jsonContent = true): Record<string, string> {
  const headers: Record<string, string> = {
    "X-Tenant-Id": session.tenantId,
    "X-ACL-Roles": session.aclRoles,
    "X-Operator": session.operator || "console_user",
  };
  if (session.apiKey.trim()) {
    headers["X-API-Key"] = session.apiKey.trim();
  }
  if (jsonContent) {
    headers["Content-Type"] = "application/json";
  }
  return headers;
}

async function apiFetch<T>(
  path: string,
  session: TenantSession,
  init: RequestInit = {},
  jsonContent = true,
): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      ...buildHeaders(session, jsonContent),
      ...(init.headers as Record<string, string> | undefined),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    let msg = `${res.status} ${res.statusText}`;
    try {
      const parsed = JSON.parse(text);
      msg = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail ?? parsed);
    } catch {
      if (text) msg = text;
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

export const apiClient = {
  health(session: TenantSession) {
    return apiFetch<ServiceHealth>("/api/v1/health", session, {}, false);
  },

  infraHealth(session: TenantSession) {
    return apiFetch<InfraHealthResponse>("/api/v1/infra/health", session, {}, false);
  },

  listDocuments(session: TenantSession, includeDeleted = false) {
    return apiFetch<{ tenant_id: string; count: number; documents: DocumentItem[] }>(
      `/api/v1/documents?include_deleted=${includeDeleted}`,
      session,
      {},
      false,
    );
  },

  ingestText(
    session: TenantSession,
    payload: { doc_id: string; title: string; raw_text: string; version_id: number; acl?: string[]; via_redis?: boolean },
  ) {
    return apiFetch<Record<string, unknown>>("/api/v1/documents/ingest", session, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  uploadFile(session: TenantSession, file: File, docId?: string, versionId = 1, acl?: string) {
    const fd = new FormData();
    fd.append("file", file);
    if (docId) fd.append("doc_id", docId);
    fd.append("version_id", String(versionId));
    if (acl) fd.append("acl", acl);
    return apiFetch<Record<string, unknown>>(
      "/api/v1/documents/upload",
      session,
      { method: "POST", body: fd },
      false,
    );
  },

  rollbackDocument(session: TenantSession, docId: string, targetVersion: number) {
    return apiFetch<Record<string, unknown>>(`/api/v1/documents/${encodeURIComponent(docId)}/rollback`, session, {
      method: "POST",
      body: JSON.stringify({ target_version: targetVersion }),
    });
  },

  deleteDocument(session: TenantSession, docId: string) {
    return apiFetch<Record<string, unknown>>(`/api/v1/documents/${encodeURIComponent(docId)}`, session, {
      method: "DELETE",
    });
  },

  listDlq(session: TenantSession) {
    return apiFetch<{ count: number; entries: Array<Record<string, unknown>> }>("/api/v1/queue/dlq", session, {}, false);
  },

  consumeWorker(session: TenantSession, maxMessages = 10) {
    return apiFetch<Record<string, unknown>>("/api/v1/queue/worker/consume", session, {
      method: "POST",
      body: JSON.stringify({ max_messages: maxMessages, block_ms: 300 }),
    });
  },

  reconcile(session: TenantSession, staleLeaseSeconds = 300, purgeSuperseded = false) {
    return apiFetch<Record<string, unknown>>("/api/v1/queue/reconcile", session, {
      method: "POST",
      body: JSON.stringify({ stale_lease_seconds: staleLeaseSeconds, purge_superseded: purgeSuperseded }),
    });
  },

  graphOverview(session: TenantSession, limitNodes = 250) {
    return apiFetch<GraphOverviewResponse>(`/api/v1/graph/overview?limit_nodes=${limitNodes}`, session, {}, false);
  },

  graphEgo(session: TenantSession, entityId: string, kHops = 2) {
    return apiFetch<{
      seed_entity_id: string;
      entities: GraphOverviewResponse["entities"];
      relations: GraphOverviewResponse["relations"];
    }>(`/api/v1/graph/ego/${encodeURIComponent(entityId)}?k_hops=${kHops}`, session, {}, false);
  },

  shaclAudit(session: TenantSession, profile?: string) {
    const qs = profile ? `?profile=${encodeURIComponent(profile)}` : "";
    return apiFetch<ShaclReportResponse>(`/api/v1/graph/shacl${qs}`, session, {}, false);
  },

  resumeHitl(session: TenantSession, threadId: string, approved: boolean) {
    return apiFetch<AgentRunOutput>(`/api/v1/agent/hitl/${encodeURIComponent(threadId)}/resume`, session, {
      method: "POST",
      body: JSON.stringify({ approved, reviewer: session.operator || "console_admin" }),
    });
  },

  listMemories(session: TenantSession) {
    return apiFetch<{ tenant_id: string; count: number; memories: VerifiedMemoryItem[] }>(
      "/api/v1/memories",
      session,
      {},
      false,
    );
  },

  listTraces(session: TenantSession, limit = 25) {
    return apiFetch<{ tenant_id: string; count: number; traces: RequestTraceItem[] }>(
      `/api/v1/traces?limit=${limit}`,
      session,
      {},
      false,
    );
  },

  evalMetrics(session: TenantSession, mode: string, items: EvalMetricItemInput[]) {
    return apiFetch<EvalSummaryResponse>("/api/v1/eval/metrics", session, {
      method: "POST",
      body: JSON.stringify({ mode, items }),
    });
  },

  async streamAgentQuery(
    session: TenantSession,
    body: {
      question: string;
      mode: string;
      reasoning: string;
      enable_writeback: boolean;
      require_hitl_approval: boolean;
    },
    onEvent: (pkt: SseEventPacket) => void,
  ): Promise<void> {
    const res = await fetch("/api/v1/agent/stream", {
      method: "POST",
      headers: buildHeaders(session, true),
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      throw new Error(`SSE stream failed: ${res.status} ${res.statusText}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() ?? "";
      for (const block of blocks) {
        let evName = "node_update";
        let dataLine = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) {
            evName = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataLine += line.slice(5).trim();
          }
        }
        if (dataLine) {
          try {
            const parsed = JSON.parse(dataLine) as SseEventPacket;
            onEvent({ ...parsed, event: evName });
          } catch {
            // ignore malformed heartbeat chunk
          }
        }
      }
    }
  },
};

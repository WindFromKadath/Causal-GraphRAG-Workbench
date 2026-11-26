import { useEffect, useState } from "react";
import {
  Activity,
  BookOpen,
  FlaskConical,
  MessageSquareText,
  Moon,
  Network,
  Shield,
  Sun,
} from "lucide-react";
import { apiClient } from "./api/client";
import { ChatReasoningWorkspace } from "./components/ChatReasoningWorkspace";
import { DocumentsWorkspace } from "./components/DocumentsWorkspace";
import { EvalLabWorkspace } from "./components/EvalLabWorkspace";
import { GraphExplorerWorkspace } from "./components/GraphExplorerWorkspace";
import { ObservabilityWorkspace } from "./components/ObservabilityWorkspace";
import type { ServiceHealth, TenantSession } from "./types/api";

type TabId = "chat" | "docs" | "graph" | "eval" | "obs";

const TABS: Array<{ id: TabId; label: string; icon: typeof MessageSquareText }> = [
  { id: "chat", label: "智能问答 & SSE", icon: MessageSquareText },
  { id: "docs", label: "知识库与增量治理", icon: BookOpen },
  { id: "graph", label: "Neo4j 图谱 & 超边", icon: Network },
  { id: "eval", label: "评测回归实验室", icon: FlaskConical },
  { id: "obs", label: "Trace & A-MEM 审计", icon: Activity },
];

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>("chat");
  const [darkMode, setDarkMode] = useState(true);
  const [session, setSession] = useState<TenantSession>({
    tenantId: "fault_diag",
    aclRoles: "public,engineer,admin",
    operator: "console_admin",
    apiKey: "",
  });
  const [health, setHealth] = useState<ServiceHealth | null>(null);
  const [tenants, setTenants] = useState<string[]>([
    "fault_diag",
    "fmea_oreda",
    "failuresensoriq",
    "grb_medical",
    "grb_novel",
  ]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  useEffect(() => {
    apiClient
      .health(session)
      .then(setHealth)
      .catch(() => setHealth(null));
    apiClient
      .listTenants(session)
      .then((res) => {
        if (res.tenants.length > 0) setTenants(res.tenants);
      })
      .catch(() => undefined);
  }, [session]);

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Sidebar */}
      <aside className="lg:w-64 shrink-0 border-b lg:border-b-0 lg:border-r border-slate-800/80 p-4 flex flex-col justify-between gap-4 bg-slate-950/30">
        <div className="space-y-5">
          <div>
            <div className="text-sm font-bold tracking-tight text-sky-400">Causal-GraphRAG-Workbench</div>
            <div className="text-[11px] text-slate-400 mt-0.5">Enterprise &amp; Causal Console</div>
          </div>

          <nav className="space-y-1">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium cursor-pointer transition-colors ${
                    active
                      ? "bg-sky-600 text-white"
                      : "hover:bg-slate-800/60 text-slate-300"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Backend Live Status */}
        <div className="panel-subtle p-3 text-[11px] space-y-1">
          <div className="flex items-center justify-between font-medium">
            <span>FastAPI 网关</span>
            <span className={health ? "text-emerald-400" : "text-rose-400"}>
              {health ? "● ONLINE" : "● OFFLINE"}
            </span>
          </div>
          {health && (
            <>
              <div className="text-slate-400">
                Store: <code>{health.vector_backend}</code> + <code>{health.graph_backend}</code>
              </div>
              <div className="text-slate-400 truncate">
                LLM: <code>{health.llm_model}</code>
              </div>
              <div className="text-slate-400 truncate">
                Embed: <code>{health.embedding_model}</code>
              </div>
            </>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Tenant & RBAC Context Bar */}
        <header className="border-b border-slate-800/80 px-5 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-slate-400 flex items-center gap-1">
              <Shield className="w-3.5 h-3.5 text-sky-400" /> 租户 / RBAC:
            </span>
            <select
              value={session.tenantId}
              onChange={(e) => setSession((s) => ({ ...s, tenantId: e.target.value }))}
              className="panel-subtle px-2.5 py-1 outline-none"
            >
              {tenants.map((t) => (
                <option key={t} value={t}>
                  Tenant: {t}
                </option>
              ))}
            </select>
            <input
              value={session.aclRoles}
              onChange={(e) => setSession((s) => ({ ...s, aclRoles: e.target.value }))}
              className="panel-subtle px-2.5 py-1 outline-none w-48 font-mono"
              placeholder="ACL roles (comma-separated)"
              title="X-ACL-Roles Header"
            />
            <input
              type="password"
              value={session.apiKey}
              onChange={(e) => setSession((s) => ({ ...s, apiKey: e.target.value }))}
              className="panel-subtle px-2.5 py-1 outline-none w-36"
              placeholder="X-API-Key (可选)"
            />
          </div>

          <button
            onClick={() => setDarkMode((d) => !d)}
            className="panel-subtle px-3 py-1.5 text-xs flex items-center gap-1.5 cursor-pointer hover:border-sky-500"
          >
            {darkMode ? <Sun className="w-3.5 h-3.5 text-amber-300" /> : <Moon className="w-3.5 h-3.5 text-sky-500" />}
            {darkMode ? "切换浅色" : "切换深色"}
          </button>
        </header>

        <main className="flex-1 p-5 overflow-y-auto">
          {activeTab === "chat" && <ChatReasoningWorkspace session={session} />}
          {activeTab === "docs" && <DocumentsWorkspace session={session} />}
          {activeTab === "graph" && <GraphExplorerWorkspace session={session} />}
          {activeTab === "eval" && <EvalLabWorkspace session={session} />}
          {activeTab === "obs" && <ObservabilityWorkspace session={session} />}
        </main>
      </div>
    </div>
  );
}

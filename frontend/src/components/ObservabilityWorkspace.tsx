import { useCallback, useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import { Activity, Brain, Database, RefreshCw } from "lucide-react";
import { apiClient } from "../api/client";
import type { InfraHealthResponse, RequestTraceItem, TenantSession, VerifiedMemoryItem } from "../types/api";

interface Props {
  session: TenantSession;
}

export function ObservabilityWorkspace({ session }: Props) {
  const wfChartRef = useRef<HTMLDivElement | null>(null);
  const [infra, setInfra] = useState<InfraHealthResponse | null>(null);
  const [traces, setTraces] = useState<RequestTraceItem[]>([]);
  const [memories, setMemories] = useState<VerifiedMemoryItem[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<RequestTraceItem | null>(null);
  const [loading, setLoading] = useState(false);

  const loadAll = useCallback(async () => {
    setLoading(true);
    try {
      const [inf, tr, mem] = await Promise.all([
        apiClient.infraHealth(session),
        apiClient.listTraces(session, 20),
        apiClient.listMemories(session),
      ]);
      setInfra(inf);
      setTraces(tr.traces);
      setMemories(mem.memories);
      if (tr.traces.length > 0) {
        setSelectedTrace((prev) => prev ?? tr.traces[0]);
      }
    } finally {
      setLoading(false);
    }
  }, [session]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  useEffect(() => {
    if (!selectedTrace || !wfChartRef.current) return;
    const chart = echarts.init(wfChartRef.current);
    const steps = selectedTrace.waterfall;
    chart.setOption({
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      grid: { left: 130, right: 32, top: 12, bottom: 24 },
      xAxis: {
        type: "value",
        axisLabel: { color: "#94a3b8", formatter: "{value} ms" },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      yAxis: {
        type: "category",
        inverse: true,
        data: steps.map((s) => s.node),
        axisLabel: { color: "#cbd5e1", fontSize: 11 },
      },
      series: [
        {
          type: "bar",
          barWidth: 16,
          data: steps.map((s) => Number(s.duration_ms.toFixed(1))),
          itemStyle: { color: "#34d399", borderRadius: [0, 5, 5, 0] },
          label: { show: true, position: "right", color: "#e2e8f0", formatter: "{c} ms" },
        },
      ],
    });
    return () => chart.dispose();
  }, [selectedTrace]);

  return (
    <div className="space-y-4">
      <div className="panel p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-emerald-400" />
          <h2 className="text-base font-semibold">Trace 全链路可观测 & A-MEM 记忆审计台</h2>
        </div>
        <button
          onClick={() => void loadAll()}
          className="px-3 py-1.5 rounded-lg panel-subtle hover:border-sky-500 text-xs flex items-center gap-1.5 cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} /> 刷新状态
        </button>
      </div>

      {/* Infrastructure Cards */}
      {infra && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {infra.services.map((svc) => (
            <div key={svc.service} className="panel p-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-sky-400" />
                  {svc.service}
                </span>
                <span
                  className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                    svc.ok ? "bg-emerald-500/20 text-emerald-300" : "bg-rose-500/20 text-rose-300"
                  }`}
                >
                  {svc.ok ? "HEALTHY" : "DOWN"}
                </span>
              </div>
              <div className="text-[11px] font-mono text-slate-400 truncate">{svc.endpoint}</div>
              <div className="text-[11px] text-slate-300">{svc.detail}</div>
            </div>
          ))}
        </div>
      )}

      {/* Traces List + ECharts Waterfall */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="panel p-4 space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            PostgreSQL Request Traces（最近 {traces.length} 条）
          </h3>
          <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
            {traces.map((t) => (
              <button
                key={t.request_id}
                onClick={() => setSelectedTrace(t)}
                className={`w-full text-left panel-subtle p-2.5 text-xs cursor-pointer transition-colors ${
                  selectedTrace?.request_id === t.request_id ? "border-sky-500" : ""
                }`}
              >
                <div className="flex items-center justify-between">
                  <code className="text-sky-400">{t.request_id}</code>
                  <span className="text-slate-400">
                    {t.total_ms.toFixed(0)} ms · Grounding{" "}
                    {t.grounding_score !== null ? `${(t.grounding_score * 100).toFixed(0)}%` : "-"}
                  </span>
                </div>
                <div className="mt-1 truncate text-slate-200">{t.question}</div>
              </button>
            ))}
            {traces.length === 0 && <p className="text-xs text-slate-400">暂无请求 Trace 记录。</p>}
          </div>
        </div>

        <div className="panel p-4 space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            节点瀑布耗时 Waterfall（{selectedTrace?.request_id ?? "未选择"}）
          </h3>
          {selectedTrace ? (
            <>
              <div ref={wfChartRef} className="w-full h-56" />
              <div className="text-[11px] text-slate-400 font-mono truncate">
                Models: {JSON.stringify(selectedTrace.model_versions)}
              </div>
            </>
          ) : (
            <p className="text-xs text-slate-400">点击左侧任意 Trace 查看节点瀑布耗时与模型快照。</p>
          )}
        </div>
      </div>

      {/* Verified A-MEM Ledger */}
      <div className="panel p-4 space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Brain className="w-4 h-4 text-purple-400" />
          已验证情景记忆账本（A-MEM · {memories.length} 条）
        </h3>
        {memories.length === 0 ? (
          <p className="text-xs text-slate-400">当前租户下暂无已验证 A-MEM 记录。</p>
        ) : (
          <div className="space-y-2">
            {memories.map((m) => (
              <div key={m.memory_id} className="panel-subtle p-3 text-xs space-y-1">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <code className="text-purple-300">{m.memory_id}</code>
                  <span className="text-emerald-300">
                    Grounding {(m.grounding_score * 100).toFixed(0)}% · Conf {m.confidence.toFixed(2)} · {m.valid_from}
                  </span>
                </div>
                <div className="font-medium">Q: {m.question}</div>
                <div className="text-slate-300">A: {m.conclusion}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

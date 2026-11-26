import { useCallback, useEffect, useRef, useState } from "react";
import cytoscape, { type Core } from "cytoscape";
import { Network, RefreshCw, ShieldCheck } from "lucide-react";
import { apiClient } from "../api/client";
import type { EntityItem, GraphOverviewResponse, ShaclReportResponse, TenantSession } from "../types/api";

interface Props {
  session: TenantSession;
}

const TYPE_COLORS: Record<string, string> = {
  Equipment: "#38bdf8",
  Asset: "#38bdf8",
  Component: "#818cf8",
  Fault: "#f87171",
  FailureMode: "#f87171",
  Symptom: "#fbbf24",
  SignalChannel: "#34d399",
  Measure: "#2dd4bf",
  Action: "#2dd4bf",
  HyperEdge: "#c084fc",
};

export function GraphExplorerWorkspace({ session }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  const [graph, setGraph] = useState<GraphOverviewResponse | null>(null);
  const [selected, setSelected] = useState<EntityItem | null>(null);
  const [layoutName, setLayoutName] = useState<"cose" | "circle" | "breadthfirst">("cose");
  const [shaclProfile, setShaclProfile] = useState("fault_diagnosis");
  const [shacl, setShacl] = useState<ShaclReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.graphOverview(session, 160);
      setGraph(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [session]);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph]);

  useEffect(() => {
    if (!containerRef.current || !graph) return;
    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }

    const elements: cytoscape.ElementDefinition[] = [];
    const entMap = new Map(graph.entities.map((e) => [e.entity_id, e]));

    for (const ent of graph.entities) {
      elements.push({
        data: {
          id: ent.entity_id,
          label: ent.name,
          kind: "entity",
          color: TYPE_COLORS[ent.entity_type] ?? "#60a5fa",
        },
      });
    }
    for (const rel of graph.relations) {
      if (entMap.has(rel.source_id) && entMap.has(rel.target_id)) {
        elements.push({
          data: {
            id: rel.relation_id,
            source: rel.source_id,
            target: rel.target_id,
            label: rel.relation_type,
            kind: "relation",
          },
        });
      }
    }
    for (const he of graph.hyperedges) {
      const heNodeId = `he::${he.hyperedge_id}`;
      elements.push({
        data: {
          id: heNodeId,
          label: `⚡ ${he.relation_type}`,
          kind: "hyperedge",
          color: TYPE_COLORS.HyperEdge,
        },
      });
      for (const cid of he.condition_entity_ids) {
        if (entMap.has(cid)) {
          elements.push({
            data: { id: `${heNodeId}_c_${cid}`, source: cid, target: heNodeId, label: "CONDITION", kind: "he_edge" },
          });
        }
      }
      if (entMap.has(he.target_entity_id)) {
        elements.push({
          data: {
            id: `${heNodeId}_t_${he.target_entity_id}`,
            source: heNodeId,
            target: he.target_entity_id,
            label: "TARGET",
            kind: "he_edge",
          },
        });
      }
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: { name: layoutName, animate: false },
      style: [
        {
          selector: 'node[kind="entity"]',
          style: {
            "background-color": "data(color)",
            label: "data(label)",
            color: "#e2e8f0",
            "font-size": 10,
            "text-valign": "bottom",
            "text-margin-y": 4,
            width: 24,
            height: 24,
          },
        },
        {
          selector: 'node[kind="hyperedge"]',
          style: {
            shape: "diamond",
            "background-color": "#c084fc",
            label: "data(label)",
            color: "#f3e8ff",
            "font-size": 9,
            width: 28,
            height: 28,
          },
        },
        {
          selector: 'edge[kind="relation"]',
          style: {
            width: 1.5,
            "line-color": "#475569",
            "target-arrow-color": "#64748b",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
        {
          selector: 'edge[kind="he_edge"]',
          style: {
            width: 2,
            "line-style": "dashed",
            "line-color": "#c084fc",
            "target-arrow-color": "#c084fc",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
      ],
    });

    cy.on("tap", 'node[kind="entity"]', (evt) => {
      const id = String(evt.target.id());
      setSelected(entMap.get(id) ?? null);
    });
    cyRef.current = cy;

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph, layoutName]);

  async function runShacl() {
    setError(null);
    try {
      const rep = await apiClient.shaclAudit(session, shaclProfile);
      setShacl(rep);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="space-y-4">
      <div className="panel p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Network className="w-5 h-5 text-sky-400" />
          <h2 className="text-base font-semibold">Neo4j 图谱 & N 元超边浏览器（Cytoscape.js）</h2>
          {graph && (
            <span className="text-xs text-slate-400 ml-2">
              实体 {graph.stats.entity_count} · 关系 {graph.stats.relation_count} · 超边 {graph.stats.hyperedge_count} ·
              社区 {graph.stats.community_count}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <select
            value={layoutName}
            onChange={(e) => setLayoutName(e.target.value as "cose" | "circle" | "breadthfirst")}
            className="panel-subtle px-2.5 py-1.5 outline-none"
          >
            <option value="cose">布局: 力导向 (COSE)</option>
            <option value="circle">布局: 环形 (Circle)</option>
            <option value="breadthfirst">布局: 层级 (Breadthfirst)</option>
          </select>
          <select
            value={shaclProfile}
            onChange={(e) => setShaclProfile(e.target.value)}
            className="panel-subtle px-2.5 py-1.5 outline-none"
          >
            <option value="fault_diagnosis">SHACL Profile: fault_diagnosis</option>
            <option value="fmea_kg_rdf">SHACL Profile: fmea_kg_rdf (OREDA)</option>
            <option value="failuresensoriq">SHACL Profile: failuresensoriq</option>
            <option value="default">SHACL Profile: default</option>
          </select>
          <button
            onClick={() => void runShacl()}
            className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5 cursor-pointer"
          >
            <ShieldCheck className="w-3.5 h-3.5" /> SHACL-lite 体检
          </button>
          <button
            onClick={() => void loadGraph()}
            className="px-3 py-1.5 rounded-lg panel-subtle hover:border-sky-500 flex items-center gap-1.5 cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} /> 刷新
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-500/50 bg-rose-500/10 px-4 py-2 text-xs text-rose-300">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 panel p-2">
          <div ref={containerRef} className="w-full h-[460px] rounded-lg bg-slate-950/70" />
        </div>

        <div className="space-y-4">
          <div className="panel p-4 space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">节点属性与溯源检查器</h3>
            {selected ? (
              <div className="text-xs space-y-1.5">
                <div className="text-sm font-semibold text-sky-400">{selected.name}</div>
                <div>
                  类型：<span className="px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300">{selected.entity_type}</span> ·
                  ID：<code>{selected.entity_id}</code>
                </div>
                <p className="text-slate-300">{selected.description || "无描述"}</p>
                <div className="text-slate-400">
                  溯源切片：<code>{selected.source_chunk_ids.join(", ") || "-"}</code>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-400">点击左侧图谱中的任意实体节点查看详情与原文溯源。</p>
            )}
          </div>

          {shacl && (
            <div className="panel p-4 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold">SHACL-lite 合规报告（{shacl.profile}）</span>
                <span
                  className={`px-2 py-0.5 rounded-full font-semibold ${
                    shacl.overall_status === "PASS"
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "bg-amber-500/20 text-amber-300"
                  }`}
                >
                  {shacl.overall_status} · {(shacl.compliance_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="space-y-1.5 text-xs">
                {shacl.checks.map((c) => (
                  <div key={c.check_id} className="panel-subtle px-2.5 py-1.5 flex items-center justify-between">
                    <span>{c.title}</span>
                    <span className={c.passed ? "text-emerald-400" : "text-amber-300"}>
                      {c.passed ? "PASS" : `${c.violation_count} 项`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

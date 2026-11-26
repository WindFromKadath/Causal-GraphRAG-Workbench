import { useState } from "react";
import {
  Activity,
  CheckCircle2,
  GitBranch,
  Play,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import { apiClient } from "../api/client";
import type { AgentRunOutput, SseEventPacket, TenantSession } from "../types/api";

interface Props {
  session: TenantSession;
}

const PIPELINE_NODES = [
  { id: "acl_guard", label: "ACL 鉴权" },
  { id: "query_rewrite", label: "查询改写" },
  { id: "recall_and_rerank", label: "混合召回+精排" },
  { id: "expand_query", label: "Corrective 重试" },
  { id: "graph_reason", label: "图谱推理 CoT/ToT" },
  { id: "self_heal_grounding", label: "Grounding 自愈" },
  { id: "writeback_gate", label: "A-MEM 门禁" },
  { id: "hitl_memory_approval", label: "HITL 审批" },
  { id: "commit_memory", label: "记忆写回" },
  { id: "record_trace", label: "Trace 落库" },
];

const SAMPLE_QUESTIONS = [
  "离心泵机组同时出现2倍频振动升高和轴承温升异常，根因与排查措施是什么？",
  "弹性联轴器老化与润滑脂劣化在振动与温度特征上有何差异？",
  "SHH 高振动报警对应的故障机理与处置措施有哪些？",
];

export function ChatReasoningWorkspace({ session }: Props) {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [mode, setMode] = useState("ppr");
  const [reasoning, setReasoning] = useState("auto");
  const [enableWriteback, setEnableWriteback] = useState(false);
  const [requireHitl, setRequireHitl] = useState(false);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveNodes, setLiveNodes] = useState<Record<string, { duration_ms?: number; summary?: unknown }>>({});
  const [hitlThreadId, setHitlThreadId] = useState<string | null>(null);
  const [hitlPayload, setHitlPayload] = useState<Record<string, unknown> | null>(null);
  const [result, setResult] = useState<AgentRunOutput | null>(null);

  async function handleRun() {
    if (!question.trim() || running) return;
    setRunning(true);
    setError(null);
    setLiveNodes({});
    setHitlThreadId(null);
    setHitlPayload(null);
    setResult(null);

    try {
      await apiClient.streamAgentQuery(
        session,
        {
          question: question.trim(),
          mode,
          reasoning,
          enable_writeback: enableWriteback,
          require_hitl_approval: enableWriteback && requireHitl,
        },
        (pkt: SseEventPacket) => {
          if (pkt.event === "node_update") {
            setLiveNodes((prev) => ({
              ...prev,
              [pkt.node]: {
                duration_ms: Number((pkt.data as { duration_ms?: number }).duration_ms ?? 0),
                summary: (pkt.data as { summary?: unknown }).summary,
              },
            }));
          } else if (pkt.event === "hitl_interrupt") {
            setHitlThreadId(pkt.thread_id ?? null);
            setHitlPayload(pkt.data);
          } else if (pkt.event === "final") {
            setResult(pkt.data as unknown as AgentRunOutput);
          } else if (pkt.event === "error") {
            setError(String(pkt.data.error ?? "SSE 流执行异常"));
          }
        },
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  async function handleHitlDecision(approved: boolean) {
    if (!hitlThreadId) return;
    setRunning(true);
    setError(null);
    try {
      const resumed = await apiClient.resumeHitl(session, hitlThreadId, approved);
      setResult(resumed);
      setHitlThreadId(null);
      setHitlPayload(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  const branches = result?.query_result?.metadata?.reasoning?.branches ?? [];
  const cotSteps = result?.query_result?.metadata?.reasoning?.steps ?? [];
  const hyperedges = result?.query_result?.metadata?.matched_hyperedges ?? [];
  const recalledMems = result?.query_result?.metadata?.recalled_memories ?? [];

  return (
    <div className="space-y-4">
      {/* Query Bar */}
      <div className="panel p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-semibold">智能问答 & LangGraph SSE 实时推理台</h2>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="panel-subtle px-2.5 py-1.5 text-xs outline-none"
            >
              <option value="ppr">模式: HippoRAG 2 PPR</option>
              <option value="hybrid">模式: Hybrid RRF</option>
              <option value="local">模式: Local KG</option>
              <option value="global">模式: Global 社区</option>
            </select>
            <select
              value={reasoning}
              onChange={(e) => setReasoning(e.target.value)}
              className="panel-subtle px-2.5 py-1.5 text-xs outline-none"
            >
              <option value="auto">推理: Auto 自适应路由</option>
              <option value="tot">推理: Graph-ToT</option>
              <option value="cot">推理: Graph-CoT</option>
              <option value="multi_branch">推理: Multi-Branch</option>
            </select>
            <label className="panel-subtle px-2.5 py-1.5 flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={enableWriteback}
                onChange={(e) => setEnableWriteback(e.target.checked)}
              />
              <span>A-MEM 写回</span>
            </label>
            <label className="panel-subtle px-2.5 py-1.5 flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                disabled={!enableWriteback}
                checked={enableWriteback && requireHitl}
                onChange={(e) => setRequireHitl(e.target.checked)}
              />
              <span>HITL 审批暂停</span>
            </label>
          </div>
        </div>

        <div className="flex gap-2">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={2}
            placeholder="输入工业故障诊断、根因分析或知识库问题..."
            className="panel-subtle flex-1 px-3 py-2 text-sm resize-none outline-none focus:border-sky-500"
          />
          <button
            onClick={handleRun}
            disabled={running}
            className="px-5 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white text-sm font-medium flex items-center gap-2 cursor-pointer"
          >
            <Play className="w-4 h-4" />
            {running ? "推演中..." : "SSE 推理"}
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          {SAMPLE_QUESTIONS.map((sq) => (
            <button
              key={sq}
              onClick={() => setQuestion(sq)}
              className="text-xs px-2.5 py-1 rounded-full panel-subtle hover:border-sky-500 cursor-pointer truncate max-w-md"
            >
              {sq}
            </button>
          ))}
        </div>
      </div>

      {/* Live SSE Pipeline Stepper */}
      <div className="panel p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            LangGraph 实时节点执行流（SSE）
          </span>
          {result && (
            <span className="text-xs text-slate-400">
              总耗时 <strong className="text-sky-400">{result.total_ms.toFixed(0)} ms</strong> · Reranker:{" "}
              <code>{result.reranker_backend}</code>
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-5 lg:grid-cols-10 gap-2">
          {PIPELINE_NODES.map((node) => {
            const hit = liveNodes[node.id];
            const isPaused = node.id === "hitl_memory_approval" && hitlThreadId !== null;
            return (
              <div
                key={node.id}
                className={`rounded-lg border px-2 py-2 text-center transition-all ${
                  isPaused
                    ? "border-amber-500 bg-amber-500/15"
                    : hit
                      ? "border-emerald-500/60 bg-emerald-500/10"
                      : "border-slate-700/50 opacity-50"
                }`}
              >
                <div className="text-[11px] font-medium truncate">{node.label}</div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {isPaused ? "等待审批" : hit ? `${(hit.duration_ms ?? 0).toFixed(1)} ms` : "—"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-500/60 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      {/* HITL Interrupt Approval Banner */}
      {hitlThreadId && hitlPayload && (
        <div className="rounded-xl border-2 border-amber-500/70 bg-amber-500/10 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-amber-300 font-semibold text-sm">
              <ShieldAlert className="w-5 h-5" />
              LangGraph HITL 人工审批已暂停（Thread: <code>{hitlThreadId}</code>）
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => handleHitlDecision(true)}
                className="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium flex items-center gap-1.5 cursor-pointer"
              >
                <ShieldCheck className="w-4 h-4" /> 批准写入 A-MEM
              </button>
              <button
                onClick={() => handleHitlDecision(false)}
                className="px-3.5 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-medium flex items-center gap-1.5 cursor-pointer"
              >
                <XCircle className="w-4 h-4" /> 驳回写入
              </button>
            </div>
          </div>
          <div className="text-xs text-slate-300 space-y-1">
            <div>
              <strong>待写入结论预览：</strong>
              {String(hitlPayload.answer_preview ?? "")}
            </div>
            <div>
              <strong>Grounding：</strong>
              {Number(hitlPayload.grounding_score ?? 0) * 100}% · <strong>置信度：</strong>
              {Number(hitlPayload.confidence ?? 0).toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Answer + Trace Panels */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 space-y-4">
            <div className="panel p-4 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-sm font-semibold">最终诊断与溯源结论</span>
                <div className="flex items-center gap-2 text-xs">
                  <span
                    className={`px-2 py-0.5 rounded-full font-medium ${
                      result.grounding_score === 1
                        ? "bg-emerald-500/20 text-emerald-300"
                        : "bg-amber-500/20 text-amber-300"
                    }`}
                  >
                    Grounding:{" "}
                    {result.grounding_score !== null ? `${(result.grounding_score * 100).toFixed(0)}%` : "N/A"}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300">
                    Paradigm: {result.query_result?.metadata?.reasoning_paradigm ?? "auto"}
                  </span>
                </div>
              </div>
              <div className="text-sm leading-relaxed whitespace-pre-wrap">{result.answer}</div>

              {(hyperedges.length > 0 || recalledMems.length > 0) && (
                <div className="pt-2 border-t border-slate-700/60 flex flex-wrap gap-2 text-xs">
                  {hyperedges.slice(0, 3).map((h, i) => (
                    <span key={i} className="px-2 py-1 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30">
                      🧬 超边: {(h.matched_condition_names ?? []).join(" + ")} ⇒ {h.target_entity_name}
                    </span>
                  ))}
                  {recalledMems.map((m, i) => (
                    <span key={i} className="px-2 py-1 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                      🧠 A-MEM 召回: {m.memory_id}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* ToT Branches */}
            {branches.length > 0 && (
              <div className="panel p-4 space-y-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <GitBranch className="w-3.5 h-3.5 text-sky-400" /> 显性 Graph-ToT 假设分支对比
                </h3>
                <div className="space-y-2">
                  {branches.map((br) => (
                    <div
                      key={br.branch_id}
                      className={`p-3 rounded-lg border text-xs ${
                        br.status === "selected"
                          ? "border-emerald-500/50 bg-emerald-500/10"
                          : "border-slate-700/60 opacity-75"
                      }`}
                    >
                      <div className="flex items-center justify-between font-medium">
                        <span>
                          [{br.branch_id}] {br.status === "selected" ? "✅ SELECTED" : "✂️ PRUNED"} ·{" "}
                          {br.target_entity_name || br.target_entity_id}
                        </span>
                        <span className="text-amber-300">Score: {br.score.toFixed(2)}</span>
                      </div>
                      <p className="mt-1 text-slate-300">{br.hypothesis}</p>
                      {br.prune_reason && <p className="mt-1 text-slate-400">剪枝理由：{br.prune_reason}</p>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* CoT Steps */}
            {cotSteps.length > 0 && (
              <div className="panel p-4 space-y-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  显性 Graph-CoT 推理链与引用审计
                </h3>
                <div className="space-y-2">
                  {cotSteps.map((st) => (
                    <div key={st.step_id} className="panel-subtle p-2.5 text-xs flex items-start gap-2">
                      {st.grounded ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      ) : (
                        <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                      )}
                      <div>
                        <div className="font-medium">
                          #{st.step_id} [{st.stage}] {st.claim}
                        </div>
                        <div className="text-[11px] text-slate-400 mt-0.5">
                          Cites:{" "}
                          {[...(st.cited_entity_ids ?? []), ...(st.cited_relation_ids ?? []), ...(st.cited_chunk_ids ?? [])].join(
                            ", ",
                          ) || "-"}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Rewritten Queries & Reranked Evidences */}
          <div className="space-y-4">
            <div className="panel p-4 space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Multi-Query 改写 ({result.rewritten_queries.length})
              </h3>
              <ul className="space-y-1 text-xs">
                {result.rewritten_queries.map((q, idx) => (
                  <li key={idx} className="panel-subtle px-2.5 py-1.5">
                    <span className="text-sky-400 mr-1.5">{idx === 0 ? "[原问 1.25×]" : `[改写 #${idx}]`}</span>
                    {q}
                  </li>
                ))}
              </ul>
            </div>

            <div className="panel p-4 space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                精排 Top-{result.reranked_hits.length} 证据切片
              </h3>
              <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                {result.reranked_hits.map((ev) => (
                  <div key={ev.chunk_id} className="panel-subtle p-2.5 text-xs space-y-1">
                    <div className="flex items-center justify-between text-[11px] text-sky-400">
                      <code className="truncate">{ev.chunk_id}</code>
                      <span>
                        #{ev.coarse_rank}→#{ev.final_rank} ({ev.rerank_score.toFixed(3)})
                      </span>
                    </div>
                    <p className="text-slate-300 leading-snug">{ev.snippet}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

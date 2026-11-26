import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import { FlaskConical, Play } from "lucide-react";
import { apiClient } from "../api/client";
import type { EvalMetricItemInput, EvalSummaryResponse, TenantSession } from "../types/api";

interface Props {
  session: TenantSession;
}

const DEFAULT_EVAL_ITEMS: EvalMetricItemInput[] = [
  {
    question_id: "q_ind_01",
    question_type: "Complex Reasoning",
    prediction: "离心泵2倍频振动升高与轴承温升异常的根因为轴系不对中，应执行激光对中复核。",
    answer_accuracy: 0.92,
    factuality_f1: 0.91,
    semantic_similarity: 0.95,
    faithfulness: 1.0,
    semantic_coverage: 0.9,
    retrieved_ids: ["doc_c4aa51e5::chunk_0000", "doc_9109d6da::chunk_0001"],
    gold_ids: ["doc_c4aa51e5::chunk_0000"],
    execution_error: false,
  },
  {
    question_id: "q_ind_02",
    question_type: "Contextual Summarize",
    prediction: "润滑脂劣化会导致高频噪声与轴承温升加剧，同时建议检查冷却水套与油位。",
    answer_accuracy: 0.81,
    factuality_f1: 0.78,
    semantic_similarity: 0.89,
    faithfulness: 0.88,
    semantic_coverage: 0.76,
    retrieved_ids: ["doc_b::chunk_0002", "doc_a::chunk_0001"],
    gold_ids: ["doc_a::chunk_0001"],
    execution_error: false,
  },
  {
    question_id: "q_ind_03",
    question_type: "Multi-Hop",
    prediction: "SHH 高振动报警可能关联联轴器磨损或机械密封泄漏，部分参数需进一步核实。",
    answer_accuracy: 0.56,
    factuality_f1: 0.52,
    semantic_similarity: 0.68,
    faithfulness: 0.75,
    semantic_coverage: 0.54,
    retrieved_ids: ["c_01", "c_02", "c_03"],
    gold_ids: ["c_03"],
    execution_error: false,
  },
  {
    question_id: "q_ind_04",
    question_type: "Out-of-Scope",
    prediction: "信息不足，知识库未提及该核电阀门的特定参数。",
    answer_accuracy: 0.1,
    factuality_f1: 0.0,
    semantic_similarity: 0.4,
    faithfulness: 1.0,
    semantic_coverage: 0.0,
    retrieved_ids: [],
    gold_ids: ["c_missing"],
    execution_error: false,
  },
  {
    question_id: "q_ind_05",
    question_type: "Timeout Guard",
    prediction: "",
    answer_accuracy: 0.0,
    factuality_f1: 0.0,
    semantic_similarity: 0.0,
    faithfulness: 0.0,
    semantic_coverage: 0.0,
    retrieved_ids: [],
    gold_ids: ["c_01"],
    execution_error: true,
  },
];

export function EvalLabWorkspace({ session }: Props) {
  const barRef = useRef<HTMLDivElement | null>(null);
  const pieRef = useRef<HTMLDivElement | null>(null);

  const [modeName, setModeName] = useState("ppr_auto_s4");
  const [rawJson, setRawJson] = useState(() => JSON.stringify(DEFAULT_EVAL_ITEMS, null, 2));
  const [res, setRes] = useState<EvalSummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  async function runEval() {
    setRunning(true);
    setError(null);
    try {
      const items = JSON.parse(rawJson) as EvalMetricItemInput[];
      const data = await apiClient.evalMetrics(session, modeName, items);
      setRes(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  useEffect(() => {
    void runEval();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!res || !barRef.current || !pieRef.current) return;
    const s = res.summary;
    const barChart = echarts.init(barRef.current);
    barChart.setOption({
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      grid: { left: 48, right: 20, top: 30, bottom: 35 },
      xAxis: {
        type: "category",
        data: ["Accuracy", "Faithfulness", "Coverage", "Hit@5", "MRR@10"],
        axisLabel: { color: "#94a3b8", fontSize: 11 },
      },
      yAxis: {
        type: "value",
        max: 100,
        axisLabel: { color: "#94a3b8", formatter: "{value}%" },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      series: [
        {
          type: "bar",
          barWidth: 32,
          data: [
            Number((s.avg_answer_accuracy * 100).toFixed(1)),
            Number((s.avg_faithfulness * 100).toFixed(1)),
            Number((s.avg_semantic_coverage * 100).toFixed(1)),
            Number(((s.avg_hit_at_5 ?? 0) * 100).toFixed(1)),
            Number(((s.avg_mrr_at_10 ?? 0) * 100).toFixed(1)),
          ],
          itemStyle: {
            borderRadius: [6, 6, 0, 0],
            color: "#38bdf8",
          },
          label: { show: true, position: "top", color: "#e2e8f0", formatter: "{c}%" },
        },
      ],
    });

    const pieChart = echarts.init(pieRef.current);
    const dist = s.answerability_distribution;
    pieChart.setOption({
      backgroundColor: "transparent",
      tooltip: { trigger: "item", formatter: "{b}: {c}%" },
      legend: { bottom: 0, textStyle: { color: "#94a3b8", fontSize: 11 } },
      series: [
        {
          type: "pie",
          radius: ["45%", "72%"],
          center: ["50%", "44%"],
          label: { show: false },
          data: [
            { name: "FULL 完全可答", value: Number(((dist.FULL ?? 0) * 100).toFixed(1)), itemStyle: { color: "#34d399" } },
            { name: "PARTIAL 部分可答", value: Number(((dist.PARTIAL ?? 0) * 100).toFixed(1)), itemStyle: { color: "#fbbf24" } },
            { name: "NONE 不可答/拒答", value: Number(((dist.NONE ?? 0) * 100).toFixed(1)), itemStyle: { color: "#f87171" } },
            {
              name: "EXECUTION_ERROR 执行异常",
              value: Number(((dist.EXECUTION_ERROR ?? 0) * 100).toFixed(1)),
              itemStyle: { color: "#94a3b8" },
            },
          ],
        },
      ],
    });

    return () => {
      barChart.dispose();
      pieChart.dispose();
    };
  }, [res]);

  return (
    <div className="space-y-4">
      <div className="panel p-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-sky-400" />
          <h2 className="text-base font-semibold">评测回归实验室 v2（4D Judge + Hit@K / MRR@10 + 三态可答性）</h2>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={modeName}
            onChange={(e) => setModeName(e.target.value)}
            className="panel-subtle px-2.5 py-1.5 text-xs outline-none w-36"
          />
          <button
            onClick={() => void runEval()}
            disabled={running}
            className="px-4 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-medium flex items-center gap-1.5 cursor-pointer"
          >
            <Play className="w-3.5 h-3.5" /> 计算评测指标
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-500/50 bg-rose-500/10 px-4 py-2 text-xs text-rose-300">
          {error}
        </div>
      )}

      {res && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div className="panel p-3">
              <div className="text-[11px] text-slate-400">有效样本 / 执行异常</div>
              <div className="text-lg font-bold mt-1">
                {res.summary.valid_scored_count} /{" "}
                <span className="text-rose-400">{res.summary.execution_error_count}</span>
              </div>
            </div>
            <div className="panel p-3">
              <div className="text-[11px] text-slate-400">Answer Accuracy</div>
              <div className="text-lg font-bold text-sky-400 mt-1">
                {(res.summary.avg_answer_accuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="panel p-3">
              <div className="text-[11px] text-slate-400">Faithfulness</div>
              <div className="text-lg font-bold text-emerald-400 mt-1">
                {(res.summary.avg_faithfulness * 100).toFixed(1)}%
              </div>
            </div>
            <div className="panel p-3">
              <div className="text-[11px] text-slate-400">MRR@10 / Hit@5</div>
              <div className="text-lg font-bold text-amber-300 mt-1">
                {((res.summary.avg_mrr_at_10 ?? 0) * 100).toFixed(1)}% /{" "}
                {((res.summary.avg_hit_at_5 ?? 0) * 100).toFixed(0)}%
              </div>
            </div>
            <div className="panel p-3">
              <div className="text-[11px] text-slate-400">业务拒答 vs 语义拒答</div>
              <div className="text-lg font-bold mt-1">
                {(res.summary.business_refusal_rate * 100).toFixed(0)}% /{" "}
                <span className="text-rose-400">{(res.summary.semantic_refusal_rate * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="panel p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                核心检索与回答质量（执行异常已隔离出分母）
              </h3>
              <div ref={barRef} className="w-full h-64" />
            </div>
            <div className="panel p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                三态可答性分布（FULL / PARTIAL / NONE / ERROR）
              </h3>
              <div ref={pieRef} className="w-full h-64" />
            </div>
          </div>
        </>
      )}

      <div className="panel p-4 space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          评测样本输入（JSON，可编辑并实时调用 <code>POST /api/v1/eval/metrics</code>）
        </h3>
        <textarea
          value={rawJson}
          onChange={(e) => setRawJson(e.target.value)}
          rows={8}
          className="panel-subtle w-full p-3 text-xs font-mono outline-none"
        />
      </div>
    </div>
  );
}

# 第二阶段收尾：高难多跳数据集双域基准报告与架构边界总结

> **记录日期**：2026-11-22  
> **评测对象**：`GraphRAG-Bench (ICLR)` 双域（`Medical` 全量 1.05 MB NCCN 临床指南 + `Novel` 2 部长篇叙事 `Novel-8559` / `Novel-44557`）  
> **评测规模**：每域分层均衡抽样 `25` 题（`Complex Reasoning` + `Contextual Summarize`，`seed=42`）× 4 范式 = **50 题 / 300 次调用**  
> **运行环境**：本机 Ollama `gemma4:12b`（`think=False`，`num_ctx=16384`，`local_hash` 嵌入，零测试题泄漏建图）

---

## 1. 建图规模与零泄漏约束

| 数据域 | 存储路径（本地 `.gitignore`） | 语料规模 | 切片数 | 实体数 | 关系边数 | Louvain 社区 |
|---|---|---|---:|---:|---:|---:|
| **Medical（全量）** | `storage/graphrag_bench_medical` | 1.05 MB（53 章节） | 1,528 | 450 | 6,522 | 9 |
| **Novel（2 部长篇）** | `storage/graphrag_bench_novel` | `Novel-8559` + `Novel-44557` | 702 | 450 | 2,683 | 15 |

> 同步完成了 `IBM/AssetOpsBench`（`main` 分支）及其 ACL 2026 `IndustryAssetEQA` 分支体检：`AssetOpsBench` 主干为需连活体 CouchDB 与 5 个 MCP 工具服务的 Agent 工具调用环境；`IndustryAssetEQA` 包含 1,290 行 OREDA 工业 FMEA RDF 图谱（`FMEA_KG.ttl`），留作第三阶段工业 RDF 融合语料。

---

## 2. 双域四范式量化评测总表

### 2.1 `GraphRAG-Bench Medical`（25 题，0 崩溃）

| 范式模式 | 检索证据召回 (`Ret EvRec`) | 回答证据覆盖 (`Ans EvCov`) | Token-F1 | ROUGE-L | Grounding | 平均调用 | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `naive_rag` | 47.7% | 41.4% | 29.6% | 22.8% | - | 1.0 | 3.90s |
| `hybrid_direct` | 52.2% (+4.5) | **48.4% (#1)** | 21.4% | 16.3% | - | 1.0 | 11.32s |
| `ppr_cot` | 55.7% (+8.0) | 44.7% | 24.3% | 18.9% | **100.0%** | 1.0 | 33.86s |
| `ppr_tot` | **60.9% (+13.2, #1)** | 29.4% | 17.5% | 14.2% | **100.0%** | 3.0 | 51.79s |

分题型拆解：
- `Complex Reasoning`：`Ret EvRec` 从 `54.3% (naive_rag)` → `57.8% (hybrid)` → `60.1% (ppr_cot)` → **`65.4% (ppr_tot)`**；`Ans EvCov` 由 `hybrid_direct`（57.6%）与 `ppr_cot`（54.6%）领跑。
- `Contextual Summarize`：`Ret EvRec` 从 `40.6%` → `46.1%` → `50.9%` → **`56.0%`**；`Ans EvCov` 由 `hybrid_direct`（38.4%）与 `ppr_cot`（33.9%）领跑，`ppr_tot` 为 22.4%。

### 2.2 `GraphRAG-Bench Novel`（25 题，0 崩溃）

| 范式模式 | 检索证据召回 (`Ret EvRec`) | 回答证据覆盖 (`Ans EvCov`) | Token-F1 | ROUGE-L | Grounding | 平均调用 | 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `naive_rag` | 29.5% | 50.9% | 27.9% | 23.0% | - | 1.0 | 3.63s |
| `hybrid_direct` | 35.5% (+6.0) | **53.9% (#1)** | 17.3% | 12.9% | - | 1.0 | 12.59s |
| `ppr_cot` | 44.1% (+14.6) | 51.4% | 20.4% | 15.8% | **94.0%** | 1.1 | 36.73s |
| `ppr_tot` | **47.1% (+17.6, #1)** | 35.9% | 14.0% | 11.9% | **100.0%** | 3.0 | 52.43s |

分题型拆解：
- `Complex Reasoning`：`Ret EvRec` 从 `28.5% (naive_rag)` → `37.0%` → `45.0%` → **`47.7% (ppr_tot, +19.2 pts)`**；
- `Contextual Summarize`：`Ret EvRec` 从 `30.5%` → `34.0%` → `43.2%` → **`46.4%`**；`Ans EvCov` 由 `hybrid_direct`（53.6%）与 `ppr_cot`（49.9%）领跑。

---

## 3. 核心法证结论：检索-生成“剪刀差”的三重根因

1. **PPR + 多分支查图在检索端取得压倒性胜利**：双域 `Ret EvRec` 严格单调递增（Medical **+13.2 pts**，Novel **+17.6 pts**），证明异构图 PPR 扩散与 ToT 分支扩展在跨文档/跨章节多跳检索上显著超越纯向量检索。
2. **纯词汇 `Token-F1` 存在“短拒答复读陷阱”**：当 `naive_rag` 召回 0% 证据时，输出简短拒答（如 *"there is no mention of X or Y"*）因复读题干关键词反而获得极高 `Token-F1` 精确率；而 `hybrid_direct` 与 `ppr_cot` 输出详实溯源回答，在更真实的 **回答证据覆盖率（`Ans EvCov`）** 上全面反超 `naive_rag`。
3. **ToT 单胜剪枝的适用边界清晰暴露**：
   - 在**互斥根因定位 / 多症状排障**场景（如工业故障诊断与 `FailureSensorIQ`），ToT 竞争剪枝优势显著；
   - 在需要多面聚合的 **`Contextual Summarize`** 以及无类型 `CO_OCCURS_WITH` 共现图上，Phase 3 强制只保留 1 条胜出分支反而会丢弃其余 2 条分支召回的有效证据片段，造成检索召回（60.9%）与回答覆盖（29.4%）的剪刀差。

---

## 4. 第二阶段总收官状态与第三阶段演进输入

- **第二阶段（Stage 2.1 ~ 2.4）全面收官**：已完成 `LLMHarness` + 三级无损优先 Compact + Scope-Aware L0 修复（`FailureSensorIQ L2_10_options` **36.8% → 95.0%**、`L3` **100.0%**）+ `HippoRAG 2` 风格 PPR + 显性 `Graph-CoT`/`Graph-ToT` + `GroundingAudit` + 6 标签页 Streamlit 工作台（锁定 `failuresensoriq` 主对象，Ego-Graph 降噪 `-82%~-88%`，`SHACL-lite` 8/8 100% PASS）。
- **第三阶段（Stage 3）重点攻关方向**：
  1. **语义向量与混合检索升级**：接入真实稠密嵌入模型（如 Ollama `nomic-embed-text` / `bge-m3`）+ BM25 混合召回，突破 `local_hash` 在千级切片上的召回上限；
  2. **自适应任务路由（Adaptive Routing）**：互斥排障走 **ToT 竞争剪枝**，多面摘要走 **PPR + 多分支无剪枝聚合（Multi-Branch Aggregation）**，结合动态早停压缩延迟；
  3. **LLM-as-a-Judge 评测器**：引入语义正确率与忠实度裁判，彻底消除 `Token-F1` 拒答失真；
  4. **2026 前沿范式吸收**：探索 `HyperGraphRAG` 多症状超边建模与 `A-MEM` 验证后记忆写回机制。

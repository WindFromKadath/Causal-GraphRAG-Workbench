# Stage 2.1 全量基准测试报告、差异法证审计与前沿架构路线

> **记录日期**：2026-11-22  
> **评测规模**：IBM FailureSensorIQ 4 级难度集 × 每集 500 题 × 4 基线模式 = **2,000 题 / 8,000 次推理调用**（`seed=42`）  
> **运行时配置**：本机 Ollama `gemma4:12b`（Q4_K_M，100% GPU 常驻，`think=False`，`num_ctx=32768`，`concise_json=True`）

---

## 1. 四基线全量量化结果总表

对比 4 种推理/检索模式：
- `llm_only`：纯参数记忆直答；
- `naive_rag`：纯向量文本片段召回；
- `graphrag_kg_raw`（Stage 1）：全资产 1-hop 子图 + 3 段来源文本切片；
- `graphrag_kg`（Stage 2.1）：`LLMHarness` + L0 题干锚点隔离压缩。

### 1.1 Exact Match / Macro-F1 总表

| 难度分级 (Suite) | 题数 | `llm_only` | `naive_rag` | `graphrag_kg_raw` | `graphrag_kg` | KG vs LLM | Compact vs Raw |
|---|---|---|---|---|---|---|---|
| `L1_standard_all`（标准 5 选 1） | 500 | **59.8%** (299/500) | **83.4%** (417/500) | **99.6%** (498/500) | **99.2%** (496/500) | +39.4% | -0.4% (-2题) |
| `L2_10_options`（开放干扰 10 选 1） | 500 | **17.4%** (87/500) | **33.4%** (167/500) | **49.4%** (247/500) | **36.8%** (184/500) | +19.4% | -12.6% (-63题) |
| `L3_multi_answers`（5 选 2 多选） | 500 | **37.8%** (F1 62.2%) | **69.8%** (F1 81.4%) | **97.2%** (F1 98.4%) | **100.0%** (F1 100.0%) | +62.2% | **+2.8% (+14题)** |
| `L4_perturbed_simple`（`P)~T)` 扰动 + 去元数据） | 500 | **61.4%** (307/500) | **81.0%** (405/500) | **98.8%** (494/500) | **99.4%** (497/500) | +38.0% | **+0.6% (+3题)** |

### 1.2 逐题胜负差距矩阵（Pairwise Win/Loss vs `graphrag_kg`）

| 难度分级 | 对比基线 | 共同答对 | 仅 KG 答对 (Win) | 仅基线答对 (Loss) | 共同答错 | 净胜差距 |
|---|---|---|---|---|---|---|
| `L1_standard_all` | `llm_only` | 299 | **197** | 0 | 4 | **+197 (+39.40%)** |
| `L1_standard_all` | `naive_rag` | 414 | **82** | 3 | 1 | **+79 (+15.80%)** |
| `L1_standard_all` | `graphrag_kg_raw` | 494 | **2** | 4 | 0 | **-2 (-0.40%)** |
| `L2_10_options` | `llm_only` | 63 | **121** | 24 | 292 | **+97 (+19.40%)** |
| `L2_10_options` | `naive_rag` | 122 | **62** | 45 | 271 | **+17 (+3.40%)** |
| `L2_10_options` | `graphrag_kg_raw` | 164 | **20** | 83 | 233 | **-63 (-12.60%)** |
| `L3_multi_answers` | `llm_only` | 189 | **311** | 0 | 0 | **+311 (+62.20%)** |
| `L3_multi_answers` | `naive_rag` | 349 | **151** | 0 | 0 | **+151 (+30.20%)** |
| `L3_multi_answers` | `graphrag_kg_raw` | 486 | **14** | 0 | 0 | **+14 (+2.80%)** |
| `L4_perturbed_simple` | `llm_only` | 306 | **191** | 1 | 2 | **+190 (+38.00%)** |
| `L4_perturbed_simple` | `naive_rag` | 403 | **94** | 2 | 1 | **+92 (+18.40%)** |
| `L4_perturbed_simple` | `graphrag_kg_raw` | 491 | **6** | 3 | 0 | **+3 (+0.60%)** |

### 1.3 成本、延迟与稳定性遥测

| 难度分级 | 模式 | 平均 Prompt Tok | 修复重试 | 空回复数 | 资产链接率 | 平均耗时 |
|---|---|---:|---:|---:|---:|---:|
| `L1_standard_all` | `graphrag_kg_raw` → `graphrag_kg` | `592 → 195` (**-67.1%**) | `0 → 0` | `0 → 0` | 100.0% | `1.42s → 0.74s` (**1.92×**) |
| `L2_10_options` | `graphrag_kg_raw` → `graphrag_kg` | `621 → 222` (**-64.3%**) | `0 → 0` | `0 → 0` | 100.0% | `1.50s → 0.82s` (**1.83×**) |
| `L3_multi_answers` | `graphrag_kg_raw` → `graphrag_kg` | `600 → 198` (**-67.0%**) | `0 → 0` | `0 → 0` | 100.0% | `1.49s → 0.82s` (**1.82×**) |
| `L4_perturbed_simple` | `graphrag_kg_raw` → `graphrag_kg` | `588 → 193` (**-67.2%**) | `0 → 0` | `0 → 0` | 100.0% | `1.37s → 0.73s` (**1.88×**) |

---

## 2. `raw` vs `compact` 差异法证审计（根因与理论上限）

通过对 2,000 题逐题 checkpoint 展开程序化法证分类，共定位 5 项根因：

1. **封闭世界边界丢失（Asset Scope Loss）+ 二元 Prompt 误导**（占 `L2` `raw` 胜出 **74/83 = 89.2%**，且占 `L2` 双方同错 **222/233 = 95.3%**）：
   - `L2_10_options` 否定题的 10 个选项属于三态：`[目标已关联]`、`[本设备合法实体但未关联（唯一标准答案）]`、`[跨设备外来噪声（5~6 个）]`。
   - 当前 L0 Compact 仅保留单行目标关联边，剪掉了设备的完整合法实体集；且 Prompt 指令仅写 *"select the option(s) NOT listed in the KG"*，导致模型面对 6~7 个未出现选项时无法区分“本设备未关联项”与“跨设备干扰项”。
   - `raw` 因全量塞入该资产的 10+ 条关系而无意间暴露了设备合法边界，故多对 74 题；但同样因缺乏显式三态区分，仍错掉 222 题。
   - **纯 KG 理论上限验算**：仅凭现有 233 实体 ISO KG，若显式给出 `Target Links` + `Asset Domain Scope`，`L2_10_options` 有 **472~482 / 500（94.4%–96.4%）** 存在唯一确定解。
2. **极简上下文被模型参数先验劫持**（`L1` 3 题、`L2` 正向 5 题、`L4` 2 题）：
   - 在 `think=False` + 极简 JSON 单步直出下，当 KG 仅提供 1 行紧凑事实时，强领域先验干扰项（如 `damaged seals → oil leakage`、`rotor windings fault → partial discharge`）会覆盖单行事实。
3. **跨标准近似词表面形式碰撞**（`L2` 3 题）：
   - 如选项同时出现 `coast down time` 与 `oil debris/ contamination`，模型在无精确匹配约束时将外部近似词误当作 KG 中 `oil debris` 的同义词。
4. **统计层否定模板漏检**（`L2` 93 题分题型归类偏差）：
   - `_infer_relevancy_label` 缺少 `least `, `disregard`, `unimportant`, `non-relevant`, `exclude`, `inapplicable`, `not pertinent` 等关键词。
5. **`raw` 冗余邻居导致多目标注意力稀释**（`compact` 在 `L3` 净胜 **14 题**、`L4` 净胜 **3 题**）：
   - `raw` 约 `2,740 字符` / `60~105 节点` 的大子图在 5 选 2 多选题中引发漏选/错选，而 `compact` 以 `198 tok` 实现 **500/500 (100.0%)** 零失误。

---

## 3. 并入 Stage 2.2 的待修复项（不再单独拆分 Stage 2.1.5）

上述审计暴露的问题将作为 **Stage 2.2 前置待修复项** 与显性 `Graph-CoT` 引擎同步落地：

- **修复项 1（Scope-Aware L0 Compaction）**：在 L0 压缩输出中固定附带 `Asset Domain Scope` 与 `In-Asset Unlinked Candidates` 差集视图 + 1 行 L1 精简 Provenance 证据强化（增量约 +35~50 tok）。
- **修复项 2（三态 Prompt 规则与精确匹配约束）**：将 MCQA 规则升级为 `LINKED` / `IN_ASSET_UNLINKED` / `OUT_OF_ASSET` 三态判别，并禁止未登录近似词模糊等价。
- **修复项 3（`_infer_relevancy_label` 否定词表补全）**：修正否定题型识别覆盖率。
- **修复项 4（Explicit Graph-CoT 逐项核对 `option_grounding`）**：在 Stage 2.2 `GraphCoTEngine` 中强制先做结构化选项/证据核对与 `GroundingAudit`，再输出最终结论，根治参数先验劫持。

---

## 4. 前沿高难基准与新范式架构对标（GitHub 查证）

### 4.1 更高难度数据集升级梯队（统一下载至被 `.gitignore` 屏蔽的 `data/private/`）

| 基准仓库 | 领域与特点 | 引入价值 |
|---|---|---|
| `IBM/AssetOpsBench` | 工业运维多模态（FMEA + 工单 + IoT 时序 + 知识图谱 + 多步 Agent 任务） | FailureSensorIQ 同源升级版，检验工业多步诊断与 Harness 编排 |
| `GraphRAG-Bench/GraphRAG-Benchmark` (ICLR) | 医学 + 长篇叙事 4 级难度（事实检索 → 多跳推理 → 全局摘要 → 综合生成） | 检验通用 GraphRAG 流水线何时真正超越 Naive RAG |
| `snap-stanford/stark` (NeurIPS) | 半结构化超大知识库（PrimeKG 12.9 万节点 / 810 万边等） | 检验图谱规模扩展与“结构关系 + 长文本属性”联合检索 |
| `Y-Sui/OKGQA` / `MuSiQue` / `aistairc/home-kgqa` | 2–4 跳防捷径多跳推理、抗噪与时空聚合 | 检验 Stage 2.3 `Graph-ToT` 动态多跳搜索价值 |

### 4.2 可吸收的前沿 GraphRAG / KG 推理新范式

1. **HippoRAG 2（`OSU-NLP-Group/HippoRAG`）**：基于实体-段落异构图上的 **Personalized PageRank (PPR)** 实现一步多跳联想扩散，拟在 Stage 2.2 优先集成至 `pipelines/retriever.py`（新增 `ppr` 检索模式）。
2. **OpenSPG KAG（`OpenSPG/KAG`）**：领域本体强约束 + 文本/图谱双向互索引 + 逻辑形式核对算子，为 Stage 2.2 `Graph-CoT` + `GroundingAudit` 提供设计参照。
3. **SubgraphRAG / G-Retriever / PathRAG**：以路径流剪枝与轻量连通子图提取替代粗暴的 k-hop 邻居铺开，持续强化 `ContextCompactor` L0 通用能力。
4. **Think-on-Graph（`GasolSun36/ToG` / `DataArcTech/ToG-3`）**：可控预算的迭代式 Beam Search 图搜索与动态剪枝，作为 Stage 2.3 `Graph-ToT` 的核心算法底座。
5. **Graphiti（`getzep/graphiti`）与 LightRAG / LazyGraphRAG**：时序有效期（`valid_from/valid_to`）与增量/按需构图，规划为 Stage 3 长期演进方向。

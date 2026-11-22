# 修改日志 (CHANGELOG)

本文件遵循“**每次 Git 提交同步记录一次修改日志**”的项目管理规范，用于追溯每次架构演进、功能实现、核验结果与隐私安全状态。

> 隐私约束：日志中禁止记录任何 API Key、本机绝对路径、个人邮箱或私有业务数据。

---

## [2026-11-22] Harder Benchmark Prep - 跨语言公平性修复与 GraphRAG-Bench 双域（Medical + Novel）断点评测就绪
- **提交类型**：`feat`
- **变更背景**：为开展 Stage 2 在更高难度多跳数据集（`GraphRAG-Bench` 医学全量与长篇叙事双域）上的公平四范式对比评测，修复 `direct` / `cot` 回答语言硬编码偏差，并补全双域建图、断点续跑与检索证据召回指标。
- **主要修改**：
  - 修复 `src/llm_graphrag_kg/pipelines/retriever.py` 与 `src/llm_graphrag_kg/reasoning/cot.py`：将原硬编码“仅用中文回答”改为严格跟随提问语言（`in the same language as the question`），确保 `naive_rag`、`hybrid_direct`、`ppr_cot`、`ppr_tot` 在英文高难基准上绝对同口径公平对比。
  - 升级 `src/llm_graphrag_kg/adapters/graphrag_bench.py` 与 `src/llm_graphrag_kg/cli.py`：支持全量医学语料与指定长篇叙事（`--source-filter`）双域零泄漏建图、新增 JSONL 逐题断点续跑（`--output-tag` / `--resume` / `--progress-every`）与 **检索证据召回率（`retrieval_evidence_recall`）** 指标。
- **验证方式**：
  - `ruff check` 0 告警，`pytest -q` **23/23 全部通过**。
  - 本地零泄漏建成两套大图：`storage/graphrag_bench_medical`（全量 1.05 MB：53 文档 / 1,528 切片 / 450 实体 / 6,522 边 / 9 社区）与 `storage/graphrag_bench_novel`（`Novel-8559` + `Novel-44557`：702 切片 / 450 实体 / 2,683 边 / 15 社区）。
- **隐私与迁移自检**：通过 `scripts/privacy_check.py --staged --require-changelog`，外部语料与图谱索引均严格隔离在 `.gitignore` 目录。

---

## [2026-11-22] Stage 2.4 - 可视化核验工作台全面升级（锁定 FailureSensorIQ 主对象：Ego-Graph 聚焦、CoT/ToT 可读思维回放与 SHACL-lite 全量本体质检）
- **提交类型**：`feat`
- **变更背景**：完成 `docs/stage2_harness_reasoning_plan.md` 第二阶段最后一个子阶段（Stage 2.4），锁定 `configs/failuresensoriq.yaml`（233 实体 / 692 边 / 四层严格 ISO 本体）作为默认主验收对象，解决大图毛线球拥挤、思维回放哈希 ID 难读及缺乏全量本体合规质检三大痛点。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/viz/schema_validator.py`（`validate_kg_shacl_lite`）：实现 100% 全量图谱 SHACL-lite 本体与溯源合规质检，覆盖 5 项通用图卫生规则（实体类型白名单、无悬空边、无自环边、零幽灵实体/三元组原文溯源、孤立实体检测）及 3 项 ISO FailureSensorIQ 四层本体强约束（`HAS_FAILURE_MODE` / `INSTANCE_OF` / `MONITORED_BY` 定义域与值域、FailureMode 资产隔离与传感器覆盖、SignalChannel ER 拼写洁净度），支持 Markdown 报告导出。
  - 升级 `src/llm_graphrag_kg/viz/workbench_data.py`：新增单设备/单节点 `extract_ego_subgraph()`（支持 1~2 跳局部聚焦、关系过滤与节点上限控制，在 233 节点 FailureSensorIQ 图谱上单设备可降噪 **-82% ~ -88%**）与 `build_readable_citation_maps()`（将冷冰冰的 `ent_*` / `rel_*` 哈希 ID 翻译为人类可读实体与三元组标签）。
  - 全面升级 `src/llm_graphrag_kg/viz/app.py`（6 标签页闭环工作台）：默认锁定 `configs/failuresensoriq.yaml`（仍支持侧边栏切换全部 YAML）；新增 `ollama_native` 与 `think=False` 开关；「交互图谱」页默认对 >80 节点大图启用 **Ego-Graph 聚焦模式**；「问答与思维回放」页集成 `ppr/hybrid/local/global` + `cot/tot/direct` + `Compact` 压缩遥测 + `GroundingAudit` 红绿灯 + **ToT 分支双轨（`G`/`S` 得分 + L2 剪枝记忆）与 CoT 可读溯源卡片**；新增独立「🛡️ SHACL-lite 本体质检」页。
  - 在 `src/llm_graphrag_kg/cli.py` 新增 `llm-graphrag-kg shacl-check` 终端质检命令，并在 `tests/test_workbench.py` 扩充 Ego-Graph、可读引用、SHACL-lite 正/负例注入检测及 6 标签页无头 `AppTest` 回归测试。
- **验证方式**：
  - `ruff check` 0 告警，`pytest -v` **23/23 全部通过**（含 Streamlit `AppTest` 无头渲染校验默认锁定 `configs/failuresensoriq.yaml` 与 6 标签页）。
  - 在本地真实 `storage/failuresensoriq`（233 实体 / 692 关系）上执行 `llm-graphrag-kg shacl-check --config-file configs/failuresensoriq.yaml`：**8/8 规则 100.0% 通过（0 Error / 0 Warning）**；单设备 Ego-Graph（如 `pump` 233 → 27 节点 `-88.4%`、`steam turbine` 233 → 29 节点 `-87.6%`）验证无毛线球拥挤。
- **隐私与迁移自检**：通过 `scripts/privacy_check.py --staged --require-changelog`，无密钥、本机路径或私有数据入库。

---

## [2026-11-22] Stage 2.3 - 显性可控 Graph-ToT 引擎（ToG 范式 + L2 剪枝折叠）与 GraphRAG-Bench 多跳基准适配器
- **提交类型**：`feat`
- **变更背景**：落地 `docs/stage2_harness_reasoning_plan.md` 规划的 Stage 2.3，解决多症状并发工业诊断与高难多跳问答中“需要并行展开竞争假设、逐分支查图并用反证剪枝”的核心需求，并接入更高难度的 `GraphRAG-Bench (ICLR)` 医学多跳基准进行跨范式实证。
- **主要修改**：
  - 扩展 `src/llm_graphrag_kg/reasoning/schema.py` 并新增 `src/llm_graphrag_kg/reasoning/tot.py`（`GraphToTEngine`）：实现 4 阶段可控预算图谱思维树（**Branch 假设生成 → Ground 逐分支独立 k-hop 查图与确定性图谱打分 → Score & Prune 联合语义打分、反证剪枝与 `ContextCompactor.fold_state_l2` 剪枝分支状态折叠 → Compact & Synthesize 综合收敛 + `audit_cot_grounding` 忠实度门禁**），默认严格控制 3 次结构化 LLM 调用（上限 `max_llm_calls=6`）。
  - 升级 `src/llm_graphrag_kg/pipelines/retriever.py`、`workflow.py`、`providers/llm/mock.py` 与 `cli.py`：全链路打通 `--reasoning tot`（`tot` 下自动关闭 Ollama 隐式思考），CLI 新增 **Explicit Graph-ToT Hypothesis Branches** 对比表展示胜出/剪枝状态、双维得分（图谱确定性得分 `G` + 语义得分 `S`）及剪枝反证依据。
  - 新增高难多跳基准适配器 `src/llm_graphrag_kg/adapters/graphrag_bench.py`、配置 `configs/graphrag_bench.yaml` 及 CLI 命令 `ingest-grb` / `eval-grb`（严格零测试题泄漏建图，内置 Token-F1、ROUGE-L、证据覆盖率与 Grounding Score 自动化评测）。
  - 新增单元测试 `tests/test_reasoning_tot.py`，更新 `docs/stage2_harness_reasoning_plan.md`。
- **验证方式**：
  - `ruff check` 0 告警，`pytest -v` **22/22 全部通过**。
  - 本机 Ollama `gemma4:12b` 实跑多症状并发工业故障诊断（`--mode ppr --reasoning tot --compact`）：稳定展开 3 条竞争假设（准确选中 `H1: 轴系不对中 [0.86]`，剪枝 `H2: 滚动轴承外圈剥落` 与 `H3: 润滑系统 [0.61]` 并给出反证），三级压缩（L0+L1+L2）将上下文从 `3,716 → 740 tok (-80.1%)`，**`GroundingAudit` 4/4 步 100.0% 通过（0 编造引用）**。
  - 本机 Ollama `gemma4:12b` 实跑 `GraphRAG-Bench (Complex Reasoning)` 四范式对比：`ppr_tot` 以 **Token-F1 16.1%、ROUGE-L 14.9%、证据覆盖率 52.8%、Grounding Score 100.0%** 全面领先 `naive_rag`（11.3% / 39.1%）、`hybrid_direct`（7.2% / 39.5%）与 `ppr_cot`（8.1% / 29.1%），平均严格收敛于 `3.0` 次调用。
- **隐私与迁移自检**：`GraphRAG-Bench` 语料、索引与评测产物均隔离在 `.gitignore` 屏蔽的 `data/private/`、`storage/` 与 `outputs/`；已通过 `scripts/privacy_check.py --staged --require-changelog`。

---

## [2026-11-22] Stage 2.2 - 显性 Graph-CoT 引擎、GroundingAudit 忠实度门禁、HippoRAG 风格 PPR 检索与 Scope-Aware L0 修复
- **提交类型**：`feat`
- **变更背景**：落地 Stage 2.2 显性结构化思维链（`Graph-CoT`）与确定性忠实度校验，同时将 Stage 2.1 全量审计定位的 4 项待修复项（封闭世界边界丢失、三态 Prompt、否定模板漏检、极简先验劫持）以及 GitHub 前沿调研中选定的 HippoRAG 风格 PPR 多跳扩散检索一并整合实现。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/reasoning/`（`schema.py`、`cot.py`、`__init__.py`）：实现 4 阶段显性 `GraphCoTEngine`（`anchor → graph_traverse → evidence_check → synthesis`）、确定性 `audit_cot_grounding()`（逐条核验 `entity_id` / `relation_id` / `chunk_id` 是否真实存在于检索子图，拦截编造引用并触发单轮忠实度自愈修复）以及 MCQA 三态核对契约 `MCQACoTOutput`。
  - 升级 `src/llm_graphrag_kg/pipelines/retriever.py` 与 `workflow.py`：新增无 `scipy` 依赖的纯 Python 幂迭代 **Personalized PageRank（`--mode ppr`）** 实体-段落异构图多跳扩散检索模式，并打通 `--reasoning direct|cot` 全链路。
  - 升级 `src/llm_graphrag_kg/adapters/failuresensoriq.py`（合入全部 Stage 2.1 审计待修复项）：L0 压缩升级为 **Scope-Aware L0 Compaction**（同步提供 `Target Links` + `Other Valid ISO ... NOT linked/monitored` 资产域边界差集 + 1 行 L1 规范句强化）、升级三态精确匹配 Prompt 规则、补全 `_infer_relevancy_label` 否定词表，并支持 `--reasoning cot`。
  - 升级 `src/llm_graphrag_kg/cli.py` 与 `providers/llm/mock.py`：`query` 支持 `--mode ppr`、`--reasoning cot` 与 `--think/--no-think`（`cot` 下默认关闭隐式思考），并输出 CoT 步骤表与 `GroundingAudit` 面板；`eval-fsiq` / `stress-fsiq` 新增 `--reasoning direct|cot`。
  - 新增单元测试 `tests/test_reasoning_cot.py`。
- **验证方式**：
  - `ruff check` 0 告警，`pytest -v` **19/19 全部通过**。
  - 本机 Ollama `gemma4:12b` 实跑工业故障诊断 `--mode ppr --reasoning cot --compact`：上下文从 `9489 → 2369 tok (-75.0%)`，**`GroundingAudit` 4/4 步 100.0% 通过（0 编造引用）**，精准定位 `轴系不对中 → 激光对中 SOP`。
  - 本机 Ollama `gemma4:12b` 实跑 `FailureSensorIQ L2_10_options`（100 题，`seed=42`）：Scope-Aware L0 将 `graphrag_kg` Exact Match 从 **36.8% 一举跃升至 95.0%（95/100）**，大幅反超 `graphrag_kg_raw`（49.4%）**+45.6%** 且 0 重试 / 0 空回复；`--reasoning cot` 冒烟验证 10/10（100.0%）命中。
- **隐私与迁移自检**：新下载的高难基准（`GraphRAG-Bench`、`IBM/AssetOpsBench`）均严格存于被 `.gitignore` 屏蔽的 `data/private/`；已通过 `scripts/privacy_check.py --staged --require-changelog`。

---

## [2026-11-22] Stage 2.1 Benchmark Report - 2,000 题四基线评测沉淀、差异法证审计与前沿路线同步
- **提交类型**：`docs`
- **变更背景**：完成 FailureSensorIQ 4 级难度集 × 每集 500 题 × 4 模式（累计 2,000 题 / 8,000 次本机 `gemma4:12b` 推理）大规模基准评测后，将量化指标、`raw` vs `compact` 逐题法证根因、并入 Stage 2.2 的待修复项以及 GitHub 高难基准/新范式调研沉淀入库并同步至云端仓库。
- **主要修改**：
  - 新增 `docs/benchmark_stage2_1_audit_and_roadmap.md`：记录完整四基线 Exact Match/Macro-F1、Pairwise Win/Loss 矩阵与 Token/耗时遥测（`L3` 500/500 满分且降本 67%/提速 ~1.9×）；量化定位 `L2_10_options` 差距根因（封闭世界 Asset Scope 丢失 + 二元 Prompt 误导占 89.2%，理论上限 94.4%–96.4%）；汇总 `IBM/AssetOpsBench`、`GraphRAG-Bench`、`STaRK` 与 `HippoRAG 2` / `OpenSPG KAG` / `PathRAG` / `ToG` 对标路线。
  - 更新 `docs/stage2_harness_reasoning_plan.md` 与 `README.md`：标注 Stage 2.1 完成状态，并将 Scope-Aware L0 压缩、三态 Prompt、否定模板修复及 HippoRAG 风格 PPR 检索直接并入 Stage 2.2 交付范围。
- **验证方式**：通过 `ruff check`、`pytest -v`（15/15 通过）与 `scripts/privacy_check.py --staged --require-changelog`。
- **隐私与迁移自检**：仅提交脱敏汇总报告，原始数据与逐题 checkpoint 仍严格隔离在 `.gitignore` 屏蔽的 `data/private/` 与 `outputs/` 目录。

---

## [2026-11-22] Benchmark Prep - 全量四基线评测就绪（`--limit 0`、逐题胜负差距矩阵与 `--dry-run` 预估）
- **提交类型**：`feat`
- **变更背景**：为开展大规模/全量 FailureSensorIQ 对比测试、精确量化 `llm_only` / `naive_rag` / `graphrag_kg_raw`（Stage 1）与 `graphrag_kg`（Stage 2.1 Compact+Harness）之间的真实差距，完善全量挂机评测与统计分析能力。
- **主要修改**：
  - 升级 `src/llm_graphrag_kg/adapters/failuresensoriq.py`：支持 `limit=0` 全量跑测、新增 `graphrag_kg_raw`（Stage 1 无压缩基线）与 `graphrag_kg`（Stage 2.1 压缩基线）同场四模式对照、新增逐题胜负差距矩阵 `compute_pairwise_gaps()`、执行前耗时/断点预估器 `plan_failuresensoriq_stress_suite()`、每 N 题实时进度与 ETA 打印，以及单题瞬时网络超时自动重试保护。
  - 在 `src/llm_graphrag_kg/cli.py` 的 `stress-fsiq` 中新增 `--suites`、`--output-tag`、`--progress-every` 与 `--dry-run` 选项，并同步扩展 `tests/test_failuresensoriq.py`。
- **验证方式**：通过 `ruff check`、`pytest -v`（15/15 全部通过），并在本地真实 FailureSensorIQ 4 套全量数据上运行 `--dry-run` 核对精确题量与耗时预估。
- **隐私与迁移自检**：通过 `scripts/privacy_check.py --staged --require-changelog`，无密钥、本机路径或运行数据入库。

---

## [2026-11-22] Stage 2.1 - LLMHarness 推理驾驭层与三级无损优先上下文压缩 (Compact)
- **提交类型**：`feat`
- **变更背景**：落地 `docs/stage2_harness_reasoning_plan.md` 规划的 Stage 2.1 底座，解决裸模型调用缺乏预算护栏、JSON 解析无自愈以及高连通子图噪声导致 12B 模型注意力串台的问题。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/harness/context_compactor.py`：实现依赖零外部库的中英文 Token 估算器及三级无损优先压缩（L0 规则级子图锚点裁剪与紧凑符号编码、L1 小数安全分句的证据抽取式压缩、L2 保留数值/单位/实体/`chunk_id` 的结构化状态折叠 `fold_state_l2`）。
  - 新增 `src/llm_graphrag_kg/harness/runtime.py`：`LLMHarness` 统一封装 Prompt 编译、调用次数/耗时熔断护栏、Pydantic 结构化 JSON 校验与自动修复重试（`run_structured`）及全链路遥测账本 `TraceLedger`。
  - 在 `config.py` 新增向后兼容的 `HarnessConfig` / `ReasoningConfig`；将 `ContextCompactor` 与 `LLMHarness` 接入 `pipelines/retriever.py`、`pipelines/workflow.py`、`adapters/failuresensoriq.py`（实现 L0 题干锚点隔离，剥离资产名防子串误匹配）与 `cli.py`（`query --compact` 与 `stress-fsiq --compact` 压缩遥测展示）。
  - 新增单元测试套件 `tests/test_harness.py`。
- **验证方式**：
  - `ruff check` 0 告警，`pytest -v` **15/15 全部通过**。
  - 本机 Ollama `gemma4:12b` 实跑 `FailureSensorIQ L3_multi_answers`（100 题，`seed=42`，`think=False`）量化验收：KG 上下文从 `449.2 tok / 64.5 节点` 压降至 `49.9 tok / 6.6 节点`（**Token 缩减 `-88.9%`**），Exact Match 从 **94.0% 跃升至 100.0%（100/100）**，Macro-F1 从 **95.5% 跃升至 100.0%**，单题平均延迟从 **1.55s 降至 0.96s**，0 空回复。
- **隐私与迁移自检**：通过 `scripts/privacy_check.py --staged --require-changelog`，无密钥、本机路径或私有数据入库。

---

## [2026-11-22] Release Prep - 补充 MIT 开源协议、统一本地模型表述并完成 GitHub 推送前审核
- **提交类型**：`chore`
- **变更背景**：在首次公开推送至 GitHub 仓库前，补充 MIT 开源协议并对文档与代码注释中的本地模型示例名称做通用化收口。
- **主要修改**：
  - 新增 MIT `LICENSE` 文件并在 `pyproject.toml` 中声明 `license = { text = "MIT" }`。
  - 将 `.env.example`、`README.md`、`docs/`、`data/sample/` 及源码注释中特定本地运行时示例名统一替换为通用的 `Ollama / vLLM / LM Studio` 表述。
- **验证方式**：通过全量 `ruff check`、`pytest -v`（11/11 通过）与 `scripts/privacy_check.py --staged --require-changelog`。
- **隐私与迁移自检**：已核查全部提交作者邮箱、仓库对象体积与敏感规则，无任何隐私泄露风险。

---

## [2026-11-22] Stage 2 Plan - 记录第二阶段实现方案（LLM Harness + Compact + 显性 CoT/ToT）
- **提交类型**：`docs`
- **变更背景**：基于第一阶段 FailureSensorIQ 四级极限压测与本地 `gemma4:12b` 遥测诊断结果，正式确立并归档第二阶段演进方案，将“裸模型调用 + 单轮问答”升级为“Harness 驾驭 + 三级无损优先上下文压缩 + 显性 Graph-CoT / Graph-ToT 思维范式”。
- **主要修改**：
  - 新增正式架构规划文档 `docs/stage2_harness_reasoning_plan.md`（涵盖总体融合架构、L0/L1/L2 Compact 策略、`LLMHarness` 护栏与账本、显性 `Graph-CoT` / `Graph-ToT` 数据结构与算法流、Streamlit 思维回放 + Ego-Graph + SHACL-lite 质检联动，以及 Stage 2.1 ~ 2.4 量化验收标准）。
  - 更新 `README.md` 文档目录索引。
- **验证方式**：文档结构与 Mermaid 语法审查完成，通过 `scripts/privacy_check.py --staged --require-changelog`。
- **隐私与迁移自检**：确认文档无敏感密钥、无本机绝对路径、无个人邮箱。

---

## [2026-11-22] Phase 3.2 - Ollama 原生控制与 FailureSensorIQ 四级难度极限压测引擎
- **提交类型**：`feat`
- **变更背景**：基于实测定位了 Ollama 默认 4K 上下文截断与 `gemma4:12b` 默认思考模式过度耗时的瓶颈，将原生 `think` / `num_ctx` 控制与四级难度极限压测引擎正式沉淀进项目源码，支持每集 100 题（及全量）断点续跑压测。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/providers/llm/ollama_native.py`（`OllamaNativeLLMClient`），通过 `/api/chat` 精确控制 `think`、`num_ctx`、`num_predict` 并记录精确 `tok/s` 遥测；同步扩展 `config.py`、`.env.example`、`providers/factory.py` 与 `openai_compat.py`（支持 `max_tokens` 与推理字段安全回退）。
  - 升级 `src/llm_graphrag_kg/adapters/failuresensoriq.py`：实现无元数据自然语言资产实体链接（`detect_asset_from_question`）、兼容 `P)~T)` 扰动选项与 `5选2` 多选题解析与 `Exact Match + Macro-F1` 评测、极简 JSON 极速答题模式、JSONL 逐题断点续跑，以及四集统一调度器 `run_failuresensoriq_stress_suite()`。
  - 在 `src/llm_graphrag_kg/cli.py` 中新增 `llm-graphrag-kg stress-fsiq` 压测命令，同步更新 `MockLLMClient`、`README.md` 与 `tests/test_failuresensoriq.py`。
- **验证方式**：通过 `ruff check`、`pytest -v`（11/11 全部通过），并准备对本地 4 套 FailureSensorIQ 基准集执行每集 100 题（共 400 题 × 3 模式）实跑压测。
- **隐私与迁移自检**：所有压测数据与断点报告均输出至被 `.gitignore` 忽略的 `data/private/` 与 `outputs/`；已通过 `privacy_check.py --staged --require-changelog`。

---

## [2026-11-21] Phase 3.1 - IBM FailureSensorIQ 公开基准集适配、实体消歧与三基线量化评测
- **提交类型**：`feat`
- **变更背景**：从候选工业/核电数据集中选定基于 ISO 国际标准且 Apache-2.0 开源的 IBM FailureSensorIQ 作为主验证数据集，落地符合行业规范的 Ontology-First 建图、实体消歧（ER）与客观 MCQA 量化评测闭环。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/adapters/failuresensoriq.py`：实现 FailureSensorIQ 映射/原题 JSONL 适配器，构建四层规范本体（`Equipment --[HAS_FAILURE_MODE]--> FailureMode --[INSTANCE_OF]--> FailureCategory` 与 `FailureMode --[MONITORED_BY]--> SignalChannel`），按设备隔离失效模式防止跨设备传感污染，并通过显式实体消歧词表自动修正原始数据中的真实拼写误差（如 `temparature` → `temperature`、`coast down` → `coast down time`）且保留 `aliases`。
  - 新增三基线分层评测框架 `evaluate_failuresensoriq()`：同题对比 `llm_only`（纯模型参数记忆）、`naive_rag`（纯向量切片）与 `graphrag_kg`（ISO 图谱多跳子图增强），导出 `eval_report.md` 与 `eval_results.json`。
  - 新增场景配置 `configs/failuresensoriq.yaml`、CLI 命令 `ingest-fsiq` / `eval-fsiq`，扩展 `viz/html_exporter.py` 工业本体节点配色方案，并补充离线单元测试 `tests/test_failuresensoriq.py`。
- **验证方式**：通过 `ruff check`、`pytest -v`（10/10 全部通过），并基于本地 `data/private/failuresensoriq/` 完整实跑 10 类设备建图、审计、可视化导出及本机 Ollama `gemma4:12b` 量化对比评测。
- **隐私与迁移自检**：原始数据仅存于被 `.gitignore` 屏蔽的 `data/private/failuresensoriq/`，配置均采用相对路径；已通过 `privacy_check.py --staged --require-changelog`。

---

## [2026-11-20] Docs & Domain Corpus - 工程部署指南、故障诊断手册与工业故障诊断脱敏语料库
- **提交类型**：`docs` / `feat`
- **变更背景**：扩充面向真实工程落地的内容量，补齐框架本身的工程部署与故障排查文档，同时新增高密度、跨文档多跳因果结构的“工业设备工程部署与故障诊断”公开脱敏语料集及专用配置。
- **主要修改**：
  - 新增工程运维文档：`docs/engineering_deployment.md`（在线/内网离线 `uv` 部署、Ollama/vLLM 推理配置、systemd 托管与 Neo4j 迁移路径）与 `docs/troubleshooting_diagnosis.md`（FD-01 ~ FD-08 六大类工程故障诊断与排障手册）。
  - 新增工业故障诊断专用配置 `configs/fault_diagnosis.yaml` 与 6 篇公开脱敏领域语料 `data/sample_fault_diagnosis/`（覆盖设备传感、机械故障机理、信号处理与 AI 诊断算法、云边工程部署、ISO 20816 与现场 SOP、跨子系统多跳排障案例）。
  - 增强 `src/llm_graphrag_kg/viz/app.py` 侧边栏，支持一键发现并切换 `configs/*.yaml` 场景配置；同步补充 `MockLLMClient` 工业领域词表、`README.md` 索引与端到端测试用例 `tests/test_pipeline_e2e.py`。
- **验证方式**：通过 `ruff check`、`pytest -v`（8/8 通过）与故障诊断语料 CLI 全链路（`build` → `audit` → `export-viz` → `query --export-html`）实跑验证。
- **隐私与迁移自检**：所有新增语料均为公开通用工程知识，已通过 `privacy_check.py --staged --require-changelog`，无私有数据、密钥或本机绝对路径入库。

---

## [2026-11-20] Phase 2 - Streamlit 一站式可视化内容核验工作台
- **提交类型**：`feat`
- **变更背景**：面向零经验用户降低 GraphRAG + KG 调试门槛，在 Phase 1 静态导出基础上构建本地交互式核验工作台，实现“切片 → 实体与三元组溯源 → 社区聚类 → 实时问答与检索子图高亮”全流程同屏核验。
- **主要修改**：
  - 重构 `src/llm_graphrag_kg/viz/html_exporter.py`，抽离 `render_pyvis_html()` 复用渲染接口，支持按实体子集过滤并对标题执行 HTML 转义防护。
  - 新增纯 Python 数据准备与溯源辅助层 `src/llm_graphrag_kg/viz/workbench_data.py`（切片、实体、三元组、社区表格构造、多维筛选与原文切片反查）。
  - 新增 Streamlit 核验工作台 `src/llm_graphrag_kg/viz/app.py`（脱敏配置侧边栏、连通性检测、一键建图/导出、5 大核验标签页与问答检索子图实时高亮）。
  - 在 `src/llm_graphrag_kg/cli.py` 中新增 `llm-graphrag-kg ui` 一键启动命令，同步更新 `README.md` 与单元测试 `tests/test_workbench.py`。
- **验证方式**：通过 `uv sync --extra viz --extra dev`、`ruff check`、`pytest -v` 与 Streamlit `AppTest` 无头渲染验证。
- **隐私与迁移自检**：侧边栏配置预览强制调用 `redacted_summary()` 隐藏密钥；已通过 `privacy_check.py --staged --require-changelog`，无密钥、本地路径或运行产物入库。

---

## [2026-11-20] Phase 1 - LLM + GraphRAG + KG 核心闭环与 PyVis/GraphML 可视化核验
- **提交类型**：`feat`
- **变更背景**：在 Phase 0 隐私与工程基线之上，落地分层解耦的 LLM + GraphRAG + 知识图谱最小可行闭环（MVP）与第一阶段可视化核验工具。
- **主要修改**：
  - 新增核心领域模型与抽象协议：`src/llm_graphrag_kg/core/schema.py`、`interfaces.py` 与 `config.py`（支持 YAML 默认配置 + `.env` 密钥注入与日志脱敏）。
  - 新增可插拔适配层：`providers/llm/`（`OpenAICompatLLMClient` 兼容云端 API 与本机 Ollama/vLLM，附 `MockLLMClient` 离线调试）、`providers/embedding/`（`OpenAICompatEmbeddingClient` + `LocalHashEmbeddingClient`）、`providers/storage/`（`NetworkXGraphStore` + `LocalVectorStore`）与工厂模块 `providers/factory.py`。
  - 新增完整业务管线：`pipelines/ingestion.py`（文档加载与带字符溯源切片）、`extraction.py`（实体关系抽取与去重融合）、`community.py`（Louvain 社区发现与摘要）、`retriever.py`（`local` / `global` / `hybrid` 三模式检索与子图追踪）及 `workflow.py`。
  - 新增 Phase 1 可视化与核验层：`viz/audit_report.py`（Markdown/CSV 质量审计）、`viz/html_exporter.py`（PyVis 单文件交互式 HTML 与检索子图高亮）、`viz/graphml_exporter.py`（Gephi/Cytoscape 兼容导出）。
  - 新增统一命令行入口 `src/llm_graphrag_kg/cli.py`（`check-llm` / `build` / `audit` / `export-viz` / `query` / `privacy-check`）、3 份公开脱敏示例语料 `data/sample/` 及自动化测试套件 `tests/`。
- **验证方式**：通过 `pytest -v` 端到端与隐私检查测试、`ruff check` 静态检查，以及 CLI 全链路（`build` → `audit` → `export-viz` → `query --export-html`）实跑验证。
- **隐私与迁移自检**：已通过 `privacy_check.py --staged --require-changelog`，所有运行产物均输出至被 `.gitignore` 忽略的 `storage/` 与 `outputs/`，无硬编码绝对路径与密钥泄露。

---

## [2026-11-20] Phase 0 - 隐私基线加固、项目配置骨架与日志规范初始化
- **提交类型**：`chore`
- **变更背景**：重置仓库后从零建立面向 GitHub 开源/托管安全的工程基线，配置 `uv` 依赖分组与每次提交强制记录修改日志的防线。
- **主要修改**：
  - 完善 `.gitignore`，隔离 `.env`、`.vscode/`、`data/private/`、`storage/`、`outputs/` 等敏感与运行产物目录。
  - 脱敏 `pyproject.toml`（移除个人邮箱），配置核心依赖与 `[viz]` / `[dev]` 可选依赖分组。
  - 新增脱敏环境变量模板 `.env.example`（兼容云端 API、本机 Ollama/vLLM OpenAI 兼容接口及离线 Mock 模式）与非敏感运行配置 `configs/default.yaml`。
  - 新增 `scripts/privacy_check.py` 隐私与提交规范检查脚本，并在本地配置 Git `pre-commit` 钩子自动拦截隐私泄露与遗漏修改日志的提交。
  - 初始化项目文档 `README.md` 与 `CHANGELOG.md`。
- **验证方式**：执行 `uv sync` 锁文件同步与 `scripts/privacy_check.py --staged --require-changelog` 校验通过。
- **隐私与迁移自检**：已确认本地 Git 作者改用 GitHub noreply 邮箱，无敏感密钥、无本机绝对路径、无私人邮箱进入仓库。

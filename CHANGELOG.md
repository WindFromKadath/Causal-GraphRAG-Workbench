# 修改日志 (CHANGELOG)

本文件遵循“**每次 Git 提交同步记录一次修改日志**”的项目管理规范，用于追溯每次架构演进、功能实现、核验结果与隐私安全状态。

> 隐私约束：日志中禁止记录任何 API Key、本机绝对路径、个人邮箱或私有业务数据。

---

## [2026-11-26] Stage 4.1 Enterprise Storage Foundation & Embedding Guardrails - PG16+pgvector `halfvec(1024)` + Neo4j 5.26 + Redis 7 双存储底座与 `qwen3-embedding:0.6b` CPU 主跑
- **提交类型**：`feat`
- **变更背景**：完成 Stage 4 企业级技术栈重构总规划及第一个完整可验收子阶段 **Stage 4.1**：在保留离线单测 `LocalVectorStore/NetworkXGraphStore` 零依赖替身的前提下，完整落地 Docker Compose 企业基础设施、`qwen3-embedding:0.6b` Fix A (`num_ctx=2048`) + Fix B (`num_gpu=0` CPU 主跑) 双防呆护栏、PostgreSQL 16 + `pgvector 0.8` 混合向量/元数据/A-MEM 存储与 Neo4j 5.26 图谱/超边/ACL 分区社区存储。
- **主要修改**：
  - 架构规划与嵌入护栏：新增 `docs/stage4_enterprise_refactor_plan.md`；升级 `src/llm_graphrag_kg/providers/embedding/ollama_native.py`、`config.py`、`.env.example` 与 `tests/test_stage3_1_retrieval.py`，默认切换 `qwen3-embedding:0.6b` 并默认启用 Fix A（`num_ctx=2048`，将内存由 `4.6 GB` 压至 `985 MB`）+ Fix B（`num_gpu=0` 纯 CPU 主跑，0 MiB 显存争抢，支持可选 MRL 弹性降维与一键切 GPU）。
  - 企业基础设施与依赖：新增 `docker-compose.yml`（PG16 `25432`、Neo4j 5.26 `27474/27687`、Redis 7 `26379`，与现有容器零冲突）、`deploy/postgres/init.sql`（`index_registry`、`documents`、`parent_chunks`、`child_chunks` `halfvec(1024)` HNSW + GIN、`verified_memories`、`audit_logs`、`request_traces`）并在 `pyproject.toml` 新增 `[enterprise]` 可选依赖组（`psycopg[binary]`、`pgvector`、`neo4j`、`redis`）。
  - 企业存储适配器与迁移链路：新增 `src/llm_graphrag_kg/providers/storage/pgvector_store.py`（`PgVectorStore` + `PgVerifiedMemoryStore`）、`src/llm_graphrag_kg/providers/storage/neo4j_store.py`（`Neo4jGraphStore`，含 `:HyperEdge` 中介节点、ACL 预过滤 Cypher 与 `acl_partition` 社区隔离）及 `src/llm_graphrag_kg/pipelines/migrate_enterprise.py`；接入 `providers/factory.py`、`pipelines/workflow.py`、`cli.py`（新增 `check-infra`、`migrate-enterprise` 与 `query/shacl-check --vector-backend/--graph-backend/--tenant/--acl`）并补充离线单元测试 `tests/test_stage4_1_enterprise_storage.py`。
  - 同步更新 `docs/evolution_and_papers_guide.md`、`docs/stage4_enterprise_refactor_plan.md` 与 `README.md`。
- **验证方式**：
  - 静态与离线测试：`ruff check src tests scripts` 0 告警，`pytest -q` **53/53** 全部通过，`scripts/privacy_check.py` 扫描通过（98 文件）。
  - 真机 Docker 验收：`check-infra` 三服务全绿（`PostgreSQL 16.16 + pgvector 0.8.7`、`Neo4j 5.26.32`、`Redis 7.4.12`）；`EXPLAIN (ANALYZE, BUFFERS)` 实证命中 `Index Scan using idx_child_chunks_hnsw_cosine`（`0.163 ms`）；`index_registry` 实证拦截模型/维度错配；10 面多租户 & RBAC/ACL 隔离测试实现 **`0` 越权泄漏** + 软删除审计生效。
  - 双图谱实测迁移与问答：工业 OREDA `fmea_oreda`（87 切片 / 87 实体 / 1,013 关系）迁入 PG16+Neo4j 达 **`SHACL-lite 100.0% PASS`**；`fault_diag`（15 切片 / 51 实体 / 94 关系 / 7 超边 / 1 记忆）在 PG16 + Neo4j + `qwen3-embedding:0.6b` (CPU) + `qwen3.5:9b` (GPU) 端到端实跑 `ppr + auto -> tot` 诊断，命中 4 条 Neo4j 超边 + 1 条 PG A-MEM，达到 **`100.0%` Grounding（0 幻觉引用）**。
- **隐私与迁移自检**：Docker 数据卷隔离在命名卷与 `.gitignore` 中，`.env.example` 与默认本地开发配置无任何真实密钥或本机绝对路径泄露。

---

## [2026-11-26] Stage 3.4 HyperGraph Multi-Condition HyperEdges, Gated A-MEM Verified Write-Back & Workbench Tab 7
- **提交类型**：`feat`
- **变更背景**：完成第三阶段收官目标 Stage 3.4，引入 N 元多条件联合超边建模（HyperGraphRAG）与经四重防投毒门禁审核的自演化情景记忆写回（A-MEM），并在学习手册中补充面向企业级技术栈重构的 Protocol → Production 迁移蓝图。
- **主要修改**：
  - 扩展 `src/llm_graphrag_kg/core/schema.py`（`HyperEdge`、`VerifiedMemoryRecord`、`GraphAuditStats.hyperedge_count`）、`config.py`（`MemoryConfig`）与 `providers/storage/networkx_store.py`（`upsert_hyperedge`、`get_all_hyperedges`、`graph.json` 向后兼容持久化）。
  - 新增 `src/llm_graphrag_kg/pipelines/hypergraph.py`（`create_hyperedge`、`mine_joint_hyperedges`、`match_hyperedges`、`format_hyperedge_context`）：自动挖掘共源多条件联合推理规则（`{Cond_A + Cond_B + ...} ===[JOINT_INDICATES]===> Target`），并在检索时精准提升联合命中目标实体与原文证据。
  - 新增 `src/llm_graphrag_kg/pipelines/memory_store.py`（`VerifiedMemoryStore`、`check_memory_conflict`、`WriteBackGateDecision`）：强制执行 `GATE_1_GROUNDING (100%)` → `GATE_2_SUBSTANCE/CONFIDENCE` → `GATE_3_CONFLICT_NLI`（极性矛盾与同条件互斥根因拦截）→ `GATE_4_AUDIT_PERSIST`（带 UTC 时间戳 `valid_from` 与 `provenance_run_id` 持久化至 `verified_memories.jsonl` 并自动晋升为 `VERIFIED_JOINT_DIAGNOSIS` 超边）。
  - 全链路接入 `pipelines/retriever.py`、`pipelines/workflow.py`、`viz/audit_report.py`、`cli.py`（新增 `mine-hyperedges`、`memory-list` 与 `query --writeback`）及 `viz/app.py`（升级至 7 标签页，新增 Tab 7「🧬 超边与自演化记忆」与问答页写回开关/命中提示）；新增单元测试 `tests/test_stage3_4_hypergraph_memory.py` 并更新 `tests/test_workbench.py`、`docs/evolution_and_papers_guide.md`（含企业级技术栈重构迁移地图）、`docs/stage3_nextgen_graphrag_plan.md` 与 `README.md`。
- **验证方式**：
  - `ruff check src tests scripts` 0 告警，`pytest -q` **45/45** 全部通过（含 7 标签页无头 `AppTest`），`scripts/privacy_check.py` 扫描通过。
  - 真机验证（`qwen3.5:9b`）：工业故障诊断图谱自动挖掘 `6` 条多条件超边（`conf` 最高 `0.98`），FailureSensorIQ 挖掘 `64` 条多传感器联合超边；`tot_dfs --writeback` 经 4 重门禁放行写入 `mem_063584d27c0b` 并自动晋升超边，二次查询自动触发 `HyperGraph Matches (4 hits)` + `Recalled A-MEM` 且达到 4/4 步 `100.0%` Grounding。
- **隐私与迁移自检**：生成的 `verified_memories.jsonl` 与超边持久化均隔离在 `.gitignore` 屏蔽的 `/storage/` 目录，无密钥、本机绝对路径或私有数据入库。

---

## [2026-11-26] Stage 3.3 Official 4D LLM-as-a-Judge, Checkpointed Semantic OpenIE & Industrial OREDA RDF/Turtle Importer
- **提交类型**：`feat`
- **变更背景**：完成 `docs/stage3_nextgen_graphrag_plan.md` 规划的 Stage 3.3 三大工程支柱：建立无词面失真与拒答伪装的官方口径四维语义裁判体系、用断点式定向语义 OpenIE 取代无向共现边、并打通工业 OREDA `FMEA_KG.ttl` 标准 RDF/Turtle 知识图谱零依赖导入与 `SHACL-lite` 全图质检。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/eval/__init__.py` 与 `src/llm_graphrag_kg/eval/llm_judge.py`（`LLMJudgeEvaluator`、`detect_refusal`、`fbeta_score`、`judge_grb_checkpoints`）：零 LangChain 依赖，对齐 `GraphRAG-Bench` 官方指标公式输出 **Answer Accuracy**（`0.75 * Factuality_F1 + 0.25 * CosineSim`）、**Faithfulness**、**Semantic Coverage** 与独立 **Refusal Rate**，支持任意已有评测断点 JSONL 的可恢复评分，并新增 CLI 命令 `judge-grb`。
  - 升级 `src/llm_graphrag_kg/pipelines/extraction.py` 新增 `SemanticOpenIEExtractor` 与 `normalize_openie_predicate`（基于 `LLMHarness` 结构化校验 + JSONL 切片级断点续抽 + 禁用 `CO_OCCURS_WITH`），接入 `adapters/graphrag_bench.py`（`use_openie` / `openie_max_chunks` 及 `GRBQuestionOutcome.retrieved_context`）与 CLI `ingest-grb --openie`。
  - 新增工业 RDF/Turtle 导入器 `src/llm_graphrag_kg/adapters/rdf_fmea.py`（零外部依赖 `parse_turtle_triples` + `ingest_fmea_rdf_ttl`，内置同名实体对齐、自环过滤、OREDA 语义类型映射、带源码行号的溯源文本卡片与 HippoRAG 2 同义边建索引）、`configs/fmea_kg_rdf.yaml`、CLI 命令 `ingest-rdf`、`viz/html_exporter.py` OREDA 配色与 `cli.py query` 按 `manifest.json` 自动对齐 Embedding 维度。
  - 扩展 `providers/llm/mock.py` 离线裁判/OpenIE 确定性桩实现，新增 `tests/test_stage3_3_judge_openie_rdf.py`，并同步更新 `docs/evolution_and_papers_guide.md`、`docs/stage3_nextgen_graphrag_plan.md` 与 `README.md`。
- **验证方式**：
  - `ruff check src tests scripts` 0 告警，`pytest -q` **42/42** 全部通过，`scripts/privacy_check.py` 扫描通过。
  - **官方四维 LLM-as-a-Judge 双域实测（`qwen3.5:9b` + `nomic-embed-text`）**：
    - Medical (10Q)：`naive_rag` 出现 `10.0%` 拒答（摘要题 `20.0%`），Accuracy `74.2%` / Faithfulness `32.2%`；图谱范式全线 **0.0% 拒答**，`ppr_auto` 反超至 **`78.9%`** (+4.7pp)，**`ppr_cot_iter` 达 `82.2%` Accuracy (+8.0pp，Complex Reasoning `93.9%`) / `52.3%` Faithfulness (+20.1pp) / `75.0%` Sem. Coverage (+9.2pp)**；
    - Novel (10Q)：**`ppr_multi_branch`** 全面夺冠，Accuracy **`78.1%`** (+4.8pp vs `naive_rag` `73.3%`)、Faithfulness **`33.9%`** (+14.9pp)、Sem. Coverage **`66.8%`** (+2.6pp)。
  - **语义 OpenIE 实测**：Medical 12 切片抽取 `42` 实体 / `49` 定向三元组，**语义谓词纯度 `100.0%`**（零 `CO_OCCURS_WITH`）。
  - **工业 `FMEA_KG.ttl` 实测**：解析 `1,192` 条三元组 / `94` 节点 → 规范化 `87` 实体 / `1,013` 关系 / `5` 社区（合并 `7` 重复、滤 `2` 自环），**`SHACL-lite` 100.0% PASS（0 Error / 0 Warning）**，`SHH` 故障码真机实问 4/4 步 **100.0% Grounding**。
- **隐私与迁移自检**：导入存储（`storage/fmea_kg_rdf`、`storage/graphrag_bench_medical_openie_demo`）与四维裁判报告均严格隔离在 `.gitignore` 屏蔽目录，无密钥、本机绝对路径或私有数据入库。

---

## [2026-11-25] Paper-Faithful Graph-CoT v2 & Multi-Depth Graph-ToT v2 (BFS/DFS) + Dual-Model Tiering (`4b` vs `9b` Ablation)
- **提交类型**：`feat`
- **变更背景**：对标原论文（Graph-CoT ACL 2024 / PoG AAAI 2025 / ToT NeurIPS 2023 / ToG & ToG-2 ICLR 2024-2025），将初版简化基线（单轮 CoT 与 Depth-1 扁平 ToT）同步升级为论文完整版迭代工具循环与多深度图边搜索（支持 BFS 束搜索与 DFS 真实回溯），并建立“轻量中间层 + 主力综合层”双模型分层架构，实证对比 `qwen3.5:4b` 与 `qwen3.5:9b` 作为中间层的质量与 8 GB 显存延迟边界。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/reasoning/cot_iterative.py`（`IterativeGraphCoTEngine` / `cot_iter`）：实现 3 轮真实 `Thought -> RetrieveNode / InspectEntity / ExpandNeighbors / FetchChunk / Finish -> Observation` KG 工具循环、PoG `[SELF-CORRECT]` 纠错反馈、全量原文保真综合与 `GroundingAudit`。
  - 新增 `src/llm_graphrag_kg/reasoning/tot_deep.py`（`DeepGraphToTEngine` / `tot_bfs` & `tot_dfs`）：实现 Depth `1 -> 2` 真实图边前沿扩展、ToT 离散状态评估（`sure` 早停 / `likely` 扩展 / `impossible` 剪枝 + L2 折叠）、BFS Beam Search 与 DFS 死胡同真实回溯（`backtracked=True`），并在最终综合上下文显式提供 `[ent_...]` 实体名录确保高 Grounding。
  - 双模型分层与全链路接入：在 `config.py` 新增 `llm_fast_model` / `llm_fast_num_ctx=8192`，`providers/factory.py` 新增 `create_fast_llm_client`，扩展 `reasoning/schema.py`、`pipelines/retriever.py`、`pipelines/workflow.py`、`adapters/graphrag_bench.py`（新增 `ppr_cot_iter` / `ppr_tot_bfs` / `ppr_tot_dfs` 与 `planner_seconds`/`load_seconds`/`max_depth_reached`/`backtrack_count` 遥测）、`cli.py`（`--fast-model` 与 v2 `compaction` 兼容显示）、`viz/app.py` 及 `viz/tot_tree_exporter.py`（展示 `D1/D2` 深度、`verdict` 与 `↩ 回溯` 标记）。
  - 新增单元测试 `tests/test_paper_faithful_cot_tot.py`，同步将 v2 基线升级与 `4b vs 9b` 消融实测结论回填至 `docs/evolution_and_papers_guide.md`。
- **验证方式**：
  - `ruff check src tests scripts` 0 告警，`pytest -q` **38/38** 全部通过，`scripts/privacy_check.py` 扫描通过；工业故障诊断 `tot_dfs` 真机查询与交互树导出验证 100% Grounding。
  - `GraphRAG-Bench Medical` 同源 10 题分层对照（5 `Complex Reasoning` + 5 `Contextual Summarize`，`seed=42`，`nomic-embed-text`）：
    - **v1 → v2 跃升（`9b` 统一）**：`cot_iter` Token-F1 `36.0% → 44.7%`（+8.7pp，超 `naive_rag` +7.4pp）、ROUGE-L `34.4%`、Grounding `94.5%`；`tot_bfs` / `tot_dfs` 检索召回飙至 **`86.7%`**（#1），`tot_bfs` Grounding **`100.0%`**（F1 `39.1%` / `20.97s`），`tot_dfs` F1 达 **`41.3%`**（+9.9pp vs v1.5 D1 ToT，8/10 题触发真实回溯平均 `1.4` 次）。
    - **`4b` vs `9b` 中间层消融**：`Complex Reasoning` 上 `9b` 中间层跨三引擎稳赢 `+2.4 ~ +3.6pp` F1；`tot_bfs` 离散打分场景 `4b` 反超 `9b`（F1 `39.5%` vs `39.1%`，ROUGE-L `29.3%` vs `27.4%`），`cot_iter (4b→9b)` 拿下全榜最高回答覆盖率 `72.4%`；纯算力上 `4b` 快 **`14.7%–24.1%`**（Net 降 `3.1s–6.7s`），但 8 GB 单卡在线交互每题换模税达 **`+7.09s–11.16s`**，实证确立“8 GB 单卡在线选单 `9b`、≥12 GB 或两阶段批处理选 `4b→9b`”的工程决策准则。
- **隐私与迁移自检**：消融断点与评测报告隔离在 `.gitignore` 屏蔽的 `/outputs/`，无密钥、本机绝对路径或私有数据入库。

---

## [2026-11-25] Stage 3.2 Adaptive Routing & Multi-Branch Synthesis - 自适应任务路由、非剪枝多分支聚合、ToG-2 截断修复与动态早停
- **提交类型**：`feat`
- **变更背景**：彻底根治 Stage 2/3.1 在 `GraphRAG-Bench` 上暴露的 `ppr_tot`“检索召回第一（79.8%/72.4%）但回答覆盖率暴跌（27.1%/32.1%）”剪刀差，落地 `docs/stage3_nextgen_graphrag_plan.md` 规划的 Stage 3.2。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/reasoning/router.py`（`route_query`、`compute_first_hop_coverage`、`RouteDecision`）：零额外 LLM 开销（`< 1ms`）实现三路自适应分流（`exclusive_diagnosis → tot`、`multi_facet_synthesis → multi_branch`、`fact_direct` 或首跳覆盖率 `>= 0.80` 动态早停 `→ cot`）。
  - 升级 `src/llm_graphrag_kg/reasoning/tot.py`：移除 Phase 4 破坏性的 `[:220]` 字符硬截断并补充高分 PPR/RRF 全局原文切片（ToG-2 图文双向无损保留）；新增 `synthesis_mode="additive"`（`multi_branch` 非剪枝多分支并集聚合，仅需 2 次 LLM 调用，保留全部高分互补分支与去重完整原文切片并集）。
  - 全链路接入 `config.py`（`ReasoningConfig` 早停与聚合阈值）、`reasoning/schema.py`、`providers/llm/mock.py`、`pipelines/retriever.py`、`pipelines/workflow.py`、`adapters/graphrag_bench.py`（新增 `ppr_multi_branch` 与 `ppr_auto` 评测模式及路由遥测）、`cli.py` 与 `viz/app.py`（新增 `auto` / `multi_branch` 选择与自适应路由卡片）。
  - 新增单元测试 `tests/test_stage3_2_router_multibranch.py`，同步回填 Stage 3.2 实测验收结论与深度学习卡片至 `docs/stage3_nextgen_graphrag_plan.md` 与 `docs/evolution_and_papers_guide.md`。
- **验证方式**：
  - `ruff check src tests scripts` 0 告警，`pytest -q` **35/35** 全部通过，`scripts/privacy_check.py` 扫描通过。
  - `GraphRAG-Bench` 同源双域 50 题（`seed=42`，`qwen3.5:9b` + `nomic-embed-text`）全面通过 Stage 3.2 验收门禁：
    - **Medical (25Q)**：`ppr_auto` 全量回答覆盖率 **`68.2%`** 登顶第 1；`Contextual Summarize` 覆盖率从 `18.7%` 暴涨至 `60.5%–61.9%`（`multi_branch` F1 **`39.9%`** 第 1）；`Complex Reasoning` 修复版 `ppr_tot` F1 达 **`41.9%`**（+5.5pp vs `naive_rag`）；`multi_branch` Grounding **100.0%**，耗时 `18.84s`（较 S2 `51.8s` 降 **-63.6%**）。
    - **Novel (25Q)**：`ppr_auto` 全量四维指标（F1 `26.4%` / EvCov `63.0%` / RetRec `72.4%`）全面反超 Dense `naive_rag`；`Contextual Summarize` F1 **`31.1%`** / EvCov **`59.2%`** 双超 `naive_rag`；`Complex Reasoning` 修复版 `ppr_tot` F1 达 **`29.6%`**（+5.4pp vs `naive_rag`）；`multi_branch` Grounding **100.0%**，耗时 `19.02s`（较 S2 `52.4s` 降 **-63.7%**）。
- **隐私与迁移自检**：评测断点与报告严格隔离在 `.gitignore` 屏蔽的 `/outputs/`，无密钥、本机绝对路径或私有数据入库。

---

## [2026-11-24] Visual Showcase - 新增「基于思维树的透明推理与因果链条解析」Node→Edge→Node 逐跳读取与收敛树形交互视图
- **提交类型**：`feat`
- **变更背景**：响应展示与汇报需求，在原有横向 ToT 对比卡片之外，补齐真正的 **Node → Edge → Node 逐跳读取顺序回放（从向量化响应到最终仅保留用到的一片核心节点）+ 思维树分支拓扑 + 因果链条解析 + 原文切片溯源抽屉** 一体化离线展示视图。
- **主要修改**：
  - 扩展 `src/llm_graphrag_kg/reasoning/schema.py`（`ThoughtBranch.explored_entity_ids`）、`reasoning/tot.py` 与 `pipelines/retriever.py`（`metadata["traversal_trace"]`）：零额外 LLM 开销记录真实 Hop 0 向量/三元组响应种子、PPR 激活分及各 ToT 分支探索节点集合。
  - 新增 `src/llm_graphrag_kg/viz/tot_tree_exporter.py`（`build_tot_tree_payload` / `render_tot_tree_html` / `export_tot_tree_html`）：零外部 CDN 依赖、100% 离线单文件 HTML，提供 **5 帧可控回放画布**（`1. 向量化响应 Hop 0` → `2. 一跳边扩散 Hop 1` → `3. 多跳深层遍历 Hop 2+` → `4. Compact & ToT 剪枝` → `5. 最终核心片收敛` + `▶ 自动播放` + `仅保留最终用到的一片核心 Node` 开关）、点击任意 Node 检视关联边与原文切片、下方联动思维树分支拓扑与 L2 折叠反证，支持深浅色切换与打印/PDF 导出。
  - 集成至 `src/llm_graphrag_kg/viz/app.py`（「🔍 问答与思维回放」页顶部原生内嵌视图并提供一键下载 `.html`）与 `src/llm_graphrag_kg/cli.py`（`query` 命令新增 `--export-tot-html` 参数）；同时新增 `configs/graphrag_bench_medical.yaml` 与 `configs/graphrag_bench_novel.yaml`，在工作台侧边栏完整编入 `GraphRAG-Bench Medical`（Stage 3.1 `nomic-embed-text` + 236 条 `SYNONYM_OF`）与 `GraphRAG-Bench Novel`（31 条 `SYNONYM_OF`）两套高难数据集，补齐 `html_exporter.py` 医学/叙事节点配色与**按索引自动对齐 Embedding Provider（256d `local_hash` ↔ 768d `ollama_native`）**防维度冲突机制，同步更新 `README.md`。
  - 在 `tests/test_workbench.py` 新增 `test_tot_tree_exporter_builds_interactive_html_for_tot_and_cot` 与 `test_workbench_discovers_medical_and_novel_configs_and_auto_detects_embedder` 单元测试覆盖 Node→Edge→Node 跳数轨迹、最终核心片收敛统计、双域配置发现与自动嵌入对齐。
- **验证方式**：`ruff check src tests scripts` 0 告警，`pytest -v` 全部通过，并通过本机 `qwen3.5:9b` 工业离心泵并发故障诊断实跑导出 `outputs/fault_diagnosis/tot_reasoning_tree.html` 验证。
- **隐私与迁移自检**：生成的 HTML 演示产物隔离在 `.gitignore` 屏蔽的 `/outputs/` 目录，无密钥、本机绝对路径或私有数据入库。

---

## [2026-11-24] Stage 3.1 Retrieval Upgrade - 真实稠密嵌入（Ollama 原生 + SQLite 缓存）、双语 Okapi BM25 + Dense RRF 与 HippoRAG 2（`SYNONYM_OF` + Query-to-Triple）
- **提交类型**：`feat`
- **变更背景**：直击 Stage 2/3.0 在千级切片开放语料上 `local_hash` 假向量锁死检索召回率（Medical `47.7%–60.3%`、Novel `29.5%–46.8%`）的核心瓶颈，完整落地 `docs/stage3_nextgen_graphrag_plan.md` 规划的 Stage 3.1 下一代检索底座。
- **主要修改**：
  - 新增 `src/llm_graphrag_kg/providers/embedding/ollama_native.py`（`OllamaNativeEmbeddingClient` 与 `EmbeddingDiskCache`）并升级 `openai_compat.py`、`config.py`、`factory.py` 与 `.env.example`：支持 Ollama 原生 `/api/embed` 分批编码、超长截断、`num_gpu` 控制与 SHA-256 SQLite 磁盘缓存；CLI 新增 `check-embed` 自检及 `ingest-grb` / `eval-grb` 的 `--embedder` / `--embedding-model` 参数。
  - 升级 `src/llm_graphrag_kg/providers/storage/local_vector.py`：零外部依赖新增中英双语 `tokenize_bm25`、Okapi BM25 倒排索引（`search_bm25`）、加权倒数排名融合（`search_hybrid` RRF）及向量维度一致性守卫；同步修正 `.gitignore` 根目录锚定（`/storage/`、`/outputs/`、`/data/private/`），确保源码目录 `src/llm_graphrag_kg/providers/storage/` 正常纳入版本追踪。
  - 新增 `src/llm_graphrag_kg/pipelines/hipporag.py` 并升级 `pipelines/retriever.py`、`pipelines/workflow.py`、`adapters/graphrag_bench.py` 与 `configs/graphrag_bench.yaml`：实现实体余弦相似度 `SYNONYM_OF` 自动连边（保留原文溯源指针满足 SHACL-lite）、持久化 `SemanticGraphIndex`（`semantic_index.npz` + `semantic_index_meta.json`）、Dense Entity Linking、**Query-to-Triple** 关系锚点以及基于候选文档池（Passage-Pool）与特异性加权边的 HippoRAG 2 异构 PPR。
  - 新增单元测试 `tests/test_stage3_1_retrieval.py` 覆盖 BM25/RRF、Ollama 原生嵌入批量缓存与工厂路由、`SYNONYM_OF` / Query-to-Triple 持久化及 PPR 端到端链路；新增持续维护学习参考文档 `docs/evolution_and_papers_guide.md`（全阶段架构演进总览、论文范式对照、升级动因与实测收益）并更新 `README.md` 与 `docs/stage3_nextgen_graphrag_plan.md`。
- **验证方式**：
  - `ruff check src tests scripts` 0 告警，`pytest -q` **31/31** 全部通过，`llm-graphrag-kg check-embed --embedder ollama_native --model nomic-embed-text` 实测连通（`dim=768`）。
  - 在 `GraphRAG-Bench` 同源双域 50 题（`seed=42`，`qwen3.5:9b` + `nomic-embed-text`）上全面打穿 Stage 3.1 验收门禁：
    - **Medical (25Q，门槛 ≥75%)**：`Ret EvRec` 严格单调跃升至 `70.5%` (`naive_rag`) → `72.6%` (`hybrid_direct`) → **`77.5%`** (`ppr_cot`) → **`79.8%`** (`ppr_tot`，Grounding `100.0%`)，新增 `236` 条 `SYNONYM_OF` 边；
    - **Novel (25Q，门槛 ≥65%)**：`Ret EvRec` 暴涨至 `65.9%` → `67.9%` → **`71.9%`** (`ppr_cot`) → **`72.4%`** (`ppr_tot`)，其中 `ppr_cot` 以 `Token-F1 28.1% / ROUGE-L 20.7% / Grounding 100.0%` 登顶，`hybrid_direct` 回答证据覆盖率 `68.2%` 登顶。
- **隐私与迁移自检**：`nomic-embed-text` 索引缓存（`.embed_cache`）、双域存储（`storage/graphrag_bench_*_s31`）与评测报告（`outputs/`）均严格隔离在 `.gitignore` 目录，无密钥、本机绝对路径或私有数据入库。

---

## [2026-11-23] Stage 3.0 Base Model - 确立第三阶段本地基础模型基线（`qwen3.5:9b` + `num_gpu` 控制）、多上下文显存压测与全套任务基准跨模型回归归档
- **提交类型**：`feat`
- **变更背景**：在进入 Stage 3.1–3.4 开发前，针对本机 8 GB 显存环境完成 `gemma4:12b`、`qwen3.5:9b`、`qwen3.5:4b` 三款本地模型的 4k→128k 多上下文吞吐与物理显存基准压测，定位 Ollama 默认调度保守卸载瓶颈与 WDDM 换页边界，将实测最优配置固化为第三阶段底层基础模型，并完成全套 4 阶段端到端任务基准重跑对比。
- **主要修改**：
  - 升级 `src/llm_graphrag_kg/config.py`、`providers/llm/ollama_native.py`、`providers/factory.py` 与 `cli.py`：默认基础模型升级为 `qwen3.5:9b`（默认超时 `180s`），新增 `GRAPHRAG_LLM_NUM_GPU`（`EnvSettings.llm_num_gpu`）透传至 Ollama `/api/chat` `options.num_gpu`，并在 `last_telemetry` 补充 `load_seconds` 与 `prefill_tok_per_s` 遥测；`stress-fsiq` 默认 `--num-ctx` 从 `32768` 对齐至 `16384`。
  - 更新 `.env.example`、`docs/engineering_deployment.md`、`docs/stage3_nextgen_graphrag_plan.md` 与 `README.md`，新增脱敏选型与跨模型回归报告 `docs/benchmark_stage3_base_model.md`（确立 `qwen3.5:9b` + `num_gpu=99` 为主力基础模型、`qwen3.5:4b` 为高吞吐辅助模型，明确 8 GB 显存下 `gemma4:12b` 禁用 `num_gpu=99` 与 Embedding 共存约束，并归档全套端到端任务跨模型对照结果）。
  - 更新 `.gitignore` 与 `scripts/privacy_check.py`：新增 `local_bench/` 本地硬件压测隔离目录及隐私门禁校验；新增 `tests/test_ollama_native.py` 单元测试覆盖 `num_gpu` 透传、工厂路由与隐私门禁。
- **验证方式**：
  - 通过 `ruff check`（0 告警）、`pytest -v`（27/27 全部通过）、`llm-graphrag-kg check-llm` 实测联通 `qwen3.5:9b`（`think=False, num_ctx=16384, num_gpu=99`）以及 `scripts/privacy_check.py --staged --require-changelog` 扫描。
  - 完成 4 阶段端到端跨模型回归（`seed=42`，总耗时 `48.9 min`）：`GraphRAG-Bench` 双域 Plan B 下 `ppr_cot` 全面反超 `naive_rag` 登顶第一（Medical Token-F1 `30.0%` / ROUGE-L `24.0%` / Grounding `100.0%`，Novel Token-F1 `25.5%` / ROUGE-L `20.7%` / Grounding `97.0%`），`ppr_tot` 提速 **3.19×–3.47×** 降至 `15.1s–16.3s`；`FailureSensorIQ` 4 套件 × 100 题实现 `L1/L3/L4` 100.0% 满分（`0.41s–0.49s`/题、0 修复/0 空回复）与 `L2` 82.0%（精准定位 15/18 错题集中于 `irrelevant_sensors_for_failure_mode`）；工业实时 `PPR + ToT` 诊断 `GroundingAudit = 100.0%`（Compact `-83.8%`）。
- **隐私与迁移自检**：本地压测脚本、原始 JSONL 与基准评测产物严格隔离在 `.gitignore` 屏蔽的 `local_bench/`、`outputs/` 与 `storage/`；无密钥、本机绝对路径或私有数据入库。

---

## [2026-11-23] Stage 3 Plan - 归档第三阶段演进方案（稠密混合检索、自适应路由、LLM-Judge、HyperGraph与自演化记忆）
- **提交类型**：`docs`
- **变更背景**：基于第二阶段 `GraphRAG-Bench` 双域（Medical + Novel）实测定位的 4 大瓶颈（`local_hash` 召回上限、无类型共现边噪声、ToT 单胜剪枝伤及多面摘要、`Token-F1` 拒答失真）及 2025–2026 前沿范式调研，正式确立并归档第三阶段（Stage 3.1 ~ 3.4）工程实施计划。
- **主要修改**：
  - 新增 `docs/stage3_nextgen_graphrag_plan.md`：规划 Stage 3.1（本机稠密嵌入 + BM25 RRF + HippoRAG 2 同义词边与 Query-to-Triple）、Stage 3.2（自适应任务路由 + 非剪枝多分支聚合 + 动态早停）、Stage 3.3（对齐 `GraphRAG-Bench` 官方口径的 LLM-as-a-Judge 评测器 + 语义 OpenIE 与工业 `FMEA_KG.ttl` RDF 导入）以及 Stage 3.4（HyperGraph 多条件超边 + 验证后自演化记忆写回 + 工作台联动），并明确对照 Stage 2 Plan B 同批 50 题的量化验收门槛。
  - 更新 `README.md` 文档目录索引。
- **验证方式**：文档结构与 Mermaid 架构图核验完成，通过 `scripts/privacy_check.py --staged --require-changelog`。
- **隐私与迁移自检**：无密钥、本机绝对路径或私有数据入库。

---

## [2026-11-22] Stage 2 Closeout - GraphRAG-Bench 双域高难基准归档、第二阶段收尾与云端同步
- **提交类型**：`docs`
- **变更背景**：完成 `GraphRAG-Bench` 双域（`Medical` 全量 1.05 MB + `Novel` 2 部长篇，共 50 题 / 300 次本地 `gemma4:12b` 推理）四范式深度基准评测，正式归档第二阶段收尾报告并准备将 Stage 2.4 至收尾的全部提交同步至 GitHub。
- **主要修改**：
  - 新增 `docs/benchmark_stage2_hard_datasets.md`：记录 Medical（1,528 切片 / 450 实体 / 6,522 边）与 Novel（702 切片 / 450 实体 / 2,683 边）四范式全量指标（`Ret EvRec` 严格单调跃升：Medical `47.7% → 60.9%`、Novel `29.5% → 47.1%`；`Ans EvCov` 由 `hybrid_direct` 与 `ppr_cot` 全面领跑，Grounding Score 达 `94%~100%`），沉淀“检索-生成剪刀差”三重法证归因（短拒答 Token-F1 陷阱、ToT 单胜剪枝对多面摘要的影响、`local_hash` 向量上限）及 Stage 3 演进输入。
  - 更新 `README.md` 文档目录索引，并将 `src/llm_graphrag_kg/adapters/graphrag_bench.py` 的分层均衡抽样与单题异常隔离加固一并纳入提交。
- **验证方式**：通过 `ruff check`（0 告警）、`pytest -q`（23/23 全部通过）与 `scripts/privacy_check.py --staged --require-changelog`。
- **隐私与迁移自检**：所有外部数据集、图谱存储与运行断点均严格隔离在 `.gitignore` 目录，无隐私泄露风险。

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

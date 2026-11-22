# LLM-GraphRAG-KG 工程故障诊断与排障手册

本手册采用“**故障现象 → 根因分析 → 诊断核验步骤 → 修复方案**”结构，系统整理环境部署、模型推理、知识图谱构建质量、GraphRAG 检索效果、可视化性能与隐私安全 6 大类典型故障的排查流程。

---

## 快速诊断总览表

| 故障编号 | 所属模块 | 典型现象 | 首选诊断命令 / 工具 |
|---|---|---|---|
| **FD-01** | 模型连接 | `ConnectionError` / `404 model not found` | `llm-graphrag-kg check-llm` |
| **FD-02** | 推理超时 | `APITimeoutError` / 建图卡住无响应 | 调大 `GRAPHRAG_LLM_TIMEOUT` + 查看 GPU/显存占用 |
| **FD-03** | 图谱抽取 | 实体数为 0 或大量切片未抽出三元组 | 查看模型 JSON 输出格式 + 调低 `temperature` |
| **FD-04** | 图谱碎片化 | 孤立实体多、连通分量过多、中英文分裂 | `llm-graphrag-kg audit` / 工作台「仅看孤立实体」 |
| **FD-05** | 检索失准 | 种子实体为空、子图未展开、答非所问 | 工作台「问答与子图追踪」检查 `seed_entity_ids` |
| **FD-06** | 向量维度冲突 | `ValueError: shapes not aligned` | 切换 Embedding 模型后未重建 `storage/` |
| **FD-07** | 可视化卡顿 | 节点 > 1000 时 PyVis 浏览器卡死 | 关闭物理布局 / 按社区筛选 / 导出 GraphML 用 Gephi |
| **FD-08** | 隐私与提交拦截 | `pre-commit` 报错阻止 `git commit` | `llm-graphrag-kg privacy-check --staged --require-changelog` |

---

## 1. 环境与模型连接类故障

### FD-01：模型服务连接失败或提示模型不存在
- **典型现象**：
  - `Failed to reach http://127.0.0.1:11434/v1: APIConnectionError`
  - `Connected ... Note: configured model 'xxx' was not in the listed IDs.`
- **根因分析**：
  1. Ollama / vLLM / LM Studio 服务未启动，或监听端口与 `.env` 中 `GRAPHRAG_LLM_BASE_URL` 不一致。
  2. `.env` 中 `GRAPHRAG_LLM_MODEL` 与服务端实际模型 tag 不一致（如误填 `qwen2.5:7b` 而本机仅安装了 `gemma4:12b`）。
  3. 本机代理环境变量（`HTTP_PROXY` / `HTTPS_PROXY`）劫持了 `127.0.0.1` 回环流量。
- **诊断核验**：
  ```bash
  uv run llm-graphrag-kg check-llm
  ollama list
  ```
- **修复方案**：
  - 核对 `ollama list` 输出的精确模型名并写入 `.env` 的 `GRAPHRAG_LLM_MODEL`。
  - 若开启了系统代理，确保设置 `NO_PROXY=127.0.0.1,localhost`。

### FD-02：本地大模型建图超时（`ReadTimeout` / 耗时极长）
- **典型现象**：执行 `build` 时中途抛出超时异常，或单个切片耗时超过 60 秒。
- **根因分析**：
  1. 12B+ 参数模型首次加载到显存（冷启动）耗时较长，或显存不足触发 CPU Offload。
  2. `chunk_size` 设置过大导致单次 Prompt Token 过长，生成长 JSON 时超出默认 60s 超时。
- **修复方案**：
  - 在 `.env` 中增大超时：`GRAPHRAG_LLM_TIMEOUT=180`（或 `300`）。
  - 在 YAML 配置中保持 `chunking.chunk_size` 在 `350~600` 字符之间，避免单切片塞入过多段落。
  - 若批量处理上百篇文档，先关闭 `--summarize-communities`（默认使用零 LLM 开销的抽取式社区摘要），待图谱质量核验通过后再开启 LLM 社区摘要。

---

## 2. 知识图谱质量类故障（核心排障区）

### FD-03：实体/关系抽取结果为空或解析失败
- **典型现象**：`build` 完成后 `Entities=0` 或远低于预期。
- **根因分析**：
  1. 部分本地小参数模型不支持 `response_format={"type": "json_object"}`，或输出了带多余解释/非法逗号的 JSON（虽然框架已内置 fallback 与代码块剥离，但严重畸形 JSON 仍会被安全丢弃为空）。
  2. `temperature` 过高导致模型发散、不遵守 Schema。
- **诊断核验**：
  - 先用 `--provider mock` 跑通同一目录，排除文档读取与编码（UTF-8）问题。
  - 在工作台「📄 原文切片」页核查切片内容是否正常读入。
- **修复方案**：
  - 将 `extraction.temperature` 设为 `0.0 ~ 0.1`。
  - 优先选用指令遵循与 JSON 输出能力较强的 7B~14B+ 指令微调模型。
  - 针对专业领域，使用自定义 YAML（如 `configs/fault_diagnosis.yaml`）明确限定 `entity_types` 列表，减少模型自由发挥。

### FD-04：图谱碎片化、孤立节点过多、同义中英文实体分裂
- **典型现象**：审计报告（`outputs/audit_report.md`）显示连通分量过多、孤立实体占比 > 20%，例如 `知识图谱` 与 `Knowledge Graph`、`轴承外圈剥落` 与 `外圈剥落` 被识别为不同节点。
- **根因分析**：
  1. 不同切片中同一概念使用了中英文混写或简称，仅靠大小写规范化无法合并。
  2. 切片过短（如 `chunk_size < 150`）把主语和宾语切分到不同切片，导致实体出现但关系丢失。
- **诊断核验**：
  - 打开 Streamlit 核验台 → 「🧩 实体与三元组溯源」→ 勾选 **「仅看孤立实体」**，反查其原文切片。
  - 打开「📊 交互图谱」→ 按 `community` 着色，观察是否出现大量单节点/双节点孤岛。
- **修复方案**：
  - 适当增大 `chunk_size`（如 `450~600`）与 `chunk_overlap`（如 `80~100`），保持因果句完整。
  - 在源文档术语编写或抽取 Prompt 中统一命名规范（优先保留统一标准术语，避免同一概念多种别名混用）。
  - 进阶优化（Phase 3 规划）：引入别名词典映射与基于向量相似度的实体对齐消歧（Entity Resolution）。

---

## 3. GraphRAG 检索与问答类故障

### FD-05：种子实体未命中或检索子图为空
- **典型现象**：问答结果中 `Seed Entities: -` 或回答未能利用图谱多跳关系。
- **根因分析**：
  1. 用户提问措辞与图谱实体名称字面差异较大（例如图谱中是 `滚动轴承外圈剥落`，提问写的是“机器异响坏了”）。
  2. 处于 `local` 模式且 `k_hops=0`，未能向邻域扩展。
- **诊断核验**：
  - 在工作台「🔍 问答与子图追踪」查看 **种子实体**、**扩展子图节点** 与 **高亮图谱**。
  - 对比 `local`、`global`、`hybrid` 三种模式的召回差异。
- **修复方案**：
  - 生产问答优先使用 `mode="hybrid"`（结合向量召回自动反推种子实体 + 社区摘要兜底）。
  - 将 `k_hops` 调至 `1` 或 `2`（多跳根因链问题推荐 `k_hops=2`，避免超过 `3` 导致噪声爆炸）。
  - 将 `.env` 的 `GRAPHRAG_EMBEDDING_PROVIDER` 升级为 `openai_compat` + `bge-m3`，大幅提升语义召回能力。

### FD-06：切换 Embedding 模型后查询报矩阵维度不匹配
- **典型现象**：执行 `query` 时抛出 `ValueError: shapes (N, 256) and (1024,) not aligned`。
- **根因分析**：`storage/embeddings.npy` 是旧 Embedding 模型（如 `local_hash` 256 维）生成的，而当前查询切换成了新模型（如 `bge-m3` 1024 维）。
- **修复方案**：切换 Embedding 模型或维度后，必须重新执行一次 `llm-graphrag-kg build` 重建向量索引。

---

## 4. 可视化与性能类故障

### FD-07：大规模图谱在浏览器中渲染卡顿或内存占用高
- **典型现象**：当图谱节点超过 800~1000 个时，PyVis 交互页面持续抖动、CPU 占满。
- **根因分析**：Vis.js 默认开启力导向物理模拟（Barnes-Hut Physics），节点数上升后浏览器端实时计算开销呈超线性增长。
- **修复方案**：
  1. 在工作台「📊 交互图谱」页取消勾选 **「物理布局」**，并使用「筛选实体类型 / 筛选社区 ID」聚焦目标子图。
  2. 对于 > 2,000 节点的全局拓扑分析，使用 `llm-graphrag-kg export-viz` 导出 `outputs/kg_graph.graphml`，在桌面端 **Gephi** 中完成布局与统计。

---

## 5. 隐私保护与 Git 提交流程故障

### FD-08：`git commit` 被本地 `pre-commit` 钩子拦截
- **典型现象**：提交时输出 `Privacy / Hygiene Check FAILED`。
- **常见原因与对应修复**：
  1. `[CHANGELOG] Missing CHANGELOG.md in staged files`：按项目规范，每次提交前须在 `CHANGELOG.md` 追加本次修改日志并 `git add CHANGELOG.md`。
  2. `[FORBIDDEN FILE] / [FORBIDDEN DIR]`：误将 `.env`、`.vscode/`、`data/private/`、`storage/` 或 `outputs/` 加入暂存区；执行 `git reset HEAD <file>` 撤出暂存区。
  3. `[SECRET] / [LOCAL PATH] / [EMAIL]`：代码中硬编码了密钥、本机用户目录绝对路径或个人邮箱；改为相对路径、环境变量或 GitHub noreply 邮箱后重新提交。
- **严禁事项**：不要使用 `git commit --no-verify` 绕过隐私检查！

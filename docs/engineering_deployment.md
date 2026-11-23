# LLM-GraphRAG-KG 工程部署与迁移指南

本文档面向将本框架从个人开发机迁移至 **Linux 服务器、工控边缘主机、内网离线环境或团队协作环境** 的工程部署场景，覆盖环境交付、推理服务配置、存储演进、Web 服务化与安全治理。

---

## 1. 部署架构与演进路线

本框架通过 `core/interfaces.py` 中的抽象协议（`BaseLLMClient`、`BaseEmbeddingClient`、`BaseGraphStore`、`BaseVectorStore`）解耦业务逻辑与底层基础设施，支持按业务规模分三阶段演进：

| 部署阶段 | 推荐规模 | 模型服务 | 图存储 / 向量存储 | 运行形态 |
|---|---|---|---|---|
| **Stage 1：单机原型与核验** | < 500 篇文档 / < 2 万实体 | 本机 Ollama / vLLM / 云端 API | `NetworkXGraphStore` (JSON) + `LocalVectorStore` (NumPy) | CLI + 本地 Streamlit 核验台 |
| **Stage 2：部门/局域网共享** | 500 ~ 5,000 篇文档 | 独立 GPU 服务器（Ollama / vLLM） | 持久化卷 + 定时增量重建（或迁移至 Kùzu / Neo4j） | Docker / systemd + 反向代理 + 定时任务 |
| **Stage 3：生产级多业务系统** | > 5,000 篇文档 / 高并发 | vLLM 集群 / 企业级 API 网关 | Neo4j（实现 `BaseGraphStore`）+ Qdrant/Milvus（实现 `BaseVectorStore`） | API 微服务 + 监控告警 + 权限审计 |

---

## 2. 标准环境部署步骤（基于 `uv`）

### 2.1 在线服务器部署（Windows / Linux 通用）

```bash
# 1. 拉取代码（仅包含脱敏代码、配置模板与示例语料）
git clone <your-repo-url>
cd LLM-GraphRAG-KG

# 2. 按需同步依赖（严格遵循 uv.lock 与 .python-version）
# 仅核心 CLI 管线（适合无界面服务器/定时任务节点）：
uv sync

# 含 Streamlit 可视化核验工作台：
uv sync --extra viz

# 含测试与静态检查工具（开发/CI 节点）：
uv sync --extra viz --extra dev

# 3. 初始化环境变量（仅在目标机器本地维护，严禁回传仓库）
cp .env.example .env
```

### 2.2 内网 / 离线环境迁移部署

当目标现场（如工业内网工控机）无法访问公网 PyPI 时：

1. **在联网机器打包离线 Wheel 缓存**：
   ```bash
   uv export --format requirements-txt --extra viz --output-file requirements.txt
   pip download -r requirements.txt -d ./wheelhouse --python-version 3.12 --only-binary=:all:
   ```
2. **拷贝代码仓库与 `wheelhouse/` 至内网主机**（不要拷贝 `.venv/`、`.env` 或 `.vscode/`）。
3. **在内网主机离线创建虚拟环境并安装**：
   ```bash
   uv venv --python 3.12
   uv pip install --no-index --find-links ./wheelhouse -r requirements.txt
   uv pip install --no-build-isolation --no-deps -e .
   ```

---

## 3. 推理模型与向量服务部署

### 3.1 本机 / 局域网 Ollama 部署（推荐起步）

> 基于 [Stage 3.0 本地模型基准报告](benchmark_stage3_base_model.md)（8 GB 显存实测），推荐以 **`qwen3.5:9b`（主力推理）** 与 **`qwen3.5:4b`（高速离线建图/路由）** 作为默认本地底座。

1. **拉取聊天模型与语义向量模型**：
   ```bash
   ollama pull qwen3.5:9b        # 主力 GraphRAG / 推理模型（8 GB 显存推荐）
   ollama pull qwen3.5:4b        # 可选：高吞吐离线建图 / 轻量路由模型（60-89 tok/s）
   ollama pull nomic-embed-text  # 轻量嵌入（~0.4 GiB，可与 qwen3.5:9b 同驻 8 GB 显存）
   # ollama pull bge-m3          # 中文增强嵌入（~1.2 GiB，建议离线索引阶段单独运行或 12 GB+ 显存同驻）
   ```
2. **在 `.env` 中配置对接参数（8 GB 显存推荐配置）**：
   ```ini
   GRAPHRAG_LLM_PROVIDER=ollama_native
   GRAPHRAG_LLM_BASE_URL=http://127.0.0.1:11434
   GRAPHRAG_LLM_API_KEY=local-no-key
   GRAPHRAG_LLM_MODEL=qwen3.5:9b
   GRAPHRAG_LLM_TIMEOUT=180
   GRAPHRAG_LLM_THINK=false
   GRAPHRAG_LLM_NUM_CTX=16384    # 离线批量抽图可调至 4096-8192；8 GB 显存安全上限 32768
   GRAPHRAG_LLM_NUM_GPU=99       # 强制全 GPU（qwen3.5:9b 16k-32k 提速 1.36x-2.55x）

   GRAPHRAG_EMBEDDING_PROVIDER=openai_compat
   GRAPHRAG_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
   GRAPHRAG_EMBEDDING_API_KEY=local-no-key
   GRAPHRAG_EMBEDDING_MODEL=nomic-embed-text
   ```
   > **8 GB 显存避坑提示**：若使用权重本身已逼近或超过物理显存的大参数模型（例如 `gemma4:12b` 权重 8.0 GB，流畅全 GPU 需 ≥12 GB 显存），**切勿设置 `GRAPHRAG_LLM_NUM_GPU=99`**，请留空交由 Ollama 自动分配，否则会触发 Windows WDDM PCIe 换页导致吞吐骤降 70%–80%。
3. **局域网跨机访问**：若 Ollama 部署在独立 GPU 主机上，需在 GPU 主机设置环境变量 `OLLAMA_HOST=0.0.0.0:11434` 并配置防火墙白名单，切勿直接暴露至公网。

### 3.2 高并发生产服务（vLLM OpenAI 兼容端点）

当需要批量构建数千份故障文档图谱时，推荐使用 vLLM 启动 OpenAI 兼容服务：

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

随后只需修改 `.env` 中的 `GRAPHRAG_LLM_BASE_URL=http://127.0.0.1:8000/v1` 与 `GRAPHRAG_LLM_MODEL`，**无需改动一行框架源码**。

---

## 4. 领域配置与多场景并行部署

本框架支持通过独立 YAML 文件隔离不同业务域的语料路径、本体类型（Entity Types）和持久化目录。例如，使用内置的**工业故障诊断场景配置** [configs/fault_diagnosis.yaml](../configs/fault_diagnosis.yaml)：

```bash
# 1. 验证模型连通性
uv run llm-graphrag-kg check-llm --config-file configs/fault_diagnosis.yaml

# 2. 构建故障诊断领域知识图谱
uv run llm-graphrag-kg build --config-file configs/fault_diagnosis.yaml

# 3. 导出质量审计报告与交互式图谱
uv run llm-graphrag-kg audit --config-file configs/fault_diagnosis.yaml
uv run llm-graphrag-kg export-viz --config-file configs/fault_diagnosis.yaml --color-by community

# 4. 执行跨文档故障根因推理问答并导出子图高亮网页
uv run llm-graphrag-kg query "离心泵出现2倍频振动升高且电机电流波动时，可能根因与处置SOP是什么？" \
  --config-file configs/fault_diagnosis.yaml --mode hybrid --export-html
```

---

## 5. 服务化部署（Streamlit 核验台与定时建图）

### 5.1 本机 / 局域网启动核验工作台

```bash
# 仅本机访问（默认安全绑定）
uv run --extra viz llm-graphrag-kg ui --host 127.0.0.1 --port 8501

# 局域网内网共享访问（建议前置 Nginx 鉴权）
uv run --extra viz llm-graphrag-kg ui --host 0.0.0.0 --port 8501
```

### 5.2 Linux `systemd` 托管示例

```ini
[Unit]
Description=LLM-GraphRAG-KG Verification Workbench
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/llm-graphrag-kg
ExecStart=/opt/llm-graphrag-kg/.venv/bin/llm-graphrag-kg ui --host 127.0.0.1 --port 8501
Restart=on-failure
RestartSec=5
EnvironmentFile=/opt/llm-graphrag-kg/.env

[Install]
WantedBy=multi-user.target
```

---

## 6. 存储向图数据库平滑迁移（NetworkX → Neo4j）

当单机图谱实体超过 **50,000** 或需要多用户并发 Cypher 查询时，按以下 3 步迁移，核心管线（`pipelines/`）与可视化导出无需重写：

1. 在 `pyproject.toml` 的 `[project.optional-dependencies]` 中新增 `neo4j = ["neo4j>=5.20"]`。
2. 在 `src/llm_graphrag_kg/providers/storage/neo4j_store.py` 中实现 `BaseGraphStore` 协议（`upsert_entity`、`upsert_relation`、`find_entities_by_query`、`get_k_hop_subgraph` 等方法）。
3. 在 `providers/factory.py` 的 `create_graph_store()` 中根据配置开关返回 `Neo4jGraphStore`。
4. **存量数据导入**：可直接读取现有 `storage/graph.json` 或 `outputs/kg_graph.graphml` 批量写入 Neo4j。

---

## 7. 上线与发布前检查清单

- [ ] 运行 `uv run llm-graphrag-kg check-llm` 确认推理与向量服务可达。
- [ ] 运行 `uv run pytest -v` 与 `uv run ruff check src tests scripts` 全部通过。
- [ ] 核查 `.env` 权限（Linux 推荐 `chmod 600 .env`），确保日志中 API Key 均已脱敏。
- [ ] 私有文档仅存放于 `data/private/`，运行产物位于 `storage/` 与 `outputs/`。
- [ ] 提交或推送 GitHub 前执行 `uv run llm-graphrag-kg privacy-check --staged --require-changelog`。

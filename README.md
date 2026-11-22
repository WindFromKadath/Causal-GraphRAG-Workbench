# LLM-GraphRAG-KG

一个面向初学者友好、同时具备平滑迁移能力的 **LLM + GraphRAG + 知识图谱（KG）** 基础 Python 框架，使用 `uv` 进行环境管理，内置**可视化内容核验**与**面向 GitHub 的隐私保护机制**。

## 核心特性

- **分层解耦架构（Migration-Ready）**：核心管线仅依赖 `Protocol` 抽象接口，默认零外部数据库依赖（`NetworkX` + 本地向量存储），后续可平滑替换至 Neo4j / 专用向量库。
- **多模型兼容**：统一采用 OpenAI-compatible 协议，通过 `.env` 一键切换云端大模型 API、本机 Magpie / Ollama / vLLM 等本地模型服务，亦内置离线 `mock` 模式用于零成本调试与自动化测试。
- **可视化内容核验**：
  - Phase 1：自动生成图谱质量审计报告（Markdown/CSV）、标准 `.graphml`（兼容 Gephi/Cytoscape）、以及 `PyVis` 单文件交互式 HTML 图谱（支持悬停原文溯源与**问答检索子图高亮**）。
  - Phase 2：基于 `Streamlit` 的一站式本地可视化核验工作台（同屏查看文档切片、过滤实体/三元组并反查原文、社区总览、实时提问与检索子图联动高亮）。
- **隐私与项目管理规范**：
  - 严格隔离代码、配置、脱敏示例数据（`data/sample/`）与私有运行数据（`data/private/`、`storage/`、`outputs/`）。
  - 每次 Git 提交前自动执行隐私与修改日志（`CHANGELOG.md`）检查。

## 目录结构

```text
LLM-GraphRAG-KG/
├── .env.example                 # 环境变量模板（复制为 .env 后填入实际配置）
├── CHANGELOG.md                 # 修改日志（每次 Git 提交同步更新）
├── configs/
│   ├── default.yaml             # 通用默认运行参数
│   ├── fault_diagnosis.yaml     # 工业工程部署与故障诊断领域本体配置
│   └── failuresensoriq.yaml     # IBM FailureSensorIQ（Apache-2.0）标准数据集配置
├── data/
│   ├── sample/                  # 轻量基础演示语料（3 篇）
│   └── sample_fault_diagnosis/  # 工程部署与工业故障诊断脱敏语料（6 篇，含多跳因果链）
├── docs/
│   ├── engineering_deployment.md     # 工程部署、离线迁移与存储演进指南
│   └── troubleshooting_diagnosis.md  # 系统故障诊断与排障手册（FD-01 ~ FD-08）
├── scripts/                     # 隐私与提交规范检查工具
├── src/llm_graphrag_kg/         # 核心框架源码
└── tests/                       # 自动化测试套件
```

## 快速开始（基于 `uv`）

```bash
# 1. 安装核心与开发依赖
uv sync --extra dev

# 2. 复制环境变量模板（可选：如需连接本地 Magpie/Ollama 或云端 API）
cp .env.example .env

# 3. 检查本地模型或 API 连通性
uv run llm-graphrag-kg check-llm

# 4. 使用示例语料构建知识图谱与向量索引（离线演示可加 --provider mock）
uv run llm-graphrag-kg build --input data/sample --provider mock

# 5. 查看图谱质量审计报告 & 导出 PyVis 交互式 HTML / GraphML
uv run llm-graphrag-kg audit
uv run llm-graphrag-kg export-viz

# 6. 执行 GraphRAG 问答并导出带检索子图高亮的交互式核验网页
uv run llm-graphrag-kg query "GraphRAG 如何结合知识图谱解决多跳推理问题？" --mode hybrid --provider mock --export-html

# 7. 启动 Phase 2 Streamlit 一站式可视化核验工作台
uv sync --extra viz --extra dev
uv run --extra viz llm-graphrag-kg ui

# 8. 接入 IBM FailureSensorIQ（Apache-2.0）公开基准集并评测 LLM-only vs Naive RAG vs GraphRAG+KG
uv run llm-graphrag-kg ingest-fsiq --source data/private/failuresensoriq/failuresensoriq.mapped.jsonl
uv run llm-graphrag-kg audit --config-file configs/failuresensoriq.yaml
uv run llm-graphrag-kg export-viz --config-file configs/failuresensoriq.yaml --color-by type
uv run llm-graphrag-kg eval-fsiq --questions data/private/failuresensoriq/failuresensoriq_standard__all.jsonl --limit 20

# 9. 四级难度极限压测（每集 100 题，含 10选1 / 5选2多选 / 扰动集，支持极简 JSON 提速与断点续跑）
uv run llm-graphrag-kg stress-fsiq --data-dir data/private/failuresensoriq --limit 100 --no-think --num-ctx 32768
```

## 隐私与提交流程

- 严禁将 `.env`、`.vscode/`、`data/private/`、`storage/`、`outputs/` 提交至 Git。
- 每次提交前，请同步更新 [CHANGELOG.md](CHANGELOG.md)，并运行：

```bash
uv run llm-graphrag-kg privacy-check --staged --require-changelog
```

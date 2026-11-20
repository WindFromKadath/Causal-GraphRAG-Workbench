# 修改日志 (CHANGELOG)

本文件遵循“**每次 Git 提交同步记录一次修改日志**”的项目管理规范，用于追溯每次架构演进、功能实现、核验结果与隐私安全状态。

> 隐私约束：日志中禁止记录任何 API Key、本机绝对路径、个人邮箱或私有业务数据。

---

## [2026-11-20] Phase 1 - LLM + GraphRAG + KG 核心闭环与 PyVis/GraphML 可视化核验
- **提交类型**：`feat`
- **变更背景**：在 Phase 0 隐私与工程基线之上，落地分层解耦的 LLM + GraphRAG + 知识图谱最小可行闭环（MVP）与第一阶段可视化核验工具。
- **主要修改**：
  - 新增核心领域模型与抽象协议：`src/llm_graphrag_kg/core/schema.py`、`interfaces.py` 与 `config.py`（支持 YAML 默认配置 + `.env` 密钥注入与日志脱敏）。
  - 新增可插拔适配层：`providers/llm/`（`OpenAICompatLLMClient` 兼容云端 API 与本机 Magpie/Ollama，附 `MockLLMClient` 离线调试）、`providers/embedding/`（`OpenAICompatEmbeddingClient` + `LocalHashEmbeddingClient`）、`providers/storage/`（`NetworkXGraphStore` + `LocalVectorStore`）与工厂模块 `providers/factory.py`。
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
  - 新增脱敏环境变量模板 `.env.example`（兼容云端 API、本机 Magpie/Ollama OpenAI 兼容接口及离线 Mock 模式）与非敏感运行配置 `configs/default.yaml`。
  - 新增 `scripts/privacy_check.py` 隐私与提交规范检查脚本，并在本地配置 Git `pre-commit` 钩子自动拦截隐私泄露与遗漏修改日志的提交。
  - 初始化项目文档 `README.md` 与 `CHANGELOG.md`。
- **验证方式**：执行 `uv sync` 锁文件同步与 `scripts/privacy_check.py --staged --require-changelog` 校验通过。
- **隐私与迁移自检**：已确认本地 Git 作者改用 GitHub noreply 邮箱，无敏感密钥、无本机绝对路径、无私人邮箱进入仓库。

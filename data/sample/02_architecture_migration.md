# 面向迁移的模块化架构与模型适配

为了支持项目平滑迁移，本框架基于 Pydantic 定义统一数据模型，并通过 BaseLLMClient、BaseGraphStore 与 BaseVectorStore 抽象接口解耦业务逻辑与底层实现。

在模型接入层，BaseLLMClient 支持 OpenAI-compatible 协议，可无缝连接云端 API 以及本机的 Ollama 或 vLLM 模型服务。

在存储层，起步阶段默认使用 NetworkX 持久化本地 知识图谱，无需部署独立图数据库；当未来数据规模扩大时，只需实现 BaseGraphStore 即可平滑迁移至 Neo4j，而无需重写 GraphRAG 检索管线。

整个项目依赖由 uv 统一管理，确保跨机器复现一致的 Python 环境。

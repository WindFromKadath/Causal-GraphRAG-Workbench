# GraphRAG 与知识图谱基础原理

GraphRAG 结合 知识图谱（Knowledge Graph）与 大语言模型（LLM），用于解决传统 向量检索（Naive RAG）在跨文档 多跳推理 和全局总结问答上的局限性。

在离线构建阶段，LLM 对文档切片执行实体与关系抽取，并通过 实体对齐（Entity Resolution）合并同名节点，形成结构化的 知识图谱。随后，系统使用 Louvain 算法执行 社区发现，并为每个主题簇生成 社区摘要。

在在线问答阶段，Local Search 通过种子实体展开多跳邻域子图并保留 原文溯源；Global Search 汇总高相关 社区摘要 回答宏观问题；Hybrid Search 结合 向量检索 与图谱子图提升召回精度。

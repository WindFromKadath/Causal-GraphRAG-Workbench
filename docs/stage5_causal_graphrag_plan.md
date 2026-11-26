# 第五阶段规划：因果知识图谱与因果增强 GraphRAG（Stage 5 — Causal GraphRAG）

> **文档定位**：本规划承接 Stage 4（企业级全栈底座：PG16+pgvector + Neo4j 5.26 + Redis 7 + Doc Governance + Thin LangGraph + FastAPI + React 19），聚焦解决 GraphRAG 最本质的认知难题——**从“相关性/共现图谱”跃迁至“有向因果与可干预知识图谱（Causal KG）”**。

---

## 1. 为什么必须设立 Stage 5：从现状问题到学术共识

在完成全部 3 组公开资料（OREDA `FMEA_KG`、IBM `FailureSensorIQ`、GraphRAG-Bench `Medical/Novel`）与示例语料（`fault_diagnosis`）向 PG16 + Neo4j 5.26 的直接迁移后，边类型分布暴露出典型分层现状：
- **结构化本体源**（`OREDA FMEA_KG`、`IBM FailureSensorIQ`）：天然具备高纯度结构/监测/故障模式边与 64 条多传感器超边；
- **中小规模文本源**（`fault_diagnosis`）：早期快速抽取沉淀了较多泛化 `RELATED_TO` 边；
- **大规模纯文本源**（`grb_medical` 6,522 条 / `grb_novel` 2,683 条）：96%–99% 为无向统计共现边 `CO_OCCURS_WITH` + 向量近邻边 `SYNONYM_OF`。

最新顶会研究（`CausalRAG` ACL 2025、`Kıcıman et al.` TMLR 2024、`Wan et al.` IJCAI Survey、`LazyGraphRAG` Microsoft 2024、`HyperGraphRAG` NeurIPS 2025）共同指出：**向量相似度 $\neq$ 因果关系，同段共现 $\neq$ 因果方向**。只有将“向量对齐层、结构骨架层、有向因果层、多条件超边层”彻底拆分并施加符号约束，才能避免多跳推理发散与共因伪相关误判。

---

## 2. 核心理论与工程建模规范

### 2.1 三层关系严格隔离（绝不混用）
1. **L1 向量语义对齐层（Semantic Bridge — 无方向）**：PG16 `halfvec(1024)` + Neo4j `SYNONYM_OF (edge_origin="vector_sim")`。仅负责跨词/跨语言实体入口对齐，**严禁作为因果路径参与根因计分**。
2. **L2 关联/共现层（Associative — 弱相关）**：`CO_OCCURS_WITH` / `RELATED_TO`，标记 `relation_layer="associative"`，在根因路径与 ToT 剪枝中强降权或隔离。
3. **L3 有向因果与干预层（Directed Causal & Actionable — 图谱核心资产）**：按正交 4 层本体抽分（`structural` / `causal` / `diagnostic_action` / `hyper_causal`）。

### 2.2 五种因果间结构（Inter-Causal Motifs — 对标 Pearl SCM）

| 因果间结构 | 拓扑形态 | 工业 / 医学示例 | Neo4j / 超图标准建模 |
|---|---|---|---|
| **① 因果链（Chain / 机理传递）** | $A \to M \to B$ | `轴系不对中 → 附加轴向载荷 → 轴承温升` | 保留完整有向 `CAUSES` 链，标注 `causal_role: root / mediator / terminal`；若同时存在捷径边 $A \to B$，标注 `is_transitive_shortcut=true` |
| **② 共因分叉（Fork / 混杂伪相关）** | $A \leftarrow C \to B$ | `2倍频振动 ← 轴系不对中 → 轴承温升` | **严禁在同层症状 $A, B$ 间连 `CAUSES`**；仅通过共同上游 $C$ 连接，或打包为指向 $C$ 的 `:HyperEdge {logic: "AND", JOINT_INDICATES}` |
| **③ 协同汇聚（Collider-AND / 联合致因）** | $(A \land B) \to E$ | `润滑脂老化 ∧ 持续超负荷 → 轴承胶合` | **严禁拆成孤立二元边**，必须建模为 `:HyperEdge {logic: "AND", JOINT_CAUSES}` |
| **④ 竞争汇聚（Collider-OR / 互斥致因）** | $A \to E \lor B \to E$ | `轴系不对中 → 轴承温升` OR `润滑脂劣化 → 轴承温升` | 保留为多条独立二元 `CAUSES` 边 + `RULES_OUT` / `DIFFERENTIATED_BY` 区分性特征边，直接驱动 **Graph-ToT 剪枝** |
| **⑤ 干预阻断与反馈环（Intervention / Feedback）** | $I \dashv (A \to B)$ 或 $A \leftrightarrows B$ | `激光对中` 阻断 `不对中→振动`；`温升 → 油膜变薄 → 摩擦加剧 → 温升` | 干预用 `MITIGATES / BLOCKS / TREATS` 指向根因或超边；仅显式标注 `is_feedback_loop=true` 的物理恶化环允许闭环 |

### 2.3 三级因果证据强度（对标 Pearl 因果阶梯）
每条因果边必须附带 `evidence_tier` 与原文依据片段 `evidence_span`：
- **`L3_INTERVENTIONAL`（权重 `1.00`）**：含干预/复机反事实实证（“执行激光对中复机后，2倍频振动与电流波动消除”）；
- **`L2_MECHANISTIC`（权重 `0.85`）**：含明确物理/病理传导机理，或源自标准本体（OREDA / FailureSensorIQ / 临床指南）；
- **`L1_OBSERVATIONAL`（权重 `0.55`）**：仅观察共现/伴随陈述（“常见于 / associated with”），仅可标为 `ASSOCIATED_WITH / INDICATES`，**禁止直接晋升为高置信度 `CAUSES`**。

### 2.4 四条 `Causal-SHACL` 机械硬校验门禁
1. **本体类型方向锁（Ontological Direction Lock）**：强制 `RootCause/Fault/Mechanism → Symptom/Signal`、`Measure/Treatment ⊣ Fault`；自动拦截并反转任何 `Symptom --[CAUSES]--> Fault`。
2. **DAG 无环与双向冲突仲裁（Reciprocal Cycle Guard）**：拦截未标注 `is_feedback_loop=true` 的 $A \leftrightarrow B$ 幻觉双向 `CAUSES`，按 `L3 > L2 > L1` 自动裁决或转入 `CONFLICT_PENDING` 人工审校。
3. **AND / OR 汇聚自动分流**：在 `:HyperEdge` 增加 `logic: "AND" | "OR"` 属性，区分联合必要条件与竞争互斥根因。
4. **传递约简标注（Transitive Shortcut Tagging）**：自动识别 $A \to M \to B$ 对应的冗余捷径边 $A \to B$，检索与前端展示优先呈现完整机理链。

---

## 3. Stage 5 分步实施规划

| 子阶段 | 核心目标 | 关键交付与验收门禁 |
|---|---|---|
| **Stage 5.1** | **因果 Schema 扩展与 `Causal-SHACL` 门禁** | 扩展 `Relation`（`relation_layer` / `evidence_tier` / `evidence_span` / `is_feedback_loop` / `is_transitive_shortcut`）与 `HyperEdge`（`logic: "AND" \| "OR"`）；在 `viz/schema_validator.py` 落地 4 条 `Causal-SHACL` 校验器与离线单测 |
| **Stage 5.2** | **企业级大规模文本因果漏斗抽取与向量阻塞 ER** | 落地 `CausalTriageFilter`（因果触发词/信息密度门控，滤除 40%–60% 无因果铺垫切片）+ `CausalOpenIEExtractor`（4 层本体 + `L1/L2/L3` 证据抽分 + 断点续抽）+ **Embedding Vector Blocking 实体消歧**；完成 `fault_diagnosis` 因果重抽（淘汰 `RELATED_TO`）与 `grb_medical` 因果图谱增量升级 |
| **Stage 5.3** | **因果感知检索、ToT 反证剪枝与反事实推演** | 升级 PPR 与 Graph-ToT：`SYNONYM_OF / CO_OCCURS_WITH` 仅作入口桥接、因果路径按 `L3 > L2 > L1` 加权、原生支持 `RULES_OUT` 硬反证剪枝与干预/反事实问答（`do(X)`）；在 React 控制台 Cytoscape.js 中按因果层与证据等级分层高亮可视化 |

---

## 4. 核心参考文献索引

1. **CausalRAG** — Wang et al., *CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation*, **ACL 2025** ([arXiv:2503.19878](https://arxiv.org/abs/2503.19878))
2. **Causal Reasoning & LLMs** — Kıcıman, Ness, Sharma, Tan, *Causal Reasoning and Large Language Models: Opening a New Frontier for Causality*, **TMLR 2024 Featured** ([arXiv:2305.00050](https://arxiv.org/abs/2305.00050))
3. **LLM Causal Discovery Survey** — Wan et al., *Large Language Models for Causal Discovery: Current Landscape and Future Directions*, **IJCAI Survey** ([arXiv:2402.11068](https://arxiv.org/abs/2402.11068))
4. **LazyGraphRAG** — Microsoft Research, *LazyGraphRAG: Setting a new standard for quality and cost*, **2024.11**
5. **HyperGraphRAG** — *HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation*, **NeurIPS 2025**
6. **GraphRAG-Causal / DKCD / Healthcare Causal KG** — Neo4j 混合因果图谱抽取、领域本体先验方向消歧与医疗物化溯源断言（2025–2026）
7. **Judea Pearl** — *Causality: Models, Reasoning, and Inference* / *The Book of Why*（三层因果阶梯与 Chain / Fork / Collider 因果结构）

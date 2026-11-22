"""Streamlit visual verification workbench for LLM + GraphRAG + KG (Stage 2.4).

Features:
- Locked primary default dataset: `configs/failuresensoriq.yaml` (233 entities / 692 relations / 4-layer ISO KG)
  while preserving YAML switching across all workspace configs.
- Tab 1: Interactive PyVis Graph Explorer + Single-Entity / Single-Equipment Ego-Graph Focus Mode.
- Tab 2: Real-Time Query with PPR / Hybrid / Local / Global, Explicit Graph-CoT & Graph-ToT Replay,
  Human-Readable Citation Badges, GroundingAudit Verification, and L0/L1/L2 Compact Telemetry.
- Tab 3: SHACL-lite 100% Full-Graph Ontology & Provenance Compliance Auditor.
- Tabs 4-6: Entities & Triples Provenance Explorer, Louvain Communities, and Source Text Chunks.

Launch via:
    uv run --extra viz llm-graphrag-kg ui
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from llm_graphrag_kg.config import AppConfig, load_config
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.factory import create_llm_client
from llm_graphrag_kg.viz.audit_report import compute_audit_stats, export_audit_artifacts
from llm_graphrag_kg.viz.graphml_exporter import export_graphml
from llm_graphrag_kg.viz.html_exporter import render_pyvis_html
from llm_graphrag_kg.viz.schema_validator import validate_kg_shacl_lite
from llm_graphrag_kg.viz.workbench_data import (
    build_chunk_records,
    build_community_records,
    build_entity_records,
    build_readable_citation_maps,
    build_relation_records,
    extract_ego_subgraph,
    filter_entities,
    filter_relations,
    resolve_provenance_chunks,
)

PRIMARY_DATASET_CONFIG = "configs/failuresensoriq.yaml"


def _render_html_embed(html: str, *, height: int) -> None:
    """Embed interactive PyVis HTML using `st.iframe` (Streamlit >=1.5x) with legacy fallback."""
    if hasattr(st, "iframe"):
        st.iframe(html, width="stretch", height=height)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=height, scrolling=True)


def _init_page() -> None:
    st.set_page_config(
        page_title="LLM-GraphRAG-KG 可视化核验与思维回放工作台",
        page_icon="🕸️",
        layout="wide",
    )
    st.title("🕸️ LLM + GraphRAG + KG 可视化核验与思维回放工作台")
    st.caption(
        "Stage 2.4 闭环工作台：Ego-Graph 单设备聚焦 → 显性 Graph-CoT / Graph-ToT 思维回放与 "
        "GroundingAudit → SHACL-lite 全量本体质检 → 原文切片溯源"
    )


def _load_pipeline_state(
    cfg: AppConfig,
    storage_dir: Path,
) -> tuple[GraphRAGPipeline | None, int, str | None]:
    pipeline = GraphRAGPipeline(cfg)
    try:
        doc_count = pipeline.load(storage_dir)
        return pipeline, doc_count, None
    except FileNotFoundError as exc:
        return None, 0, str(exc)


def _discover_config_files() -> list[str]:
    cfg_dir = Path.cwd() / "configs"
    if not cfg_dir.is_dir():
        return ["configs/default.yaml"]
    found = sorted(
        p.relative_to(Path.cwd()).as_posix()
        for p in cfg_dir.glob("*.yaml")
        if not p.name.endswith(".local.yaml")
    )
    if not found:
        return ["configs/default.yaml"]
    # Prioritize locked primary Stage 2.4 dataset (`failuresensoriq.yaml`) as default index 0
    if PRIMARY_DATASET_CONFIG in found:
        found.remove(PRIMARY_DATASET_CONFIG)
        found.insert(0, PRIMARY_DATASET_CONFIG)
    return found


def _default_question_for_config(cfg_rel: str) -> str:
    if "failuresensoriq" in cfg_rel:
        return (
            "For pump, if damaged seals occurs, which sensors monitor it and how does "
            "it differ from bearing wear?"
        )
    if "fault_diagnosis" in cfg_rel:
        return "离心泵机组同时出现2倍频振动升高、轴承温升异常且电机定子电流波动，应如何定位根因并执行SOP？"
    if "graphrag_bench" in cfg_rel:
        return (
            "Why is a patient with fair skin and a history of organ transplant at high risk "
            "for basal cell carcinoma?"
        )
    return "GraphRAG 如何结合知识图谱解决多跳推理并支持迁移至 Neo4j？"


def _render_sidebar() -> tuple[AppConfig, Path, Path, str]:
    with st.sidebar:
        st.header("⚙️ 运行配置与操作")

        config_files = _discover_config_files()
        selected_cfg_rel = st.selectbox(
            "场景配置文件 (YAML)",
            options=config_files,
            index=0,
            help=(
                "默认锁定 Stage 2.4 主验收数据集 `configs/failuresensoriq.yaml`（233实体/692边），"
                "亦可随时切换至故障诊断或 GraphRAG-Bench 配置。"
            ),
        )
        selected_cfg_path = Path.cwd() / selected_cfg_rel

        provider_choice = st.selectbox(
            "LLM Provider",
            options=["default (.env)", "ollama_native", "openai_compat", "mock"],
            index=0,
            help="默认读取本地 .env；选择 ollama_native 可精确控制 --no-think；mock 可离线调试。",
        )
        provider_override = None if provider_choice == "default (.env)" else provider_choice
        disable_think = st.checkbox(
            "关闭黑盒思考 (think=False)",
            value=True,
            help="Stage 2 推荐：关闭模型不可控隐式思考，交由显性 Graph-CoT/ToT 引擎输出可审计步骤。",
        )

        cfg = load_config(
            config_path=selected_cfg_path,
            llm_provider_override=provider_override,
        )
        if disable_think:
            cfg.env.llm_think = False

        input_rel = st.text_input(
            "输入文档目录（相对路径）",
            value=cfg.paths.input_dir,
            key=f"input_dir_{selected_cfg_rel}",
        )
        storage_rel = st.text_input(
            "图谱存储目录（相对路径）",
            value=cfg.paths.storage_dir,
            key=f"storage_dir_{selected_cfg_rel}",
        )

        input_dir = cfg.resolve_path(input_rel)
        storage_dir = cfg.resolve_path(storage_rel)

        with st.expander("当前脱敏配置预览", expanded=False):
            st.json(cfg.redacted_summary())

        if st.button("🔌 检查模型连通性", width="stretch"):
            client = create_llm_client(cfg)
            ok, msg = client.health_check()
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.divider()
        st.subheader("🏗️ 离线建图")
        summarize_comm = st.checkbox("使用 LLM 生成社区摘要", value=False)
        if st.button("▶️ 重新构建知识图谱", type="primary", width="stretch"):
            with st.spinner("正在执行文档切片、实体关系抽取与社区发现..."):
                try:
                    builder = GraphRAGPipeline(cfg)
                    stats = builder.build(
                        input_dir=input_dir,
                        storage_dir=storage_dir,
                        summarize_communities=summarize_comm,
                    )
                    st.session_state.pop("last_query_result", None)
                    st.success(
                        f"构建完成：{stats.entity_count} 实体 / {stats.relation_count} 关系 / "
                        f"{stats.community_count} 社区"
                    )
                except Exception as exc:
                    st.error(f"构建失败：{type(exc).__name__}: {exc}")

        if st.button("🔄 刷新存储视图", width="stretch"):
            st.rerun()

    return cfg, input_dir, storage_dir, selected_cfg_rel


def _render_exports_sidebar(pipeline: GraphRAGPipeline, cfg: AppConfig, doc_count: int) -> None:
    with st.sidebar:
        st.divider()
        st.subheader("📦 导出核验产物")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _, audit_paths = export_audit_artifacts(
                pipeline.graph_store,
                pipeline.vector_store,
                tmp_dir,
                doc_count=doc_count,
            )
            graphml_path = export_graphml(pipeline.graph_store, tmp_dir / "kg_graph.graphml")
            report_bytes = audit_paths["audit_report"].read_bytes()
            graphml_bytes = graphml_path.read_bytes()

        shacl_report = validate_kg_shacl_lite(pipeline.graph_store, pipeline.vector_store, cfg)
        html_str = render_pyvis_html(
            pipeline.graph_store,
            vector_store=pipeline.vector_store,
            color_by="type",
        )

        st.download_button(
            "⬇️ 下载质量审计报告 (.md)",
            data=report_bytes,
            file_name="audit_report.md",
            mime="text/markdown",
            width="stretch",
        )
        st.download_button(
            "⬇️ 下载 SHACL-lite 质检报告 (.md)",
            data=shacl_report.to_markdown().encode("utf-8"),
            file_name="shacl_validation_report.md",
            mime="text/markdown",
            width="stretch",
        )
        st.download_button(
            "⬇️ 下载交互式图谱 (.html)",
            data=html_str.encode("utf-8"),
            file_name="kg_graph.html",
            mime="text/html",
            width="stretch",
        )
        st.download_button(
            "⬇️ 下载 GraphML (Gephi)",
            data=graphml_bytes,
            file_name="kg_graph.graphml",
            mime="application/xml",
            width="stretch",
        )


def _render_kpi_bar(pipeline: GraphRAGPipeline, cfg: AppConfig, doc_count: int) -> None:
    stats = compute_audit_stats(pipeline.graph_store, pipeline.vector_store, doc_count=doc_count)
    shacl = validate_kg_shacl_lite(pipeline.graph_store, pipeline.vector_store, cfg)

    cols = st.columns(8)
    cols[0].metric("文档数", stats.doc_count)
    cols[1].metric("切片数", stats.chunk_count)
    cols[2].metric("实体数", stats.entity_count)
    cols[3].metric("关系数", stats.relation_count)
    cols[4].metric("社区数", stats.community_count)
    cols[5].metric("连通分量", stats.connected_components)
    cols[6].metric("孤立实体", len(stats.isolated_entities))
    cols[7].metric(
        "SHACL-lite",
        f"{shacl.overall_status} ({shacl.compliance_score * 100:.0f}%)",
    )


def _render_graph_tab(
    pipeline: GraphRAGPipeline,
    entity_records: list[dict[str, Any]],
    relation_records: list[dict[str, Any]],
) -> None:
    all_types = sorted({r["entity_type"] for r in entity_records})
    all_comms = sorted({r["community_id"] for r in entity_records})
    all_rel_types = sorted({r["relation_type"] for r in relation_records})

    default_ego = len(entity_records) > 80
    top_c1, top_c2, top_c3 = st.columns([1.6, 1.2, 1.0])
    ego_mode = top_c1.toggle(
        "🎯 单设备 / 单节点 Ego-Graph 聚焦模式（推荐 >80 节点大图开启）",
        value=default_ego,
        help="以单个设备或故障节点为中心抽取 1~2 跳局部邻域，彻底消除全图毛线球拥挤。",
    )
    color_by = top_c2.radio("节点着色方式", options=["type", "community"], horizontal=True)
    physics = top_c3.checkbox("物理布局", value=True)

    if ego_mode:
        # Prioritize Equipment nodes at top for FailureSensorIQ ergonomic inspection
        prioritized = sorted(
            entity_records,
            key=lambda r: (
                0 if r["entity_type"] == "Equipment" else (1 if r["entity_type"] == "FailureMode" else 2),
                -int(r["degree"]),
                r["name"],
            ),
        )
        label_to_id = {
            f"{r['name']} [{r['entity_type']} | deg={r['degree']}]": r["entity_id"]
            for r in prioritized
        }
        e1, e2, e3, e4 = st.columns([2.2, 1.0, 1.0, 1.8])
        chosen_center_label = e1.selectbox(
            "选择中心节点（默认优先列出设备 Equipment）",
            options=list(label_to_id.keys()),
            index=0,
        )
        k_hops = e2.slider("聚焦跳数 (k-hop)", min_value=1, max_value=2, value=2)
        max_nodes = e3.slider("最大展示节点数", min_value=15, max_value=80, value=45, step=5)
        sel_rel_types = e4.multiselect(
            "筛选关系类型", options=all_rel_types, default=all_rel_types, key="ego_rel_types"
        )

        center_id = label_to_id[chosen_center_label]
        allowed_ids, ego_trace, ego_summary = extract_ego_subgraph(
            pipeline.graph_store,
            center_id,
            k_hops=k_hops,
            max_nodes=max_nodes,
            relation_types=sel_rel_types or None,
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("中心节点", f"{ego_summary['center_name']}")
        k2.metric("聚焦子图节点", f"{ego_summary['ego_nodes']} / {ego_summary['total_nodes']}")
        k3.metric("视图降噪率", f"-{ego_summary['reduction_pct']:.1f}%")
        k4.metric("子图关系边", ego_summary["ego_relations"])

        if not allowed_ids:
            st.warning("当前聚焦条件下没有可展示的实体节点。")
            return

        html = render_pyvis_html(
            pipeline.graph_store,
            vector_store=pipeline.vector_store,
            color_by=color_by,
            physics_enabled=physics,
            allowed_entity_ids=allowed_ids,
            trace=ego_trace,
            height="680px",
            title=(
                f"Ego-Graph ({ego_summary['center_name']} | {k_hops}-hop | "
                f"{ego_summary['ego_nodes']} nodes)"
            ),
        )
        _render_html_embed(html, height=740)
        return

    # Full-graph filter mode
    c2, c3 = st.columns(2)
    selected_types = c2.multiselect("筛选实体类型", options=all_types, default=all_types)
    selected_comms = c3.multiselect("筛选社区 ID", options=all_comms, default=all_comms)

    filtered = filter_entities(
        entity_records,
        entity_types=selected_types,
        community_ids=selected_comms,
    )
    allowed_ids = {r["entity_id"] for r in filtered}

    if not allowed_ids:
        st.warning("当前筛选条件下没有可展示的实体节点。")
        return

    html = render_pyvis_html(
        pipeline.graph_store,
        vector_store=pipeline.vector_store,
        color_by=color_by,
        physics_enabled=physics,
        allowed_entity_ids=allowed_ids,
        height="680px",
        title=f"Knowledge Graph Explorer (Filtered: {len(allowed_ids)} nodes)",
    )
    _render_html_embed(html, height=740)


def _render_reasoning_replay(
    reasoning_data: dict[str, Any],
    pipeline: GraphRAGPipeline,
) -> None:
    """Render MindMap + Faithful-CoT dual-track replay with readable labels & audit badges."""
    ent_labels, rel_labels = build_readable_citation_maps(pipeline.graph_store)
    paradigm = str(reasoning_data.get("paradigm", "cot")).upper()
    audit = reasoning_data.get("grounding_audit", {})

    st.markdown(f"### 🧠 显性思维回放与忠实度审计（`Graph-{paradigm}`）")
    if isinstance(audit, dict):
        score_pct = float(audit.get("grounding_score", 0.0)) * 100.0
        g_steps = int(audit.get("grounded_steps", 0))
        t_steps = int(audit.get("total_steps", 0))
        ungrounded = audit.get("ungrounded_citations", []) or []
        passed = bool(audit.get("passed", False))
        banner_text = (
            f"**GroundingAudit {'PASS' if passed else 'ATTENTION'}** · "
            f"忠实度 **{score_pct:.1f}%**（{g_steps}/{t_steps} 步扎根）· "
            f"编造引用数 `{len(ungrounded)}`"
            + (" · 已触发自愈修复" if audit.get("repair_triggered") else "")
        )
        if passed:
            st.success(banner_text)
        else:
            st.warning(banner_text + (f" | 未扎根引用: `{ungrounded}`" if ungrounded else ""))

    # Graph-ToT Competing Hypothesis Branches (Selected vs Pruned + L2 Folded Counter-Evidences)
    branches = reasoning_data.get("branches") or []
    if branches:
        st.markdown("#### 🌳 Graph-ToT 竞争假设分支对比（图谱分 `G` vs 语义分 `S` + 反证剪枝）")
        cols = st.columns(min(len(branches), 3))
        for idx, br in enumerate(branches):
            col = cols[idx % len(cols)]
            is_win = br.get("status") == "selected"
            eid = str(br.get("target_entity_id", ""))
            readable_target = ent_labels.get(eid, br.get("target_entity_name") or eid or "-")
            with col:
                if is_win:
                    st.success(
                        f"**{br.get('branch_id')} · SELECTED** | 综合分 **{br.get('score', 0.0):.2f}**\n\n"
                        f"{br.get('hypothesis', '')}\n\n"
                        f"🎯 目标节点：`{readable_target}`\n\n"
                        f"📊 图谱分 `G={br.get('deterministic_graph_score', 0.0):.2f}` · "
                        f"语义分 `S={br.get('semantic_score', 0.0):.2f}`"
                    )
                else:
                    st.info(
                        f"**{br.get('branch_id')} · PRUNED** | 综合分 **{br.get('score', 0.0):.2f}**\n\n"
                        f"{br.get('hypothesis', '')}\n\n"
                        f"🎯 目标节点：`{readable_target}`\n\n"
                        f"📊 图谱分 `G={br.get('deterministic_graph_score', 0.0):.2f}` · "
                        f"语义分 `S={br.get('semantic_score', 0.0):.2f}`\n\n"
                        f"✂️ 剪枝依据：{br.get('prune_reason') or '综合证据弱于主假设'}"
                    )
        pruned_mems = reasoning_data.get("pruned_memories") or []
        if pruned_mems:
            with st.expander("🗜️ 查看 L2 剪枝分支状态折叠记忆 (CompactedMemory)", expanded=False):
                for mem in pruned_mems:
                    st.code(str(mem), language="text")

    # Explicit Graph-CoT Step-by-Step Auditable Cards (with readable entity/relation names!)
    steps = reasoning_data.get("steps") or []
    if steps:
        st.markdown("#### 🪜 分步推导与可读溯源凭证")
        for step in steps:
            ok = bool(step.get("grounded", False))
            badge = "🟢 PASS" if ok else "🔴 UNGROUNDED"
            title = f"{badge} | Step {step.get('step_id', '')} · `{step.get('stage', '')}`"
            with st.expander(title, expanded=True):
                st.write(step.get("claim", ""))
                e_cites = [ent_labels.get(x, x) for x in step.get("cited_entity_ids", [])]
                r_cites = [rel_labels.get(x, x) for x in step.get("cited_relation_ids", [])]
                c_cites = list(step.get("cited_chunk_ids", []))
                if e_cites:
                    st.markdown("**引用实体**： " + " · ".join(f"`{x}`" for x in e_cites))
                if r_cites:
                    st.markdown("**引用关系边**： " + " · ".join(f"`{x}`" for x in r_cites))
                if c_cites:
                    st.markdown("**引用原文切片**： " + " · ".join(f"`{x}`" for x in c_cites))
                notes = step.get("grounding_notes") or []
                if notes:
                    st.warning("审计备注：" + "；".join(str(n) for n in notes))


def _render_query_tab(pipeline: GraphRAGPipeline, cfg: AppConfig, cfg_rel: str) -> None:
    st.subheader("🔍 GraphRAG 实时问答、显性思维回放与子图追踪")
    question = st.text_input(
        "输入测试问题",
        value=_default_question_for_config(cfg_rel),
        key=f"query_input_{cfg_rel}",
    )

    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.0, 1.0, 1.0])
    mode = c1.selectbox("检索模式", options=["ppr", "hybrid", "local", "global"], index=0)
    reasoning = c2.selectbox(
        "显性思维范式",
        options=["cot", "tot", "direct"],
        index=0,
        help="cot=4步显性图谱思维链+GroundingAudit；tot=多假设分支查图+反证剪枝+L2折叠；direct=单轮直出",
    )
    enable_compact = c3.checkbox("启用 Compact", value=cfg.harness.enable_compact)
    k_hops = c4.slider("子图跳数", min_value=1, max_value=3, value=cfg.retrieval.k_hops)
    top_k_entities = c5.slider("种子上限", min_value=2, max_value=15, value=cfg.retrieval.top_k_entities)

    if st.button("🚀 执行 GraphRAG 检索与显性推理", type="primary"):
        cfg.retrieval.k_hops = k_hops
        cfg.retrieval.top_k_entities = top_k_entities
        with st.spinner(
            f"正在通过 `{cfg.env.llm_provider}` 执行 `mode={mode}` / `reasoning={reasoning}` ..."
        ):
            try:
                result = pipeline.query(
                    question,
                    mode=mode,
                    reasoning=reasoning,
                    enable_compact=enable_compact,
                    auto_load=False,
                )
                st.session_state["last_query_result"] = result
            except Exception as exc:
                st.error(f"查询失败：{type(exc).__name__}: {exc}")

    result = st.session_state.get("last_query_result")
    if result is None:
        st.caption(
            "点击上方按钮运行一次查询，下方将同步展示 Compact 压缩遥测、显性 CoT/ToT 思维回放、"
            "GroundingAudit 忠实度门禁、联动高亮子图与原文证据链。"
        )
        return

    st.markdown("### 💬 最终回答")
    st.success(result.answer)

    # Compact & Harness Telemetry Bar
    comp = result.metadata.get("compaction")
    ledger = result.metadata.get("harness_ledger")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("种子实体", len(result.trace.seed_entity_ids))
    m2.metric("子图节点", len(result.trace.expanded_entity_ids))
    m3.metric("子图关系", len(result.trace.relation_ids))
    if isinstance(comp, dict):
        m4.metric(
            "Compact Token",
            f"{comp.get('compacted_tokens_est', 0)} tok",
            delta=f"-{float(comp.get('compression_ratio', 0.0)) * 100:.1f}%",
        )
    else:
        m4.metric("Compact", "关闭")
    if isinstance(ledger, dict):
        m5.metric("LLM 调用", int(ledger.get("total_calls", 1)))
        m6.metric("总耗时", f"{float(ledger.get('total_wall_seconds', 0.0)):.2f}s")
    else:
        m5.metric("原文证据", len(result.evidences))
        m6.metric("模式", result.mode)

    # Render Explicit CoT / ToT Replay when available
    reasoning_data = result.metadata.get("reasoning")
    if isinstance(reasoning_data, dict):
        _render_reasoning_replay(reasoning_data, pipeline)

    st.markdown("### 🕸️ 检索与推理子图联动高亮")
    st.caption(
        "图例说明：**红色金边节点** = 命中的种子实体；**橙色节点** = 检索/PPR/ToT 扩展子图实体；"
        "**黄色粗边** = 推理子图关系。"
    )
    focus_trace_only = st.checkbox(
        "仅显示本次命中的子图节点（聚焦无干扰视角）",
        value=len(pipeline.graph_store.get_all_entities()) > 80,
    )
    allowed_trace_ids = set(result.trace.expanded_entity_ids) if focus_trace_only else None
    trace_html = render_pyvis_html(
        pipeline.graph_store,
        vector_store=pipeline.vector_store,
        color_by="type",
        trace=result.trace,
        allowed_entity_ids=allowed_trace_ids or None,
        height="560px",
        title=f"Query Subgraph Trace ({result.mode} | {result.metadata.get('reasoning_paradigm', 'direct')})",
    )
    _render_html_embed(trace_html, height=620)

    st.markdown("### 📎 引用原文证据链 (Provenance Evidences)")
    if not result.evidences:
        st.info("本次查询未附加原文切片证据。")
    for idx, ev in enumerate(result.evidences, start=1):
        with st.expander(
            f"证据 {idx} | {ev.chunk_id} | 来源: {ev.source} | 命中方式: {ev.matched_via} | 分数: {ev.score:.3f}",
            expanded=(idx <= 2),
        ):
            st.write(ev.text)


def _render_shacl_tab(pipeline: GraphRAGPipeline, cfg: AppConfig) -> None:
    st.subheader("🛡️ SHACL-lite 全量本体与溯源合规质检")
    report = validate_kg_shacl_lite(pipeline.graph_store, pipeline.vector_store, cfg)

    if report.overall_status == "PASS":
        st.success(
            f"🟢 全量质检通过（Profile: `{report.profile}`）· 合规率 **{report.compliance_score * 100:.1f}%** · "
            f"已扫描 `{report.total_entities_scanned}` 实体 / `{report.total_relations_scanned}` 关系 / "
            f"`{report.total_chunks_indexed}` 切片，0 Error / 0 Warning。"
        )
    elif report.overall_status == "WARN":
        st.warning(
            f"🟡 存在轻微告警（Profile: `{report.profile}`）· 合规率 **{report.compliance_score * 100:.1f}%** · "
            f"Error `{report.error_count}` / Warning `{report.warning_count}`。"
        )
    else:
        st.error(
            f"🔴 检测到本体合规错误（Profile: `{report.profile}`）· 合规率 **{report.compliance_score * 100:.1f}%** · "
            f"Error `{report.error_count}` / Warning `{report.warning_count}`。"
        )

    rows = [
        {
            "状态": "🟢 PASS" if c.passed else ("🔴 FAIL" if c.severity == "error" else "🟡 WARN"),
            "规则 ID": c.check_id,
            "质检项": c.title,
            "严重级": c.severity,
            "违例数": c.violation_count,
            "规则说明": c.description,
        }
        for c in report.checks
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    violations = [c for c in report.checks if not c.passed]
    if violations:
        st.markdown("#### 🔎 违例抽样明细")
        for chk in violations:
            with st.expander(f"{chk.check_id} | {chk.title} ({chk.violation_count} 处)", expanded=True):
                for item in chk.sample_violations:
                    st.markdown(f"- {item}")
    else:
        st.caption("全部规则 100% 通过，未发现悬空边、自环边、缺失溯源的幽灵三元组或本体域值域违例。")


def _render_entities_triples_tab(
    pipeline: GraphRAGPipeline,
    entity_records: list[dict[str, Any]],
    relation_records: list[dict[str, Any]],
) -> None:
    all_types = sorted({r["entity_type"] for r in entity_records})
    all_comms = sorted({r["community_id"] for r in entity_records})

    f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 1])
    keyword = f1.text_input("关键词搜索（实体名 / 别名 / 关系 / 描述）", value="")
    selected_types = f2.multiselect("实体类型", options=all_types, default=all_types, key="tbl_types")
    selected_comms = f3.multiselect("社区 ID", options=all_comms, default=all_comms, key="tbl_comms")
    isolated_only = f4.checkbox("仅看孤立实体", value=False)

    filtered_entities = filter_entities(
        entity_records,
        keyword=keyword,
        entity_types=selected_types,
        community_ids=selected_comms,
        isolated_only=isolated_only,
    )
    filtered_ids = {r["entity_id"] for r in filtered_entities}
    filtered_relations = filter_relations(
        relation_records,
        keyword=keyword,
        allowed_entity_ids=filtered_ids if (selected_types or selected_comms or isolated_only) else None,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader(f"实体列表 ({len(filtered_entities)})")
        if filtered_entities:
            df_ent = pd.DataFrame(filtered_entities)[
                ["name", "entity_type", "aliases", "community_id", "degree", "source_chunks_str"]
            ]
            st.dataframe(df_ent, width="stretch", hide_index=True)
        else:
            st.info("无匹配实体。")

    with col_right:
        st.subheader(f"关系三元组 ({len(filtered_relations)})")
        if filtered_relations:
            df_rel = pd.DataFrame(filtered_relations)[
                ["source_name", "relation_type", "target_name", "weight", "source_chunks_str"]
            ]
            st.dataframe(df_rel, width="stretch", hide_index=True)
        else:
            st.info("无匹配关系。")

    st.divider()
    st.subheader("🔎 原文溯源反查（点击选择任意实体或关系查看原始切片）")
    inspect_mode = st.radio("反查对象", options=["实体 (Entity)", "关系三元组 (Relation)"], horizontal=True)

    target_chunk_ids: list[str] = []
    if inspect_mode == "实体 (Entity)" and filtered_entities:
        options = {
            f"{r['name']} ({r['entity_type']} | deg={r['degree']})": r for r in filtered_entities
        }
        chosen_label = st.selectbox("选择实体", options=list(options.keys()))
        chosen = options[chosen_label]
        st.write(f"**描述**：{chosen['description'] or '（无）'}")
        if chosen["aliases"]:
            st.write(f"**归一化别名 (ER Aliases)**：`{chosen['aliases']}`")
        target_chunk_ids = chosen["source_chunk_ids"]
    elif inspect_mode == "关系三元组 (Relation)" and filtered_relations:
        options = {
            f"{r['source_name']} --[{r['relation_type']}]--> {r['target_name']}": r
            for r in filtered_relations
        }
        chosen_label = st.selectbox("选择关系三元组", options=list(options.keys()))
        chosen = options[chosen_label]
        st.write(f"**关系说明**：{chosen['description'] or '（无）'}")
        target_chunk_ids = chosen["source_chunk_ids"]

    chunks = resolve_provenance_chunks(target_chunk_ids, pipeline.vector_store)
    if chunks:
        for ch in chunks:
            with st.expander(
                f"📄 {ch.chunk_id} | 来源: {ch.source} | 字符区间: {ch.start_char}-{ch.end_char}",
                expanded=True,
            ):
                st.write(ch.text)
    else:
        st.caption("暂无可溯源的原文切片。")


def _render_communities_tab(community_records: list[dict[str, Any]]) -> None:
    st.subheader(f"🗂️ Louvain 社区聚类总览 ({len(community_records)})")
    if not community_records:
        st.info("当前图谱尚未生成社区信息。")
        return
    df_comm = pd.DataFrame(community_records)[
        ["community_id", "title", "entity_count", "relation_count", "members", "summary"]
    ]
    st.dataframe(df_comm, width="stretch", hide_index=True)


def _render_chunks_tab(chunk_records: list[dict[str, Any]]) -> None:
    st.subheader(f"📄 原文文档切片核验 ({len(chunk_records)})")
    if not chunk_records:
        st.info("当前存储中没有文档切片。")
        return
    sources = sorted({r["source"] for r in chunk_records})
    selected_sources = st.multiselect("按源文件筛选", options=sources, default=sources)
    visible = [r for r in chunk_records if r["source"] in selected_sources]

    df_chunks = pd.DataFrame(visible)[["chunk_id", "source", "index", "char_span", "length"]]
    st.dataframe(df_chunks, width="stretch", hide_index=True)

    for row in visible[:50]:
        with st.expander(
            f"{row['chunk_id']} | {row['source']} | 字符 {row['char_span']} ({row['length']} chars)"
        ):
            st.write(row["text"])


def main() -> None:
    _init_page()
    cfg, _input_dir, storage_dir, selected_cfg_rel = _render_sidebar()
    pipeline, doc_count, err = _load_pipeline_state(cfg, storage_dir)

    if pipeline is None:
        st.warning(
            f"尚未在 `{storage_dir}` 中找到已构建的知识图谱（{err}）。"
            "请先在终端执行构建命令或点击左侧栏「▶️ 重新构建知识图谱」。"
        )
        return

    _render_exports_sidebar(pipeline, cfg, doc_count)
    _render_kpi_bar(pipeline, cfg, doc_count)

    entity_records = build_entity_records(pipeline.graph_store)
    relation_records = build_relation_records(pipeline.graph_store)
    community_records = build_community_records(pipeline.graph_store)
    chunk_records = build_chunk_records(pipeline.vector_store)

    tab_graph, tab_query, tab_shacl, tab_kg, tab_comm, tab_chunks = st.tabs(
        [
            "📊 交互图谱 & Ego-Graph",
            "🔍 问答与思维回放",
            "🛡️ SHACL-lite 本体质检",
            "🧩 实体与三元组溯源",
            "🗂️ 社区聚类",
            "📄 原文切片",
        ]
    )

    with tab_graph:
        _render_graph_tab(pipeline, entity_records, relation_records)
    with tab_query:
        _render_query_tab(pipeline, cfg, selected_cfg_rel)
    with tab_shacl:
        _render_shacl_tab(pipeline, cfg)
    with tab_kg:
        _render_entities_triples_tab(pipeline, entity_records, relation_records)
    with tab_comm:
        _render_communities_tab(community_records)
    with tab_chunks:
        _render_chunks_tab(chunk_records)


if __name__ == "__main__":
    main()

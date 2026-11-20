"""Streamlit visual verification workbench for LLM + GraphRAG + KG (Phase 2).

Launch via:
    uv run --extra viz llm-graphrag-kg ui
or:
    uv run --extra viz streamlit run src/llm_graphrag_kg/viz/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from llm_graphrag_kg.config import AppConfig, load_config
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.factory import create_llm_client
from llm_graphrag_kg.viz.audit_report import compute_audit_stats, export_audit_artifacts
from llm_graphrag_kg.viz.graphml_exporter import export_graphml
from llm_graphrag_kg.viz.html_exporter import render_pyvis_html
from llm_graphrag_kg.viz.workbench_data import (
    build_chunk_records,
    build_community_records,
    build_entity_records,
    build_relation_records,
    filter_entities,
    filter_relations,
    resolve_provenance_chunks,
)


def _render_html_embed(html: str, *, height: int) -> None:
    """Embed interactive PyVis HTML using `st.iframe` (Streamlit >=1.5x) with legacy fallback."""
    if hasattr(st, "iframe"):
        st.iframe(html, width="stretch", height=height)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=height, scrolling=True)


def _init_page() -> None:
    st.set_page_config(
        page_title="LLM-GraphRAG-KG 可视化核验工作台",
        page_icon="🕸️",
        layout="wide",
    )
    st.title("🕸️ LLM + GraphRAG + KG 可视化内容核验工作台")
    st.caption(
        "一站式核验：文档切片 → 实体与三元组溯源 → Louvain 社区聚类 → GraphRAG 问答与检索子图高亮"
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
    return found or ["configs/default.yaml"]


def _render_sidebar() -> tuple[AppConfig, Path, Path, str | None]:
    with st.sidebar:
        st.header("⚙️ 运行配置与操作")

        config_files = _discover_config_files()
        selected_cfg_rel = st.selectbox(
            "场景配置文件 (YAML)",
            options=config_files,
            index=0,
            help="可在默认示例与故障诊断领域配置（configs/fault_diagnosis.yaml）间一键切换。",
        )
        selected_cfg_path = Path.cwd() / selected_cfg_rel

        provider_choice = st.selectbox(
            "LLM Provider",
            options=["default (.env)", "openai_compat", "mock"],
            index=0,
            help="默认读取本地 .env；选择 mock 可离线零费用调试。",
        )
        provider_override = None if provider_choice == "default (.env)" else provider_choice

        cfg = load_config(
            config_path=selected_cfg_path,
            llm_provider_override=provider_override,
        )
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

    return cfg, input_dir, storage_dir, provider_override


def _render_exports_sidebar(pipeline: GraphRAGPipeline, doc_count: int) -> None:
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


def _render_kpi_bar(pipeline: GraphRAGPipeline, doc_count: int) -> None:
    stats = compute_audit_stats(pipeline.graph_store, pipeline.vector_store, doc_count=doc_count)
    cols = st.columns(7)
    cols[0].metric("文档数", stats.doc_count)
    cols[1].metric("切片数", stats.chunk_count)
    cols[2].metric("实体数", stats.entity_count)
    cols[3].metric("关系数", stats.relation_count)
    cols[4].metric("社区数", stats.community_count)
    cols[5].metric("连通分量", stats.connected_components)
    cols[6].metric("孤立实体", len(stats.isolated_entities))

    if stats.isolated_entities:
        st.info(
            f"检测到 {len(stats.isolated_entities)} 个孤立实体（"
            f"{', '.join(stats.isolated_entities[:8])}），可在「实体与三元组核验」页勾选"
            "「仅看孤立实体」检查是否需要优化抽取 Prompt 或合并同义别名。"
        )


def _render_graph_tab(pipeline: GraphRAGPipeline, entity_records: list[dict]) -> None:
    all_types = sorted({r["entity_type"] for r in entity_records})
    all_comms = sorted({r["community_id"] for r in entity_records})

    c1, c2, c3, c4 = st.columns([1.2, 1.8, 1.8, 1.0])
    color_by = c1.radio("节点着色方式", options=["type", "community"], horizontal=True)
    selected_types = c2.multiselect("筛选实体类型", options=all_types, default=all_types)
    selected_comms = c3.multiselect("筛选社区 ID", options=all_comms, default=all_comms)
    physics = c4.checkbox("物理布局", value=True)

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


def _render_query_tab(pipeline: GraphRAGPipeline, cfg: AppConfig) -> None:
    st.subheader("🔍 GraphRAG 实时问答与检索子图追踪")
    question = st.text_input(
        "输入测试问题",
        value="GraphRAG 如何结合知识图谱解决多跳推理并支持迁移至 Neo4j？",
    )

    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox("检索模式", options=["hybrid", "local", "global"], index=0)
    k_hops = c2.slider("子图扩展跳数 (k-hop)", min_value=0, max_value=3, value=cfg.retrieval.k_hops)
    top_k_entities = c3.slider(
        "种子实体上限", min_value=1, max_value=15, value=cfg.retrieval.top_k_entities
    )
    top_k_chunks = c4.slider(
        "向量召回切片数", min_value=1, max_value=10, value=cfg.retrieval.top_k_chunks
    )

    if st.button("🚀 执行 GraphRAG 检索与问答", type="primary"):
        cfg.retrieval.k_hops = k_hops
        cfg.retrieval.top_k_entities = top_k_entities
        cfg.retrieval.top_k_chunks = top_k_chunks
        with st.spinner(f"正在通过 `{cfg.env.llm_provider}` 执行 {mode} 模式检索与生成..."):
            try:
                result = pipeline.query(question, mode=mode, auto_load=False)
                st.session_state["last_query_result"] = result
            except Exception as exc:
                st.error(f"查询失败：{type(exc).__name__}: {exc}")

    result = st.session_state.get("last_query_result")
    if result is None:
        st.caption("点击上方按钮运行一次查询，下方将同步展示生成回答、命中子图高亮图谱与原文证据链。")
        return

    st.markdown("### 💬 模型回答")
    st.success(result.answer)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("种子实体", len(result.trace.seed_entity_ids))
    m2.metric("扩展子图节点", len(result.trace.expanded_entity_ids))
    m3.metric("命中关系边", len(result.trace.relation_ids))
    m4.metric("原文证据条数", len(result.evidences))

    st.caption(
        "图例说明：**红色金边节点** = 命中的种子实体；**橙色节点** = k-hop 扩展邻域实体；"
        "**黄色粗边** = 检索子图关系；灰色节点为未命中背景。"
    )
    trace_html = render_pyvis_html(
        pipeline.graph_store,
        vector_store=pipeline.vector_store,
        color_by="community",
        trace=result.trace,
        height="580px",
        title=f"Query Subgraph Trace ({result.mode}): {result.question[:60]}",
    )
    _render_html_embed(trace_html, height=640)

    st.markdown("### 📎 引用原文证据链 (Provenance Evidences)")
    if not result.evidences:
        st.info("本次查询未附加原文切片证据（例如纯 Global 社区摘要模式）。")
    for idx, ev in enumerate(result.evidences, start=1):
        with st.expander(
            f"证据 {idx} | {ev.chunk_id} | 来源: {ev.source} | 命中方式: {ev.matched_via} | 分数: {ev.score:.3f}",
            expanded=(idx <= 2),
        ):
            st.write(ev.text)


def _render_entities_triples_tab(
    pipeline: GraphRAGPipeline,
    entity_records: list[dict],
    relation_records: list[dict],
) -> None:
    all_types = sorted({r["entity_type"] for r in entity_records})
    all_comms = sorted({r["community_id"] for r in entity_records})

    f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 1])
    keyword = f1.text_input("关键词搜索（实体名 / 关系 / 描述）", value="")
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
                ["name", "entity_type", "community_id", "degree", "source_chunks_str", "description"]
            ]
            st.dataframe(df_ent, width="stretch", hide_index=True)
        else:
            st.info("无匹配实体。")

    with col_right:
        st.subheader(f"关系三元组 ({len(filtered_relations)})")
        if filtered_relations:
            df_rel = pd.DataFrame(filtered_relations)[
                ["source_name", "relation_type", "target_name", "weight", "source_chunks_str", "description"]
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


def _render_communities_tab(community_records: list[dict]) -> None:
    st.subheader(f"🗂️ Louvain 社区聚类总览 ({len(community_records)})")
    if not community_records:
        st.info("当前图谱尚未生成社区信息。")
        return
    df_comm = pd.DataFrame(community_records)[
        ["community_id", "title", "entity_count", "relation_count", "members", "summary"]
    ]
    st.dataframe(df_comm, width="stretch", hide_index=True)


def _render_chunks_tab(chunk_records: list[dict]) -> None:
    st.subheader(f"📄 原文文档切片核验 ({len(chunk_records)})")
    if not chunk_records:
        st.info("当前存储中没有文档切片。")
        return
    sources = sorted({r["source"] for r in chunk_records})
    selected_sources = st.multiselect("按源文件筛选", options=sources, default=sources)
    visible = [r for r in chunk_records if r["source"] in selected_sources]

    df_chunks = pd.DataFrame(visible)[["chunk_id", "source", "index", "char_span", "length"]]
    st.dataframe(df_chunks, width="stretch", hide_index=True)

    for row in visible:
        with st.expander(
            f"{row['chunk_id']} | {row['source']} | 字符 {row['char_span']} ({row['length']} chars)"
        ):
            st.write(row["text"])


def main() -> None:
    _init_page()
    cfg, _input_dir, storage_dir, _provider_override = _render_sidebar()
    pipeline, doc_count, err = _load_pipeline_state(cfg, storage_dir)

    if pipeline is None:
        st.warning(
            f"尚未在 `{storage_dir}` 中找到已构建的知识图谱（{err}）。"
            "请点击左侧栏的「▶️ 重新构建知识图谱」按钮先执行离线建图。"
        )
        return

    _render_exports_sidebar(pipeline, doc_count)
    _render_kpi_bar(pipeline, doc_count)

    entity_records = build_entity_records(pipeline.graph_store)
    relation_records = build_relation_records(pipeline.graph_store)
    community_records = build_community_records(pipeline.graph_store)
    chunk_records = build_chunk_records(pipeline.vector_store)

    tab_graph, tab_query, tab_kg, tab_comm, tab_chunks = st.tabs(
        [
            "📊 交互图谱",
            "🔍 问答与子图追踪",
            "🧩 实体与三元组溯源",
            "🗂️ 社区聚类",
            "📄 原文切片",
        ]
    )

    with tab_graph:
        _render_graph_tab(pipeline, entity_records)
    with tab_query:
        _render_query_tab(pipeline, cfg)
    with tab_kg:
        _render_entities_triples_tab(pipeline, entity_records, relation_records)
    with tab_comm:
        _render_communities_tab(community_records)
    with tab_chunks:
        _render_chunks_tab(chunk_records)


if __name__ == "__main__":
    main()

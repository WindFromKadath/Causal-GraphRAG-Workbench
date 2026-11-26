"""Knowledge Graph quality metrics, Markdown audit report, and CSV table exports."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import networkx as nx

from llm_graphrag_kg.core.interfaces import BaseGraphStore, BaseVectorStore
from llm_graphrag_kg.core.schema import GraphAuditStats


def compute_audit_stats(
    graph_store: BaseGraphStore,
    vector_store: BaseVectorStore | None = None,
    *,
    doc_count: int | None = None,
) -> GraphAuditStats:
    """Compute structural health metrics for quick manual inspection."""
    entities = graph_store.get_all_entities()
    relations = graph_store.get_all_relations()
    communities = graph_store.get_all_communities()
    chunks = vector_store.get_all_chunks() if vector_store is not None else []
    resolved_docs = doc_count if doc_count is not None else len({c.doc_id for c in chunks})

    nx_graph = graph_store.to_networkx()
    undirected = nx_graph.to_undirected()

    isolated = [
        str(nx_graph.nodes[nid].get("name", nid))
        for nid in nx.isolates(nx_graph)
    ]
    components = nx.number_connected_components(undirected) if undirected.number_of_nodes() else 0

    type_counts = dict(Counter(e.entity_type for e in entities).most_common())
    degrees = sorted(
        ((str(nx_graph.nodes[nid].get("name", nid)), int(deg)) for nid, deg in undirected.degree()),
        key=lambda item: (-item[1], item[0]),
    )

    hyperedges = (
        graph_store.get_all_hyperedges() if hasattr(graph_store, "get_all_hyperedges") else []
    )
    return GraphAuditStats(
        doc_count=resolved_docs,
        chunk_count=len(chunks),
        entity_count=len(entities),
        relation_count=len(relations),
        community_count=len(communities),
        hyperedge_count=len(hyperedges),
        isolated_entities=sorted(isolated),
        connected_components=components,
        entity_type_counts=type_counts,
        top_degree_entities=degrees[:10],
    )


def export_audit_artifacts(
    graph_store: BaseGraphStore,
    vector_store: BaseVectorStore | None,
    output_dir: Path,
    *,
    doc_count: int | None = None,
) -> tuple[GraphAuditStats, dict[str, Path]]:
    """Write Markdown audit summary + entities.csv + relations.csv for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_audit_stats(graph_store, vector_store, doc_count=doc_count)

    # 1. entities.csv
    entities_csv = output_dir / "entities.csv"
    with entities_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["entity_id", "name", "entity_type", "community_id", "source_chunks", "description"]
        )
        for ent in sorted(graph_store.get_all_entities(), key=lambda e: e.name):
            writer.writerow(
                [
                    ent.entity_id,
                    ent.name,
                    ent.entity_type,
                    "" if ent.community_id is None else ent.community_id,
                    "; ".join(ent.source_chunk_ids),
                    ent.description,
                ]
            )

    # 2. relations.csv
    entity_names = {e.entity_id: e.name for e in graph_store.get_all_entities()}
    relations_csv = output_dir / "relations.csv"
    with relations_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "relation_id",
                "source_name",
                "relation_type",
                "target_name",
                "weight",
                "source_chunks",
                "description",
            ]
        )
        for rel in sorted(graph_store.get_all_relations(), key=lambda r: (-r.weight, r.relation_id)):
            writer.writerow(
                [
                    rel.relation_id,
                    entity_names.get(rel.source_id, rel.source_id),
                    rel.relation_type,
                    entity_names.get(rel.target_id, rel.target_id),
                    f"{rel.weight:.2f}",
                    "; ".join(rel.source_chunk_ids),
                    rel.description,
                ]
            )

    # 3. audit_report.md
    report_md = output_dir / "audit_report.md"
    lines: list[str] = [
        "# 知识图谱质量核验报告 (KG Audit Report)",
        "",
        "## 1. 核心统计概览",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 文档数 (Documents) | {stats.doc_count} |",
        f"| 切片数 (Chunks) | {stats.chunk_count} |",
        f"| 实体数 (Entities) | {stats.entity_count} |",
        f"| 关系数 (Relations) | {stats.relation_count} |",
        f"| 社区数 (Communities) | {stats.community_count} |",
        f"| 连通分量数 (Connected Components) | {stats.connected_components} |",
        f"| 孤立实体数 (Isolated Entities) | {len(stats.isolated_entities)} |",
        "",
        "## 2. 实体类型分布",
        "",
        "| 实体类型 | 数量 |",
        "|---|---|",
    ]
    for etype, count in stats.entity_type_counts.items():
        lines.append(f"| {etype} | {count} |")

    lines.extend(["", "## 3. 核心枢纽实体 (Top Degree)", "", "| 实体名称 | 度 (Degree) |", "|---|---|"])
    for name, deg in stats.top_degree_entities:
        lines.append(f"| {name} | {deg} |")

    lines.extend(["", "## 4. 孤立实体核验 (Isolated Entities)", ""])
    if stats.isolated_entities:
        for name in stats.isolated_entities:
            lines.append(f"- {name}")
    else:
        lines.append("- 无孤立实体（所有实体均至少存在一条连接关系）。")

    communities = graph_store.get_all_communities()
    if communities:
        lines.extend(["", "## 5. 社区发现概览 (Communities)", ""])
        for comm in communities:
            lines.append(
                f"- **{comm.title}**（{len(comm.entity_ids)} 个实体，"
                f"{comm.relation_count} 条内部边）：{comm.summary}"
            )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats, {
        "audit_report": report_md,
        "entities_csv": entities_csv,
        "relations_csv": relations_csv,
    }

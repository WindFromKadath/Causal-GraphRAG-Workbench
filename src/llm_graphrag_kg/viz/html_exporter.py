"""Interactive single-file HTML Knowledge Graph exporter using PyVis."""

from __future__ import annotations

from html import escape
from pathlib import Path

from pyvis.network import Network

from llm_graphrag_kg.core.interfaces import BaseGraphStore, BaseVectorStore
from llm_graphrag_kg.core.schema import SubgraphTrace

TYPE_PALETTE: dict[str, str] = {
    "Concept": "#4C78A8",
    "Technology": "#F58518",
    "Method": "#54A24B",
    "Component": "#B279A2",
    "Organization": "#E45756",
    "Person": "#72B7B2",
    "Document": "#EECA3B",
    "Metric": "#FF9DA6",
    # Industrial fault diagnosis & FailureSensorIQ ontology types
    "Equipment": "#3B82F6",
    "FailureMode": "#EF4444",
    "FailureCategory": "#F97316",
    "SignalChannel": "#10B981",
    "Sensor": "#10B981",
    "Symptom": "#F59E0B",
    "RootCause": "#DC2626",
    "Algorithm": "#8B5CF6",
    "DeploymentComponent": "#06B6D4",
    "RemediationSOP": "#22C55E",
    "Standard": "#EAB308",
    # GraphRAG-Bench Medical & Novel ontology types
    "DiseaseConcept": "#EF4444",
    "Disease": "#DC2626",
    "RiskFactor": "#F59E0B",
    "DiagnosticTest": "#06B6D4",
    "Treatment": "#10B981",
    "Anatomy": "#8B5CF6",
    "Biomarker": "#EC4899",
    "Guideline": "#3B82F6",
    "NarrativeEntity": "#6366F1",
    "Location": "#14B8A6",
    "Event": "#F97316",
}

COMMUNITY_PALETTE: list[str] = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#EECA3B",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def _pick_color(entity_type: str, community_id: int | None, color_by: str) -> str:
    if color_by == "community" and community_id is not None and community_id >= 0:
        return COMMUNITY_PALETTE[community_id % len(COMMUNITY_PALETTE)]
    return TYPE_PALETTE.get(entity_type, "#6C8EBF")


def render_pyvis_html(
    graph_store: BaseGraphStore,
    *,
    vector_store: BaseVectorStore | None = None,
    color_by: str = "type",
    physics_enabled: bool = True,
    trace: SubgraphTrace | None = None,
    allowed_entity_ids: set[str] | None = None,
    height: str = "780px",
    title: str = "LLM-GraphRAG-KG Interactive Verification View",
) -> str:
    """Render an interactive Vis.js HTML string with provenance tooltips and optional trace highlight."""
    nx_graph = graph_store.to_networkx()
    undirected = nx_graph.to_undirected()

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#111827",
        font_color="#F9FAFB",
        notebook=False,
    )
    net.toggle_physics(physics_enabled)

    seed_set = set(trace.seed_entity_ids) if trace else set()
    expanded_set = set(trace.expanded_entity_ids) if trace else set()
    active_rel_set = set(trace.relation_ids) if trace else set()
    has_trace = bool(seed_set or expanded_set)

    entities = graph_store.get_all_entities()
    if allowed_entity_ids is not None:
        entities = [e for e in entities if e.entity_id in allowed_entity_ids]
    included_ids = {e.entity_id for e in entities}

    for entity in entities:
        degree = int(undirected.degree(entity.entity_id)) if entity.entity_id in undirected else 0
        size = 16 + min(degree * 4, 28)

        base_color = _pick_color(entity.entity_type, entity.community_id, color_by)
        border_width = 2
        border_color = "#E5E7EB"

        if has_trace:
            if entity.entity_id in seed_set:
                color_spec: str | dict[str, str] = {
                    "background": "#EF4444",
                    "border": "#FDE047",
                }
                border_width = 4
                size += 8
            elif entity.entity_id in expanded_set:
                color_spec = {
                    "background": "#F59E0B",
                    "border": "#FEF08A",
                }
                border_width = 3
                size += 4
            else:
                color_spec = {
                    "background": "#374151",
                    "border": "#4B5563",
                }
                border_width = 1
        else:
            color_spec = {"background": base_color, "border": border_color}

        chunk_previews: list[str] = []
        if vector_store is not None:
            for cid in entity.source_chunk_ids[:2]:
                chunk = vector_store.get_chunk(cid)
                if chunk is not None:
                    snippet = chunk.text.replace("\n", " ")[:120]
                    chunk_previews.append(f"[{cid}] {snippet}...")

        tooltip_lines = [
            f"Entity: {entity.name}",
            f"Type: {entity.entity_type} | Community: {entity.community_id}",
            f"Degree: {degree}",
        ]
        if entity.description:
            tooltip_lines.append(f"Description: {entity.description}")
        if entity.source_chunk_ids:
            tooltip_lines.append(f"Chunks: {', '.join(entity.source_chunk_ids[:4])}")
        if chunk_previews:
            tooltip_lines.append("Provenance:\n" + "\n".join(chunk_previews))

        net.add_node(
            entity.entity_id,
            label=entity.name,
            title="\n".join(tooltip_lines),
            color=color_spec,
            size=size,
            borderWidth=border_width,
        )

    rendered_edges = 0
    for rel in graph_store.get_all_relations():
        if rel.source_id not in included_ids or rel.target_id not in included_ids:
            continue

        is_active = rel.relation_id in active_rel_set
        edge_color = "#FDE047" if is_active else ("#374151" if has_trace else "#9CA3AF")
        width = 3.5 if is_active else (1.0 if has_trace else min(1.5 + rel.weight * 0.6, 4.0))

        edge_tooltip = "\n".join(
            [
                f"Relation: {rel.relation_type} (weight={rel.weight:.1f})",
                f"Description: {rel.description}" if rel.description else "",
                f"Chunks: {', '.join(rel.source_chunk_ids[:4])}",
            ]
        ).strip()

        net.add_edge(
            rel.source_id,
            rel.target_id,
            label=rel.relation_type,
            title=edge_tooltip,
            color=edge_color,
            width=width,
            arrows="to",
        )
        rendered_edges += 1

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "edges": {
            "font": { "size": 11, "color": "#D1D5DB", "strokeWidth": 0 },
            "smooth": { "type": "dynamic" }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3200,
              "centralGravity": 0.25,
              "springLength": 140
            }
          }
        }
        """
    )

    html = net.generate_html(notebook=False)
    safe_title = escape(title)
    banner = (
        f'<div style="font-family:sans-serif;padding:10px 16px;background:#0F172A;color:#F8FAFC;'
        f'border-bottom:1px solid #334155;">'
        f"<strong>{safe_title}</strong> "
        f'<span style="color:#94A3B8;font-size:13px;margin-left:12px;">'
        f"ColorBy={escape(color_by)} | Nodes={len(entities)} | Edges={rendered_edges}"
        + (
            f" | Trace Highlight: Seeds={len(seed_set)}, Subgraph={len(expanded_set)}"
            if has_trace
            else ""
        )
        + "</span></div>"
    )
    return html.replace("<body>", f"<body>\n{banner}", 1)


def export_pyvis_html(
    graph_store: BaseGraphStore,
    output_path: Path,
    *,
    vector_store: BaseVectorStore | None = None,
    color_by: str = "type",
    physics_enabled: bool = True,
    trace: SubgraphTrace | None = None,
    allowed_entity_ids: set[str] | None = None,
    height: str = "780px",
    title: str = "LLM-GraphRAG-KG Interactive Verification View",
) -> Path:
    """Write interactive Vis.js HTML graph to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_pyvis_html(
        graph_store,
        vector_store=vector_store,
        color_by=color_by,
        physics_enabled=physics_enabled,
        trace=trace,
        allowed_entity_ids=allowed_entity_ids,
        height=height,
        title=title,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path

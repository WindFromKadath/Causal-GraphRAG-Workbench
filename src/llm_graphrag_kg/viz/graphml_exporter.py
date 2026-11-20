"""GraphML exporter compatible with Gephi and Cytoscape."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from llm_graphrag_kg.core.interfaces import BaseGraphStore


def export_graphml(graph_store: BaseGraphStore, output_path: Path) -> Path:
    """Export the KG to standard `.graphml` by flattening complex attributes to strings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = graph_store.to_networkx()
    clean = nx.DiGraph()

    for node_id, data in raw.nodes(data=True):
        attrs = {}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                attrs[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                attrs[key] = ""
            else:
                attrs[key] = value
        clean.add_node(node_id, **attrs)

    for src, tgt, data in raw.edges(data=True):
        attrs = {}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                attrs[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                attrs[key] = ""
            else:
                attrs[key] = value
        clean.add_edge(src, tgt, **attrs)

    nx.write_graphml(clean, output_path)
    return output_path

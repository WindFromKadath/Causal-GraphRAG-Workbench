"""Pure-Python data preparation, filtering, and provenance helpers for the Streamlit workbench."""

from __future__ import annotations

from typing import Any

from llm_graphrag_kg.core.interfaces import BaseGraphStore, BaseVectorStore
from llm_graphrag_kg.core.schema import TextChunk


def build_chunk_records(vector_store: BaseVectorStore) -> list[dict[str, Any]]:
    """Build tabular chunk records for slice-level inspection."""
    chunks = sorted(
        vector_store.get_all_chunks(),
        key=lambda c: (c.source, c.index, c.chunk_id),
    )
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source": c.source,
            "index": c.index,
            "char_span": f"{c.start_char}-{c.end_char}",
            "length": len(c.text),
            "text": c.text,
        }
        for c in chunks
    ]


def build_entity_records(graph_store: BaseGraphStore) -> list[dict[str, Any]]:
    """Build tabular entity records enriched with graph degree."""
    nx_graph = graph_store.to_networkx()
    undirected = nx_graph.to_undirected()
    entities = sorted(graph_store.get_all_entities(), key=lambda e: (e.entity_type, e.name))

    records: list[dict[str, Any]] = []
    for ent in entities:
        degree = int(undirected.degree(ent.entity_id)) if ent.entity_id in undirected else 0
        records.append(
            {
                "entity_id": ent.entity_id,
                "name": ent.name,
                "entity_type": ent.entity_type,
                "community_id": ent.community_id if ent.community_id is not None else -1,
                "degree": degree,
                "is_isolated": degree == 0,
                "aliases": ", ".join(ent.aliases),
                "source_chunk_ids": list(ent.source_chunk_ids),
                "source_chunks_str": ", ".join(ent.source_chunk_ids),
                "description": ent.description,
            }
        )
    return records


def build_relation_records(graph_store: BaseGraphStore) -> list[dict[str, Any]]:
    """Build tabular relation triple records with resolved endpoint names."""
    entity_map = {e.entity_id: e for e in graph_store.get_all_entities()}
    relations = sorted(
        graph_store.get_all_relations(),
        key=lambda r: (-r.weight, r.relation_type, r.relation_id),
    )

    records: list[dict[str, Any]] = []
    for rel in relations:
        src = entity_map.get(rel.source_id)
        tgt = entity_map.get(rel.target_id)
        records.append(
            {
                "relation_id": rel.relation_id,
                "source_id": rel.source_id,
                "source_name": src.name if src else rel.source_id,
                "relation_type": rel.relation_type,
                "target_id": rel.target_id,
                "target_name": tgt.name if tgt else rel.target_id,
                "weight": round(rel.weight, 2),
                "source_chunk_ids": list(rel.source_chunk_ids),
                "source_chunks_str": ", ".join(rel.source_chunk_ids),
                "description": rel.description,
            }
        )
    return records


def build_community_records(graph_store: BaseGraphStore) -> list[dict[str, Any]]:
    """Build tabular community records with readable member names."""
    entity_names = {e.entity_id: e.name for e in graph_store.get_all_entities()}
    records: list[dict[str, Any]] = []
    for comm in graph_store.get_all_communities():
        member_names = [entity_names.get(eid, eid) for eid in comm.entity_ids]
        records.append(
            {
                "community_id": comm.community_id,
                "title": comm.title,
                "entity_count": len(comm.entity_ids),
                "relation_count": comm.relation_count,
                "members": ", ".join(member_names),
                "summary": comm.summary,
            }
        )
    return records


def filter_entities(
    records: list[dict[str, Any]],
    *,
    keyword: str = "",
    entity_types: list[str] | None = None,
    community_ids: list[int] | None = None,
    isolated_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter entity records by keyword, entity types, community IDs, and isolation status."""
    kw = keyword.strip().lower()
    type_set = set(entity_types) if entity_types else None
    comm_set = set(community_ids) if community_ids else None

    result: list[dict[str, Any]] = []
    for row in records:
        if type_set is not None and row["entity_type"] not in type_set:
            continue
        if comm_set is not None and row["community_id"] not in comm_set:
            continue
        if isolated_only and not row["is_isolated"]:
            continue
        if kw:
            haystack = f"{row['name']} {row['aliases']} {row['description']} {row['entity_type']}".lower()
            if kw not in haystack:
                continue
        result.append(row)
    return result


def filter_relations(
    records: list[dict[str, Any]],
    *,
    keyword: str = "",
    allowed_entity_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter relation triple records by keyword and/or allowed endpoint entity IDs."""
    kw = keyword.strip().lower()
    result: list[dict[str, Any]] = []
    for row in records:
        if allowed_entity_ids is not None:
            if row["source_id"] not in allowed_entity_ids and row["target_id"] not in allowed_entity_ids:
                continue
        if kw:
            haystack = (
                f"{row['source_name']} {row['relation_type']} {row['target_name']} {row['description']}"
            ).lower()
            if kw not in haystack:
                continue
        result.append(row)
    return result


def resolve_provenance_chunks(
    chunk_ids: list[str],
    vector_store: BaseVectorStore,
) -> list[TextChunk]:
    """Resolve provenance chunk IDs back to full `TextChunk` records preserving order."""
    chunks: list[TextChunk] = []
    seen: set[str] = set()
    for cid in chunk_ids:
        if cid in seen:
            continue
        seen.add(cid)
        chunk = vector_store.get_chunk(cid)
        if chunk is not None:
            chunks.append(chunk)
    return chunks

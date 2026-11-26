"""Lightweight HyperGraph multi-condition modeling & matching (Stage 3.4, HyperGraphRAG).

Models N-ary joint inference rules such as:
    {2倍频振动升高, 轴承温升异常, 电机定子电流波动} ===[JOINT_INDICATES]===> 轴系不对中
on top of any `BaseGraphStore` using a clean bipartite-compatible representation that migrates
directly to Neo4j HyperNode / N-ary relationship patterns in production.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.interfaces import BaseGraphStore
from llm_graphrag_kg.core.schema import Entity, HyperEdge

# Entity types that naturally act as joint evidence conditions vs diagnosis/outcome targets
CONDITION_ENTITY_TYPES: set[str] = {
    "Symptom",
    "SignalChannel",
    "Sensor",
    "RiskFactor",
    "DiagnosticTest",
    "Biomarker",
    "FailureConcept",
    "Component",
}

TARGET_ENTITY_TYPES: set[str] = {
    "FaultMode",
    "RootCause",
    "FailureMode",
    "FailureModeCode",
    "DiseaseConcept",
    "Disease",
    "RemediationSOP",
    "Treatment",
}


class HyperEdgeMatch(BaseModel):
    """Ranked match between active query entities and an N-ary HyperEdge."""

    hyperedge_id: str
    relation_type: str
    target_entity_id: str
    target_entity_name: str
    matched_condition_ids: list[str] = Field(default_factory=list)
    matched_condition_names: list[str] = Field(default_factory=list)
    missing_condition_ids: list[str] = Field(default_factory=list)
    missing_condition_names: list[str] = Field(default_factory=list)
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    joint_score: float = Field(ge=0.0)
    source_chunk_ids: list[str] = Field(default_factory=list)
    description: str = ""


def make_hyperedge_id(
    condition_entity_ids: list[str],
    relation_type: str,
    target_entity_id: str,
) -> str:
    """Order-invariant deterministic ID for an N-ary hyperedge."""
    canon_conds = ",".join(sorted(set(condition_entity_ids)))
    canonical = f"{canon_conds}|{relation_type.strip().upper()}|{target_entity_id}"
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"he_{digest}"


def create_hyperedge(
    condition_entity_ids: list[str],
    target_entity_id: str,
    *,
    relation_type: str = "JOINT_INDICATES",
    description: str = "",
    confidence: float = 1.0,
    weight: float = 1.0,
    source_chunk_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> HyperEdge:
    """Construct a canonical N-ary HyperEdge."""
    uniq_conds = sorted(dict.fromkeys(c for c in condition_entity_ids if c != target_entity_id))
    rel_norm = relation_type.strip().upper() or "JOINT_INDICATES"
    hid = make_hyperedge_id(uniq_conds, rel_norm, target_entity_id)
    return HyperEdge(
        hyperedge_id=hid,
        condition_entity_ids=uniq_conds,
        target_entity_id=target_entity_id,
        relation_type=rel_norm,
        description=description,
        confidence=max(0.0, min(1.0, confidence)),
        weight=weight,
        source_chunk_ids=list(dict.fromkeys(source_chunk_ids or [])),
        metadata=dict(metadata or {}),
    )


def mine_joint_hyperedges(
    graph_store: BaseGraphStore,
    *,
    min_conditions: int = 2,
    max_conditions: int = 5,
    max_hyperedges: int = 64,
) -> list[HyperEdge]:
    """Deterministically mine multi-condition N-ary hyperedges from grounded KG co-occurrence.

    A joint hyperedge `{c_1, c_2, ...} ===[JOINT_INDICATES]===> target` is mined whenever
    `>= min_conditions` distinct condition-type entities share at least one evidence chunk with
    a target diagnosis/root-cause entity AND are directly connected to that target (or share a
    1-hop neighborhood) in the knowledge graph.
    """
    entities = graph_store.get_all_entities()
    relations = graph_store.get_all_relations()
    if not entities or not relations:
        return []

    ent_map = {e.entity_id: e for e in entities}

    # Build 1-hop adjacency and per-pair shared chunk lookup
    neighbors: dict[str, set[str]] = defaultdict(set)
    edge_chunks: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rel in relations:
        if rel.relation_type == "SYNONYM_OF":
            continue
        if rel.source_id in ent_map and rel.target_id in ent_map:
            neighbors[rel.source_id].add(rel.target_id)
            neighbors[rel.target_id].add(rel.source_id)
            key = (min(rel.source_id, rel.target_id), max(rel.source_id, rel.target_id))
            edge_chunks[key].update(rel.source_chunk_ids)

    mined: list[HyperEdge] = []
    seen_ids: set[str] = set()

    for target in sorted(entities, key=lambda e: e.entity_id):
        if target.entity_type not in TARGET_ENTITY_TYPES:
            continue

        # Group connected condition entities by shared source chunk so every hyperedge is
        # strictly grounded in a real passage that mentions the joint conditions + target together.
        target_chunks = set(target.source_chunk_ids)
        cond_by_chunk: dict[str, set[str]] = defaultdict(set)

        for nb_id in neighbors.get(target.entity_id, ()):
            nb = ent_map.get(nb_id)
            if nb is None or nb.entity_type not in CONDITION_ENTITY_TYPES:
                continue
            pair_key = (min(nb_id, target.entity_id), max(nb_id, target.entity_id))
            shared = (set(nb.source_chunk_ids) & target_chunks) | edge_chunks.get(pair_key, set())
            for cid in shared:
                cond_by_chunk[cid].add(nb_id)

        # Merge chunks supporting the exact same condition set for this target
        condset_to_chunks: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for cid, cond_set in sorted(cond_by_chunk.items()):
            if len(cond_set) < min_conditions:
                continue
            # Prefer high-signal Symptom/SignalChannel/RiskFactor first when capping at max_conditions
            ranked_conds = sorted(
                cond_set,
                key=lambda eid: (
                    0
                    if ent_map[eid].entity_type in {"Symptom", "SignalChannel", "RiskFactor"}
                    else 1,
                    ent_map[eid].name,
                ),
            )[:max_conditions]
            condset_to_chunks[tuple(sorted(ranked_conds))].append(cid)

        for cond_tuple, cids in condset_to_chunks.items():
            cond_names = [ent_map[eid].name for eid in cond_tuple]
            desc = (
                f"Joint multi-condition rule: {{{', '.join(cond_names)}}} "
                f"===[JOINT_INDICATES]===> {target.name}"
            )
            he = create_hyperedge(
                list(cond_tuple),
                target.entity_id,
                relation_type="JOINT_INDICATES",
                description=desc,
                confidence=round(min(0.98, 0.72 + 0.06 * len(cond_tuple) + 0.04 * len(cids)), 3),
                weight=float(len(cids)),
                source_chunk_ids=cids,
                metadata={"mined_by": "chunk_co_grounded_hypergraph", "condition_count": len(cond_tuple)},
            )
            if he.hyperedge_id not in seen_ids:
                seen_ids.add(he.hyperedge_id)
                mined.append(he)
            if len(mined) >= max_hyperedges:
                break
        if len(mined) >= max_hyperedges:
            break

    # Persist into graph store when supported
    upsert_fn = getattr(graph_store, "upsert_hyperedge", None)
    if callable(upsert_fn):
        for he in mined:
            upsert_fn(he)
    return mined


def match_hyperedges(
    active_entity_ids: list[str],
    hyperedges: list[HyperEdge],
    entity_lookup: dict[str, Entity],
    *,
    min_match_ratio: float = 0.60,
    min_matched_conditions: int = 2,
    top_k: int = 5,
) -> list[HyperEdgeMatch]:
    """Match active query/seed entities against N-ary hyperedges for joint multi-condition inference."""
    active_set = set(active_entity_ids)
    if not active_set or not hyperedges:
        return []

    matches: list[HyperEdgeMatch] = []
    for he in hyperedges:
        if not he.condition_entity_ids:
            continue
        matched_ids = [cid for cid in he.condition_entity_ids if cid in active_set]
        if len(matched_ids) < min(min_matched_conditions, len(he.condition_entity_ids)):
            continue
        coverage = len(matched_ids) / len(he.condition_entity_ids)
        if coverage < min_match_ratio:
            continue
        missing_ids = [cid for cid in he.condition_entity_ids if cid not in active_set]
        # Reward higher coverage + more joint conditions + higher hyperedge confidence
        joint_score = round(
            coverage * he.confidence * (1.0 + 0.25 * (len(matched_ids) - 1)),
            4,
        )
        tgt_ent = entity_lookup.get(he.target_entity_id)
        matches.append(
            HyperEdgeMatch(
                hyperedge_id=he.hyperedge_id,
                relation_type=he.relation_type,
                target_entity_id=he.target_entity_id,
                target_entity_name=tgt_ent.name if tgt_ent else he.target_entity_id,
                matched_condition_ids=matched_ids,
                matched_condition_names=[
                    entity_lookup[c].name if c in entity_lookup else c for c in matched_ids
                ],
                missing_condition_ids=missing_ids,
                missing_condition_names=[
                    entity_lookup[c].name if c in entity_lookup else c for c in missing_ids
                ],
                coverage_ratio=round(coverage, 4),
                joint_score=joint_score,
                source_chunk_ids=list(he.source_chunk_ids),
                description=he.description,
            )
        )

    matches.sort(key=lambda m: (-m.joint_score, -m.coverage_ratio, m.hyperedge_id))
    return matches[:top_k]


def format_hyperedge_context(matches: list[HyperEdgeMatch]) -> str:
    """Format matched N-ary hyperedges as a concise grounded evidence block for LLM reasoning."""
    if not matches:
        return ""
    lines = ["Joint Multi-Condition HyperGraph Signals (N-ary Verified Rules):"]
    for m in matches:
        matched_str = " + ".join(m.matched_condition_names)
        missing_str = f" (missing: {', '.join(m.missing_condition_names)})" if m.missing_condition_names else ""
        cids = ", ".join(m.source_chunk_ids[:3])
        lines.append(
            f"- [{m.hyperedge_id}] {{{matched_str}}} ===[{m.relation_type}]===> "
            f"[{m.target_entity_id}] {m.target_entity_name} "
            f"(match={m.coverage_ratio * 100:.0f}%, score={m.joint_score:.2f}, chunks=[{cids}]){missing_str}"
        )
    return "\n".join(lines)

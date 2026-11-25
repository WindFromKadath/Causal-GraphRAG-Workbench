"""Standalone interactive Node-to-Node Multi-Hop Replay + Tree-of-Thoughts Causal Visualizer.

Generates a zero-CDN, 100% offline self-contained HTML showcase titled
"基于思维树的透明推理与因果链条解析" featuring:
1. Interactive Node -> Edge -> Node multi-hop reading sequence & 5-frame pruning convergence replay
   (Vector Seed Response -> Hop-1 -> Hop-2+ -> Compact/ToT Pruning -> Final Grounded Core Patch).
2. Interactive Tree-of-Thoughts competing branch topology (Selected vs Pruned + G/S scores).
3. Step-by-step auditable causal chain with clickable source-chunk provenance drawer.
"""

from __future__ import annotations

import html
import json
from collections import deque
from pathlib import Path
from typing import Any

from llm_graphrag_kg.core.interfaces import BaseGraphStore, BaseVectorStore
from llm_graphrag_kg.core.schema import QueryResult
from llm_graphrag_kg.viz.workbench_data import build_readable_citation_maps


def _build_node_traversal_graph(
    *,
    result: QueryResult,
    graph_store: BaseGraphStore,
    branches: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    max_nodes: int = 26,
) -> dict[str, Any]:
    """Build truthful Hop-by-Hop Node -> Edge -> Node traversal waves & final core convergence."""
    all_entities = {e.entity_id: e for e in graph_store.get_all_entities()}
    all_relations = {r.relation_id: r for r in graph_store.get_all_relations()}

    trav_meta = result.metadata.get("traversal_trace")
    trav_dict = trav_meta if isinstance(trav_meta, dict) else {}
    ppr_scores: dict[str, float] = (
        dict(trav_dict.get("ppr_entity_scores") or {}) if isinstance(trav_dict, dict) else {}
    )
    dense_scores: dict[str, float] = (
        dict(trav_dict.get("dense_entity_scores") or {}) if isinstance(trav_dict, dict) else {}
    )
    triple_scores: dict[str, float] = (
        dict(trav_dict.get("triple_seed_scores") or {}) if isinstance(trav_dict, dict) else {}
    )

    seed_ids = [eid for eid in result.trace.seed_entity_ids if eid in all_entities]
    seed_set = set(seed_ids)

    # Identify entities explicitly cited in GroundingAudit-verified steps or selected branch
    cited_eids: set[str] = set()
    cited_rids: set[str] = set()
    for st in steps:
        for e in st.get("entities") or []:
            eid = str(e.get("id", ""))
            if eid in all_entities:
                cited_eids.add(eid)
        for r in st.get("relations") or []:
            rid = str(r.get("id", ""))
            if rid in all_relations:
                cited_rids.add(rid)
                cited_eids.add(all_relations[rid].source_id)
                cited_eids.add(all_relations[rid].target_id)

    selected_branch_eids: set[str] = set()
    selected_branch_rids: set[str] = set()
    pruned_branch_eids: dict[str, set[str]] = {}
    branch_tags_by_eid: dict[str, list[str]] = {}

    for br in branches:
        bid = str(br.get("branch_id", ""))
        status = str(br.get("status", "active"))
        tid = str(br.get("target_entity_id", ""))
        explored = [str(x) for x in (br.get("explored_entity_ids") or []) if str(x) in all_entities]
        sup_rids = [
            str(r.get("id", ""))
            for r in (br.get("supporting_relations") or [])
            if str(r.get("id", "")) in all_relations
        ]
        br_nodes = set(explored)
        if tid in all_entities:
            br_nodes.add(tid)
        for rid in sup_rids:
            br_nodes.add(all_relations[rid].source_id)
            br_nodes.add(all_relations[rid].target_id)

        for eid in br_nodes:
            branch_tags_by_eid.setdefault(eid, [])
            if bid and bid not in branch_tags_by_eid[eid]:
                branch_tags_by_eid[eid].append(bid)

        if status == "selected":
            if tid in all_entities:
                selected_branch_eids.add(tid)
            selected_branch_rids.update(sup_rids[:6])
            for rid in sup_rids[:6]:
                selected_branch_eids.add(all_relations[rid].source_id)
                selected_branch_eids.add(all_relations[rid].target_id)
        elif status == "pruned":
            pruned_branch_eids[bid] = br_nodes

    # Candidate pool of all nodes read during retrieval + ToT grounding
    candidate_eids: set[str] = set(seed_ids)
    candidate_eids.update(eid for eid in result.trace.expanded_entity_ids if eid in all_entities)
    candidate_eids.update(cited_eids)
    candidate_eids.update(selected_branch_eids)
    for bnodes in pruned_branch_eids.values():
        candidate_eids.update(bnodes)

    # Candidate relations connecting candidate nodes
    trace_rids = set(result.trace.relation_ids) | cited_rids | selected_branch_rids
    candidate_rels = [
        r
        for r in all_relations.values()
        if (
            r.source_id in candidate_eids
            and r.target_id in candidate_eids
            and (r.relation_id in trace_rids or r.source_id in seed_set or r.target_id in seed_set)
        )
    ]

    # Compute exact shortest hop distance from Wave-0 seeds via BFS
    adj: dict[str, list[tuple[str, str]]] = {eid: [] for eid in candidate_eids}
    for r in candidate_rels:
        adj[r.source_id].append((r.target_id, r.relation_id))
        adj[r.target_id].append((r.source_id, r.relation_id))

    hop_by_eid: dict[str, int] = {}
    parent_by_eid: dict[str, str | None] = {}
    queue: deque[str] = deque()
    for sid in seed_ids:
        hop_by_eid[sid] = 0
        parent_by_eid[sid] = None
        queue.append(sid)

    while queue:
        cur = queue.popleft()
        cur_hop = hop_by_eid[cur]
        for nxt, _rid in adj.get(cur, ()):
            if nxt not in hop_by_eid:
                hop_by_eid[nxt] = min(cur_hop + 1, 3)
                parent_by_eid[nxt] = cur
                queue.append(nxt)

    # Assign disconnected ToT branch targets/nodes to hop 2 so they appear cleanly in tree waves
    for eid in candidate_eids:
        if eid not in hop_by_eid:
            hop_by_eid[eid] = 1 if eid in cited_eids or eid in selected_branch_eids else 2

    # Final core patch ("只留下一片最后有被用到的Node"):
    # Include cited nodes + selected target + bridge nodes along BFS shortest paths back to seeds
    used_eids: set[str] = set(cited_eids)
    for br in branches:
        if br.get("status") == "selected" and str(br.get("target_entity_id", "")) in all_entities:
            used_eids.add(str(br["target_entity_id"]))
    if not used_eids:
        used_eids.update(selected_branch_eids or set(seed_ids[:4]))

    # Pull shortest-path bridge ancestors back to Wave-0 seeds so the final patch is a connected chain
    bridge_additions: set[str] = set()
    for eid in list(used_eids):
        cur: str | None = eid
        guard = 0
        while cur is not None and guard < 6:
            bridge_additions.add(cur)
            cur = parent_by_eid.get(cur)
            guard += 1
    used_eids.update(bridge_additions)
    # Also ensure directly cited relation endpoints stay in used_eids
    for rid in cited_rids:
        if rid in all_relations:
            used_eids.add(all_relations[rid].source_id)
            used_eids.add(all_relations[rid].target_id)
    # Include at least one seed that touches used_eids if any seed is directly linked
    for r in candidate_rels:
        if r.source_id in seed_set and r.target_id in used_eids:
            used_eids.add(r.source_id)
        elif r.target_id in seed_set and r.source_id in used_eids:
            used_eids.add(r.target_id)

    all_pruned_union: set[str] = set()
    for bnodes in pruned_branch_eids.values():
        all_pruned_union.update(bnodes)

    def _activation(eid: str) -> float:
        ppr_v = float(ppr_scores.get(eid, 0.0))
        d_v = float(dense_scores.get(eid, 0.0))
        t_v = float(triple_scores.get(eid, 0.0))
        base = max(ppr_v * 5.0, d_v, t_v)
        if eid in seed_set:
            base = max(base, 0.85)
        if eid in used_eids:
            base = max(base, 0.72)
        return round(min(1.0, max(0.15, base)), 3)

    def _node_status(eid: str) -> str:
        if eid in used_eids:
            return "used"
        if eid in all_pruned_union:
            return "pruned"
        return "dropped"

    # Prioritize keeping ALL used nodes + seeds + branch targets, then highest-activation neighbors
    branch_target_ids = {
        str(br.get("target_entity_id", ""))
        for br in branches
        if str(br.get("target_entity_id", "")) in all_entities
    }
    ordered_eids = sorted(
        candidate_eids,
        key=lambda eid: (
            0
            if eid in used_eids
            else (1 if eid in branch_target_ids else (2 if eid in seed_set else 3)),
            hop_by_eid.get(eid, 2),
            -_activation(eid),
            all_entities[eid].name,
        ),
    )
    kept_eids = ordered_eids[: max(len(used_eids) + 4, max_nodes)]
    kept_set = set(kept_eids)

    # True reading order: Hop 0 -> Hop 1 -> Hop 2 -> Hop 3, within each hop by activation descending
    read_sorted = sorted(
        kept_eids,
        key=lambda eid: (
            hop_by_eid.get(eid, 2),
            -_activation(eid),
            0 if eid in used_eids else 1,
            all_entities[eid].name,
        ),
    )

    nodes_out: list[dict[str, Any]] = []
    for order_idx, eid in enumerate(read_sorted, start=1):
        ent = all_entities[eid]
        channels: list[str] = []
        if eid in seed_set:
            channels.append("seed")
        if eid in dense_scores:
            channels.append("dense")
        if eid in triple_scores:
            channels.append("q2t")
        if eid in ppr_scores:
            channels.append("ppr")
        nodes_out.append(
            {
                "id": eid,
                "name": ent.name,
                "entity_type": ent.entity_type,
                "description": ent.description[:220],
                "hop": int(hop_by_eid.get(eid, 2)),
                "read_order": order_idx,
                "activation": _activation(eid),
                "status": _node_status(eid),
                "is_seed": eid in seed_set,
                "is_target": eid in branch_target_ids,
                "branches": branch_tags_by_eid.get(eid, []),
                "channels": channels,
                "chunk_ids": list(ent.source_chunk_ids[:4]),
            }
        )

    # Filter & classify edges between kept nodes (cap edges cleanly for clarity)
    edges_out: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for r in sorted(
        candidate_rels,
        key=lambda rel: (
            0 if rel.relation_id in cited_rids else (1 if rel.relation_id in selected_branch_rids else 2),
            -float(rel.weight),
            rel.relation_id,
        ),
    ):
        if r.source_id not in kept_set or r.target_id not in kept_set or r.source_id == r.target_id:
            continue
        sig = (r.source_id, r.relation_type, r.target_id)
        if sig in seen_pairs:
            continue
        seen_pairs.add(sig)

        s_hop = hop_by_eid.get(r.source_id, 2)
        t_hop = hop_by_eid.get(r.target_id, 2)
        # Direct edge left-to-right visually when hops differ; preserve semantic direction label
        if r.relation_id in cited_rids or (
            r.source_id in used_eids and r.target_id in used_eids
        ):
            e_status = "used"
        elif r.source_id in all_pruned_union or r.target_id in all_pruned_union:
            e_status = "pruned"
        else:
            e_status = "dropped"

        edges_out.append(
            {
                "id": r.relation_id,
                "source": r.source_id,
                "target": r.target_id,
                "relation_type": r.relation_type,
                "description": r.description[:180],
                "hop_wave": max(s_hop, t_hop),
                "status": e_status,
                "cited": r.relation_id in cited_rids,
                "chunk_ids": list(r.source_chunk_ids[:3]),
            }
        )
        if len(edges_out) >= 48:
            break

    max_hop = max((n["hop"] for n in nodes_out), default=0)
    used_count = sum(1 for n in nodes_out if n["status"] == "used")
    pruned_count = sum(1 for n in nodes_out if n["status"] == "pruned")
    dropped_count = sum(1 for n in nodes_out if n["status"] == "dropped")

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "max_hop": max_hop,
        "stats": {
            "total_read_nodes": len(nodes_out),
            "total_expanded_nodes": len(candidate_eids),
            "used_nodes": used_count,
            "pruned_nodes": pruned_count,
            "dropped_nodes": dropped_count,
            "total_edges": len(edges_out),
            "used_edges": sum(1 for e in edges_out if e["status"] == "used"),
        },
    }


def build_tot_tree_payload(
    result: QueryResult,
    graph_store: BaseGraphStore,
    vector_store: BaseVectorStore | None = None,
) -> dict[str, Any]:
    """Assemble a clean JSON-serializable Node->Edge->Node + ToT tree + causal chain payload."""
    ent_labels, rel_labels = build_readable_citation_maps(graph_store)
    reasoning = result.metadata.get("reasoning")
    reasoning_dict = reasoning if isinstance(reasoning, dict) else {}
    paradigm = str(
        reasoning_dict.get("paradigm") or result.metadata.get("reasoning_paradigm") or "direct"
    ).lower()

    # Chunk text lookup (retrieved evidences first, then vector_store fallback)
    chunk_lookup: dict[str, dict[str, str]] = {}
    for ev in result.evidences:
        chunk_lookup[ev.chunk_id] = {
            "chunk_id": ev.chunk_id,
            "source": ev.source,
            "matched_via": ev.matched_via,
            "text": ev.text,
        }

    def _ensure_chunk(cid: str) -> None:
        if cid in chunk_lookup or vector_store is None:
            return
        ch = vector_store.get_chunk(cid)
        if ch is not None:
            chunk_lookup[cid] = {
                "chunk_id": ch.chunk_id,
                "source": ch.source,
                "matched_via": "store",
                "text": ch.text,
            }

    # Build normalized causal steps
    raw_steps = reasoning_dict.get("steps") or []
    steps: list[dict[str, Any]] = []
    for idx, st in enumerate(raw_steps, start=1):
        if not isinstance(st, dict):
            continue
        c_chunks = [str(x) for x in (st.get("cited_chunk_ids") or [])]
        for cid in c_chunks:
            _ensure_chunk(cid)
        steps.append(
            {
                "step_id": int(st.get("step_id", idx)),
                "stage": str(st.get("stage", "graph_traverse")),
                "claim": str(st.get("claim", "")),
                "grounded": bool(st.get("grounded", False)),
                "grounding_notes": [str(n) for n in (st.get("grounding_notes") or [])],
                "entities": [
                    {"id": str(eid), "label": ent_labels.get(str(eid), str(eid))}
                    for eid in (st.get("cited_entity_ids") or [])
                ],
                "relations": [
                    {"id": str(rid), "label": rel_labels.get(str(rid), str(rid))}
                    for rid in (st.get("cited_relation_ids") or [])
                ],
                "chunks": c_chunks,
            }
        )

    # Build normalized branches (ToT multi-branch OR synthetic single-branch for CoT/Direct)
    raw_branches = reasoning_dict.get("branches") or []
    branches: list[dict[str, Any]] = []
    for br in raw_branches:
        if not isinstance(br, dict):
            continue
        eid = str(br.get("target_entity_id", ""))
        readable_target = ent_labels.get(eid, str(br.get("target_entity_name") or eid or "-"))
        sup_rels = [
            {"id": str(rid), "label": rel_labels.get(str(rid), str(rid))}
            for rid in (br.get("supporting_relations") or [])
        ]
        sup_chunks = [str(cid) for cid in (br.get("supporting_chunks") or [])]
        for cid in sup_chunks:
            _ensure_chunk(cid)
        branches.append(
            {
                "branch_id": str(br.get("branch_id", "H1")),
                "parent_branch_id": str(br.get("parent_branch_id", "")),
                "depth": int(br.get("depth", 1)),
                "verdict": str(br.get("verdict", "likely")),
                "backtracked": bool(br.get("backtracked", False)),
                "status": str(br.get("status", "active")),
                "hypothesis": str(br.get("hypothesis", "")),
                "target_entity_id": eid,
                "target_label": readable_target,
                "explored_entity_ids": [str(x) for x in (br.get("explored_entity_ids") or [])],
                "score": round(float(br.get("score", 0.0)), 3),
                "graph_score": round(float(br.get("deterministic_graph_score", 0.0)), 3),
                "semantic_score": round(float(br.get("semantic_score", 0.0)), 3),
                "supporting_relations": sup_rels,
                "supporting_chunks": sup_chunks,
                "contradicting_evidences": [
                    str(x) for x in (br.get("contradicting_evidences") or [])
                ],
                "prune_reason": str(br.get("prune_reason", "")),
                "folded_memory": str(br.get("folded_memory", "")),
            }
        )

    if not branches:
        anchor_names = [
            ent_labels.get(eid, eid) for eid in result.trace.seed_entity_ids[:3]
        ] or list(result.metadata.get("seed_entity_names", []))[:3]
        branches.append(
            {
                "branch_id": "C1",
                "status": "selected",
                "hypothesis": f"主干因果链推导（Graph-{paradigm.upper()}）",
                "target_entity_id": result.trace.seed_entity_ids[0]
                if result.trace.seed_entity_ids
                else "",
                "target_label": ", ".join(str(a) for a in anchor_names) or "检索核心子图",
                "explored_entity_ids": list(result.trace.expanded_entity_ids[:16]),
                "score": round(float(reasoning_dict.get("confidence", 0.85)), 3),
                "graph_score": 1.0 if result.trace.relation_ids else 0.5,
                "semantic_score": round(float(reasoning_dict.get("confidence", 0.85)), 3),
                "supporting_relations": [
                    {"id": rid, "label": rel_labels.get(rid, rid)}
                    for rid in result.trace.relation_ids[:6]
                ],
                "supporting_chunks": [ev.chunk_id for ev in result.evidences[:4]],
                "contradicting_evidences": [],
                "prune_reason": "",
                "folded_memory": "",
            }
        )

    branches.sort(key=lambda b: (0 if b["status"] == "selected" else 1, -b["score"]))

    traversal = _build_node_traversal_graph(
        result=result,
        graph_store=graph_store,
        branches=branches,
        steps=steps,
    )
    # Ensure chunk lookup covers all chunks referenced by traversal nodes
    for node in traversal["nodes"]:
        for cid in node.get("chunk_ids") or []:
            _ensure_chunk(cid)

    audit = reasoning_dict.get("grounding_audit")
    audit_dict = audit if isinstance(audit, dict) else {}
    comp = result.metadata.get("compaction")
    comp_dict = comp if isinstance(comp, dict) else {}
    ledger = result.metadata.get("harness_ledger")
    ledger_dict = ledger if isinstance(ledger, dict) else {}

    seed_names = [
        str(x)
        for x in (
            result.metadata.get("seed_entity_names")
            or [ent_labels.get(eid, eid) for eid in result.trace.seed_entity_ids]
        )
    ]

    return {
        "question": result.question,
        "mode": result.mode,
        "paradigm": paradigm,
        "answer": result.answer,
        "seed_entities": seed_names[:8],
        "subgraph_stats": {
            "nodes": len(result.trace.expanded_entity_ids),
            "relations": len(result.trace.relation_ids),
            "communities": len(result.trace.community_ids),
            "evidences": len(result.evidences),
        },
        "compaction": {
            "enabled": bool(comp_dict),
            "raw_tokens": int(comp_dict.get("raw_tokens_est", 0)),
            "compacted_tokens": int(comp_dict.get("compacted_tokens_est", 0)),
            "compression_pct": round(float(comp_dict.get("compression_ratio", 0.0)) * 100.0, 1),
            "levels": [str(x) for x in (comp_dict.get("applied_levels") or [])],
        },
        "audit": {
            "passed": bool(audit_dict.get("passed", False)),
            "score_pct": round(float(audit_dict.get("grounding_score", 0.0)) * 100.0, 1),
            "grounded_steps": int(audit_dict.get("grounded_steps", 0)),
            "total_steps": int(audit_dict.get("total_steps", len(steps))),
            "ungrounded_count": len(audit_dict.get("ungrounded_citations") or []),
            "repair_triggered": bool(audit_dict.get("repair_triggered", False)),
        },
        "ledger": {
            "calls": int(ledger_dict.get("total_calls", 1)),
            "wall_seconds": round(float(ledger_dict.get("total_wall_seconds", 0.0)), 2),
        },
        "traversal": traversal,
        "branches": branches,
        "steps": steps,
        "pruned_memories": [str(m) for m in (reasoning_dict.get("pruned_memories") or [])],
        "chunks": chunk_lookup,
    }


def render_tot_tree_html(
    result: QueryResult,
    graph_store: BaseGraphStore,
    vector_store: BaseVectorStore | None = None,
    *,
    title: str = "基于思维树的透明推理与因果链条解析",
    embedded: bool = False,
) -> str:
    """Render a zero-CDN, offline interactive HTML page with Node->Edge->Node replay & ToT tree."""
    payload = build_tot_tree_payload(result, graph_store, vector_store=vector_store)
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    safe_title = html.escape(title)
    body_pad = "10px 14px 18px" if embedded else "22px 26px 36px"

    return f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-alt: #f1f5f9;
      --border: #cbd5e1;
      --border-soft: #e2e8f0;
      --fg: #0f172a;
      --muted: #475569;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --win: #059669;
      --win-bg: #ecfdf5;
      --win-border: #6ee7b7;
      --prune: #dc2626;
      --prune-bg: #fef2f2;
      --prune-border: #fca5a5;
      --warn: #d97706;
      --warn-bg: #fffbeb;
      --ghost: #94a3b8;
      --shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    }}
    [data-theme="dark"] {{
      --bg: #0b1120;
      --surface: #111827;
      --surface-alt: #1e293b;
      --border: #334155;
      --border-soft: #1f2937;
      --fg: #f1f5f9;
      --muted: #94a3b8;
      --accent: #60a5fa;
      --accent-soft: #172554;
      --win: #34d399;
      --win-bg: #064e3b;
      --win-border: #059669;
      --prune: #f87171;
      --prune-bg: #450a0a;
      --prune-border: #b91c1c;
      --warn: #fbbf24;
      --warn-bg: #451a03;
      --ghost: #64748b;
      --shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: {body_pad};
      background: var(--bg);
      color: var(--fg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 13px;
      line-height: 1.5;
    }}
    .wrap {{ max-width: 1260px; margin: 0 auto; }}
    .topbar {{
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: 10px; padding-bottom: 12px; border-bottom: 1px solid var(--border-soft); margin-bottom: 14px;
    }}
    .title-group h1 {{ margin: 0; font-size: 18px; font-weight: 700; }}
    .subtitle {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
    .pill-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
    .pill {{
      display: inline-flex; align-items: center; gap: 4px;
      padding: 3px 9px; border-radius: 999px; font-size: 12px; font-weight: 600;
      background: var(--surface); border: 1px solid var(--border); color: var(--fg);
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }}
    .pill.win {{ background: var(--win-bg); border-color: var(--win-border); color: var(--win); }}
    .pill.warn {{ background: var(--warn-bg); border-color: var(--warn); color: var(--warn); }}
    .pill.accent {{ background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }}
    .btn {{
      padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border);
      background: var(--surface); color: var(--fg); cursor: pointer; font-size: 12px; font-weight: 500;
    }}
    .btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    .btn.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    .btn.primary:hover {{ opacity: 0.92; color: #fff; }}

    .panel {{
      background: var(--surface); border: 1px solid var(--border-soft);
      border-radius: 12px; padding: 14px 16px; box-shadow: var(--shadow); margin-bottom: 14px;
    }}
    .section-tag {{
      font-size: 11.5px; font-weight: 700; color: var(--muted);
      margin-bottom: 10px; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px;
    }}

    /* ===== Node -> Edge -> Node Replay Stage ===== */
    .replay-bar {{
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: 8px; padding: 9px 12px; border-radius: 9px;
      background: var(--surface-alt); border: 1px solid var(--border-soft); margin-bottom: 12px;
    }}
    .frame-pills {{ display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }}
    .frame-btn {{
      padding: 4px 9px; border-radius: 6px; border: 1px solid var(--border);
      background: var(--surface); color: var(--muted); cursor: pointer; font-size: 11.5px; font-weight: 600;
      transition: all 0.15s ease;
    }}
    .frame-btn:hover {{ border-color: var(--accent); color: var(--fg); }}
    .frame-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    .frame-btn.core-step.active {{ background: var(--win); border-color: var(--win); color: #fff; }}
    .toggle-label {{
      display: inline-flex; align-items: center; gap: 5px; font-size: 12px; color: var(--fg); cursor: pointer;
    }}
    .frame-caption {{
      font-size: 12px; color: var(--muted); margin-bottom: 10px;
      display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px;
    }}
    .graph-stage {{
      position: relative; overflow-x: auto; padding: 8px 4px 10px;
      border-radius: 10px; border: 1px solid var(--border-soft); background: var(--bg);
    }}
    .graph-inner {{
      position: relative; min-width: 780px; padding: 10px 44px 12px 14px;
    }}
    #hop-svg {{
      position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1;
      overflow: visible;
    }}
    .hop-columns {{
      position: relative; z-index: 2; display: grid; gap: 64px; align-items: start;
    }}
    .hop-col {{ display: flex; flex-direction: column; gap: 8px; min-width: 155px; }}
    .hop-col-title {{
      font-size: 11px; font-weight: 700; text-align: center; padding: 4px 8px; border-radius: 6px;
      background: var(--surface); border: 1px solid var(--border-soft); color: var(--muted);
      font-family: ui-monospace, monospace;
    }}
    .n-card {{
      position: relative; text-align: left; width: 100%;
      padding: 7px 9px; border-radius: 8px; cursor: pointer;
      background: var(--surface); border: 1.5px solid var(--border);
      transition: opacity 0.25s ease, transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .n-card:hover {{ transform: translateY(-1px); border-color: var(--accent); }}
    .n-card.state-seed {{ border-color: var(--accent); background: var(--accent-soft); }}
    .n-card.state-used {{ border-color: var(--win); background: var(--win-bg); }}
    .n-card.state-pruned {{ border-color: var(--prune-border); background: var(--prune-bg); }}
    .n-card.dimmed {{ opacity: 0.28; filter: grayscale(0.65); border-style: dashed; }}
    .n-card.hidden-node {{ display: none; }}
    .n-card.node-selected {{ box-shadow: 0 0 0 2.5px var(--accent), var(--shadow); opacity: 1; }}
    .n-top {{ display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 2px; }}
    .n-order {{
      font-size: 10px; font-weight: 800; padding: 1px 5px; border-radius: 4px;
      background: var(--surface-alt); color: var(--fg); font-family: ui-monospace, monospace;
    }}
    .n-card.state-used .n-order {{ background: var(--win); color: #fff; }}
    .n-card.state-pruned .n-order {{ background: var(--prune); color: #fff; }}
    .n-type {{ font-size: 10px; color: var(--muted); font-family: ui-monospace, monospace; }}
    .n-name {{ font-size: 12px; font-weight: 700; color: var(--fg); word-break: break-word; }}
    .n-foot {{
      margin-top: 3px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: 4px; font-size: 10px; color: var(--muted); font-family: ui-monospace, monospace;
    }}
    .n-tag {{
      padding: 0 4px; border-radius: 3px; font-size: 9.5px; font-weight: 700;
      border: 1px solid var(--border); background: var(--surface);
    }}
    .n-tag.win {{ border-color: var(--win-border); color: var(--win); }}
    .n-tag.prune {{ border-color: var(--prune-border); color: var(--prune); }}
    .node-inspector {{
      margin-top: 10px; padding: 10px 12px; border-radius: 8px;
      background: var(--surface-alt); border: 1px solid var(--accent); display: none;
    }}
    .node-inspector.open {{ display: block; }}

    /* ===== ToT Branch Tree ===== */
    .root-box {{
      max-width: 780px; margin: 0 auto; padding: 10px 14px; border-radius: 10px;
      background: var(--accent-soft); border: 1.5px solid var(--accent); text-align: center;
    }}
    .root-q {{ font-size: 13.5px; font-weight: 700; color: var(--fg); margin-bottom: 5px; }}
    .seed-wrap {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 5px; }}
    .seed-chip {{
      font-size: 11px; padding: 2px 7px; border-radius: 5px;
      background: var(--surface); border: 1px solid var(--border); color: var(--muted);
    }}
    #tree-svg {{ display: block; width: 100%; height: 48px; overflow: visible; }}
    .branch-grid {{ display: grid; gap: 10px; }}
    .branch-card {{
      text-align: left; border-radius: 10px; padding: 11px; cursor: pointer;
      background: var(--surface); border: 1.5px solid var(--border);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .branch-card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow); }}
    .branch-card.selected-status {{ background: var(--win-bg); border-color: var(--win-border); }}
    .branch-card.pruned-status {{ opacity: 0.88; }}
    .branch-card.active-focus {{ box-shadow: 0 0 0 2.5px var(--accent), var(--shadow); opacity: 1; }}
    .br-head {{ display: flex; align-items: center; justify-content: space-between; gap: 6px; margin-bottom: 5px; }}
    .br-badge {{
      font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 999px;
      font-family: ui-monospace, monospace;
    }}
    .br-badge.win {{ background: var(--win); color: #fff; }}
    .br-badge.prune {{ background: var(--prune-bg); color: var(--prune); border: 1px solid var(--prune-border); }}
    .br-score {{ font-size: 14px; font-weight: 800; font-family: ui-monospace, monospace; }}
    .br-hyp {{ font-size: 12.5px; font-weight: 600; color: var(--fg); margin-bottom: 4px; }}
    .br-target {{ font-size: 11px; color: var(--muted); margin-bottom: 7px; }}
    .bar-row {{
      display: flex; align-items: center; gap: 6px; font-size: 11px;
      margin-bottom: 3px; font-family: ui-monospace, monospace;
    }}
    .bar-track {{ flex: 1; height: 5px; border-radius: 999px; background: var(--surface-alt); overflow: hidden; }}
    .bar-fill-g {{ height: 100%; background: var(--accent); border-radius: 999px; }}
    .bar-fill-s {{ height: 100%; background: var(--win); border-radius: 999px; }}
    .br-prune-note {{
      margin-top: 5px; padding-top: 5px; border-top: 1px dashed var(--border);
      font-size: 11px; color: var(--prune);
    }}

    /* ===== Causal Chain & Evidence Drawer ===== */
    .detail-head {{
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: 8px; padding-bottom: 9px; border-bottom: 1px solid var(--border-soft); margin-bottom: 11px;
    }}
    .detail-title {{ font-size: 14.5px; font-weight: 700; }}
    .causal-flow {{
      display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 10px; align-items: stretch; padding-bottom: 4px;
    }}
    .step-card {{
      min-width: 230px; flex: 1 1 0; border-radius: 9px; padding: 10px;
      background: var(--surface-alt); border: 1px solid var(--border-soft);
      display: flex; flex-direction: column; justify-content: space-between; gap: 7px;
    }}
    .step-arrow {{
      display: flex; align-items: center; justify-content: center;
      font-size: 18px; font-weight: 700; color: var(--accent); flex: 0 0 auto;
    }}
    .step-top {{ display: flex; justify-content: space-between; align-items: center; gap: 6px; }}
    .stage-pill {{
      font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 5px;
      background: var(--surface); border: 1px solid var(--border); font-family: ui-monospace, monospace;
    }}
    .ok-pill {{ font-size: 11px; font-weight: 700; color: var(--win); font-family: ui-monospace, monospace; }}
    .bad-pill {{ font-size: 11px; font-weight: 700; color: var(--prune); font-family: ui-monospace, monospace; }}
    .step-claim {{ font-size: 12.5px; color: var(--fg); }}
    .cite-box {{ display: flex; flex-wrap: wrap; gap: 4px; }}
    .cite-chip {{
      font-size: 11px; padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border);
      background: var(--surface); color: var(--muted); max-width: 100%;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .cite-chip.chunk-btn {{
      cursor: pointer; border-color: var(--accent); color: var(--accent); font-family: ui-monospace, monospace;
    }}
    .cite-chip.chunk-btn:hover {{ background: var(--accent-soft); }}
    .cite-chip.rel-chip {{ border-color: var(--win-border); color: var(--win); }}
    .answer-box {{
      margin-top: 12px; padding: 11px 13px; border-radius: 9px;
      background: var(--win-bg); border: 1px solid var(--win-border);
    }}
    .answer-label {{ font-size: 12px; font-weight: 700; color: var(--win); margin-bottom: 4px; }}
    .answer-text {{ font-size: 13px; color: var(--fg); white-space: pre-wrap; }}
    .evidence-drawer {{
      margin-top: 10px; padding: 10px 12px; border-radius: 8px;
      background: var(--surface-alt); border: 1px solid var(--accent); display: none;
    }}
    .evidence-drawer.open {{ display: block; }}
    .ev-head {{
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 5px; font-size: 12px; font-weight: 700; color: var(--accent);
    }}
    .ev-text {{ font-size: 12px; color: var(--fg); white-space: pre-wrap; max-height: 170px; overflow-y: auto; }}
    .prune-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }}
    .prune-box {{
      padding: 10px; border-radius: 8px; background: var(--surface-alt); border: 1px solid var(--border-soft);
    }}
    .prune-box h4 {{ margin: 0 0 6px 0; font-size: 12px; color: var(--muted); }}
    @media (max-width: 820px) {{
      .causal-flow {{ flex-direction: column; }}
      .step-arrow {{ transform: rotate(90deg); }}
    }}
    @media print {{
      .no-print {{ display: none !important; }}
      body {{ background: #fff; padding: 0; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="title-group">
        <h1>{safe_title}</h1>
        <div class="subtitle" id="sub-meta"></div>
      </div>
      <div class="pill-row">
        <span id="pill-grounding" class="pill"></span>
        <span id="pill-compact" class="pill accent"></span>
        <span id="pill-calls" class="pill"></span>
        <button class="btn no-print" id="theme-toggle" type="button">🌓 深浅色</button>
        <button class="btn no-print" onclick="window.print()" type="button">🖨️ 打印 / PDF</button>
      </div>
    </div>

    <!-- Section 1: Node -> Edge -> Node Multi-Hop Reading & Pruning Convergence Replay -->
    <div class="panel">
      <div class="section-tag">
        <span>🧭 Node → Edge → Node 逐跳读取顺序与最终核心片收敛回放（按真实跳数与激活序展开）</span>
        <span id="hop-stats-badge" style="font-family:ui-monospace,monospace;"></span>
      </div>

      <div class="replay-bar no-print">
        <div class="frame-pills">
          <button type="button" class="btn primary" id="btn-play">▶ 自动播放</button>
          <button type="button" class="frame-btn" data-frame="1">1. 向量化响应 (Hop 0)</button>
          <button type="button" class="frame-btn" data-frame="2">2. 一跳边扩散 (Hop 1)</button>
          <button type="button" class="frame-btn" data-frame="3">3. 多跳深层遍历 (Hop 2+)</button>
          <button type="button" class="frame-btn" data-frame="4">4. Compact & ToT 剪枝</button>
          <button type="button" class="frame-btn core-step active" data-frame="5">5. 最终核心片收敛 ✨</button>
        </div>
        <label class="toggle-label">
          <input type="checkbox" id="chk-only-used" checked />
          仅保留最终用到的一片核心 Node
        </label>
      </div>

      <div class="frame-caption">
        <span id="frame-desc"></span>
        <span style="font-family:ui-monospace,monospace;">
          图例：🔵 向量种子 · 🟢 最终留用核心 · 🔴 ToT剪枝 · ⚪ 检索未引用（点击任意 Node 查看边与原文）
        </span>
      </div>

      <div class="graph-stage">
        <div class="graph-inner" id="graph-inner">
          <svg id="hop-svg" aria-hidden="true"></svg>
          <div class="hop-columns" id="hop-columns"></div>
        </div>
      </div>

      <div class="node-inspector" id="node-inspector">
        <div class="ev-head">
          <span id="ni-title"></span>
          <button class="btn" type="button" onclick="closeNodeInspector()">关闭</button>
        </div>
        <div id="ni-body" style="font-size:12px;"></div>
      </div>
    </div>

    <!-- Section 2: Tree-of-Thoughts Competing Branches -->
    <div class="panel">
      <div class="section-tag">
        <span>🌳 思维树竞争假设展开与反证剪枝（点击分支同步高亮上方探索节点并切换下方因果解析）</span>
        <span id="tree-legend"></span>
      </div>
      <div class="root-box" id="root-box">
        <div class="root-q" id="root-question"></div>
        <div class="seed-wrap" id="seed-chips"></div>
      </div>
      <svg id="tree-svg" aria-hidden="true"></svg>
      <div class="branch-grid" id="branch-grid"></div>
    </div>

    <!-- Section 3: Causal Chain & Source Evidence Inspector -->
    <div class="panel">
      <div class="detail-head">
        <div class="detail-title" id="detail-title"></div>
        <div class="pill-row" id="detail-badges"></div>
      </div>
      <div id="detail-body"></div>
      <div class="evidence-drawer" id="evidence-drawer">
        <div class="ev-head">
          <span id="ev-title"></span>
          <button class="btn" type="button" onclick="closeEvidence()">关闭</button>
        </div>
        <div class="ev-text" id="ev-text"></div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    let currentFrame = 5;
    let onlyUsedMode = true;
    let playTimer = null;
    let selectedNodeId = null;
    let hoveredNodeId = null;
    let activeBranchId = DATA.branches.length ? DATA.branches[0].branch_id : null;
    let highlightBranchFilter = null;

    const FRAME_META = {{
      1: 'Wave 0 · 问题向量化与锚点响应：优先点亮与输入语义/词法/三元组最匹配的种子 Node（按激活序 #1, #2...）。',
      2: 'Wave 1 · 一跳边扩散：从种子 Node 沿有向关系边（Edge）读取直接邻居 Node，建立初步因果关联。',
      3: 'Wave 2+ · 多跳深层遍历 & ToT 分支查图：多跳关联路径向右延伸，展开全部候选知识节点与竞争分支。',
      4: 'Wave 3 · Compact 降噪与 ToT 反证剪枝：与最终诊断无关的冗余节点与淘汰分支（PRUNED）虚线灰显。',
      5: 'Wave 4 · 最终核心片收敛：收敛到真正被模型推理与 GroundingAudit 核验引用的连通核心 Node 与因果边！'
    }};

    function esc(s) {{
      return String(s || '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }}

    function stageLabel(stage) {{
      const map = {{
        anchor: '锚点定位',
        graph_traverse: '图谱多跳遍历',
        evidence_check: '原文证据核验',
        synthesis: '综合决策收敛'
      }};
      return map[stage] ? `${{map[stage]}} (${{stage}})` : stage;
    }}

    function initHeader() {{
      const st = DATA.subgraph_stats;
      const ts = DATA.traversal.stats;
      document.getElementById('sub-meta').textContent = (
        `检索模式: ${{DATA.mode.toUpperCase()}} · 思维范式: Graph-${{DATA.paradigm.toUpperCase()}} · ` +
        `读取子图: ${{st.nodes}} 实体 / ${{st.relations}} 关系 → 最终收敛 ${{ts.used_nodes}} 核心节点`
      );
      const gp = document.getElementById('pill-grounding');
      gp.className = 'pill ' + (DATA.audit.passed ? 'win' : 'warn');
      gp.textContent = (
        `Grounding ${{DATA.audit.score_pct.toFixed(1)}}% ` +
        `(${{DATA.audit.grounded_steps}}/${{DATA.audit.total_steps}})`
      );
      const cp = document.getElementById('pill-compact');
      if (DATA.compaction.enabled) {{
        cp.textContent = (
          `Compact ${{DATA.compaction.raw_tokens}}→${{DATA.compaction.compacted_tokens}} tok ` +
          `(-${{DATA.compaction.compression_pct}}%)`
        );
      }} else {{
        cp.textContent = 'Compact 未启用';
      }}
      document.getElementById('pill-calls').textContent =
        `LLM ${{DATA.ledger.calls}}次 · ${{DATA.ledger.wall_seconds}}s`;

      const winCnt = DATA.branches.filter(b => b.status === 'selected').length;
      const prCnt = DATA.branches.filter(b => b.status === 'pruned').length;
      document.getElementById('tree-legend').textContent =
        `共 ${{DATA.branches.length}} 分支（胜出 ${{winCnt}} · 剪枝 ${{prCnt}}）`;
      document.getElementById('root-question').textContent = DATA.question;
      document.getElementById('seed-chips').innerHTML = DATA.seed_entities.length
        ? DATA.seed_entities.map(s => `<span class="seed-chip">⚓ ${{esc(s)}}</span>`).join('')
        : '<span class="seed-chip">自动语义锚点</span>';
    }}

    function isNodeVisible(node) {{
      if (currentFrame === 1) return node.hop === 0;
      if (currentFrame === 2) return node.hop <= 1;
      if (currentFrame === 5 && onlyUsedMode) return node.status === 'used';
      return true;
    }}

    function isNodeDimmed(node) {{
      if (highlightBranchFilter) {{
        return !(node.branches && node.branches.includes(highlightBranchFilter));
      }}
      if (currentFrame === 4) return node.status !== 'used';
      if (currentFrame === 5 && !onlyUsedMode) return node.status !== 'used';
      return false;
    }}

    function renderHopGraph() {{
      const trav = DATA.traversal;
      const maxHop = Math.max(trav.max_hop || 0, 1);
      const colsEl = document.getElementById('hop-columns');
      const hopLabels = [
        'Hop 0 · 向量响应种子',
        'Hop 1 · 一跳关联节点',
        'Hop 2 · 二跳因果延伸',
        'Hop 3 · 深层规范 / SOP'
      ];
      const activeMaxCol = currentFrame === 1 ? 0 : (currentFrame === 2 ? Math.min(1, maxHop) : maxHop);
      const numCols = activeMaxCol + 1;
      colsEl.style.gridTemplateColumns = `repeat(${{numCols}}, minmax(160px, 1fr))`;

      let visibleCount = 0;
      let htmlCols = '';
      for (let h = 0; h <= activeMaxCol; h++) {{
        const colNodes = trav.nodes.filter(n => n.hop === h && isNodeVisible(n));
        visibleCount += colNodes.length;
        const cards = colNodes.length
          ? colNodes.map(n => {{
              const dimmed = isNodeDimmed(n);
              let stateCls = '';
              if (currentFrame >= 4) {{
                if (n.status === 'used') stateCls = 'state-used';
                else if (n.status === 'pruned') stateCls = 'state-pruned';
              }} else if (n.is_seed) {{
                stateCls = 'state-seed';
              }}
              const selCls = selectedNodeId === n.id ? 'node-selected' : '';
              const dimCls = dimmed ? 'dimmed' : '';
              const statusTag = currentFrame >= 4
                ? (n.status === 'used'
                    ? '<span class="n-tag win">✓ 留用</span>'
                    : (n.status === 'pruned'
                        ? '<span class="n-tag prune">✂ 剪枝</span>'
                        : '<span class="n-tag">降噪</span>'))
                : (n.is_seed ? '<span class="n-tag">⚓ 种子</span>' : '');
              const brTags = (n.branches || []).slice(0, 2)
                .map(b => `<span class="n-tag">${{esc(b)}}</span>`).join('');
              return `
                <button type="button" class="n-card ${{stateCls}} ${{dimCls}} ${{selCls}}"
                  data-nid="${{esc(n.id)}}">
                  <div class="n-top">
                    <span class="n-order">#${{n.read_order}}</span>
                    <span class="n-type">${{esc(n.entity_type)}}</span>
                  </div>
                  <div class="n-name">${{esc(n.name)}}</div>
                  <div class="n-foot">
                    <span>act=${{n.activation.toFixed(2)}}</span>
                    <span>${{brTags}} ${{statusTag}}</span>
                  </div>
                </button>
              `;
            }}).join('')
          : (
            '<div style="font-size:11px; color:var(--muted); text-align:center; padding:12px 4px;">'
            + '该跳无留用节点</div>'
          );

        htmlCols += `
          <div class="hop-col">
            <div class="hop-col-title">${{hopLabels[h] || ('Hop ' + h)}} (${{colNodes.length}})</div>
            ${{cards}}
          </div>
        `;
      }}
      colsEl.innerHTML = htmlCols;

      const ts = trav.stats;
      document.getElementById('hop-stats-badge').textContent = (
        `当前可见 ${{visibleCount}} / 读取 ${{ts.total_read_nodes}} 节点 · ` +
        `最终核心片 ${{ts.used_nodes}} 节点 (${{ts.used_edges}} 边)`
      );
      document.getElementById('frame-desc').textContent = FRAME_META[currentFrame] || '';

      colsEl.querySelectorAll('[data-nid]').forEach(btn => {{
        btn.addEventListener('click', () => {{
          selectedNodeId = btn.dataset.nid;
          renderHopGraph();
          openNodeInspector(selectedNodeId);
        }});
        btn.addEventListener('mouseenter', () => {{
          hoveredNodeId = btn.dataset.nid;
          drawHopEdges();
        }});
        btn.addEventListener('mouseleave', () => {{
          hoveredNodeId = null;
          drawHopEdges();
        }});
      }});
      requestAnimationFrame(drawHopEdges);
    }}

    function drawHopEdges() {{
      const svg = document.getElementById('hop-svg');
      const inner = document.getElementById('graph-inner');
      if (!svg || !inner) return;
      const baseRect = inner.getBoundingClientRect();
      const cardMap = {{}};
      inner.querySelectorAll('[data-nid]').forEach(el => {{
        cardMap[el.dataset.nid] = el.getBoundingClientRect();
      }});

      const nodeById = {{}};
      DATA.traversal.nodes.forEach(n => {{ nodeById[n.id] = n; }});

      const defs = `
        <defs>
          <marker id="arr-win" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--win)" />
          </marker>
          <marker id="arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--accent)" />
          </marker>
          <marker id="arr-prune" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--prune)" />
          </marker>
          <marker id="arr-ghost" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--ghost)" />
          </marker>
        </defs>
      `;

      const maxEdgeWave = currentFrame === 1 ? 0 : (currentFrame === 2 ? 1 : 99);
      let drawn = '';
      DATA.traversal.edges.forEach((e, idx) => {{
        if (e.hop_wave > maxEdgeWave) return;
        const r1 = cardMap[e.source];
        const r2 = cardMap[e.target];
        if (!r1 || !r2) return;
        if (currentFrame === 5 && onlyUsedMode && e.status !== 'used') return;

        const n1 = nodeById[e.source];
        const n2 = nodeById[e.target];
        const sameCol = n1 && n2 && n1.hop === n2.hop;
        const leftToRight = !n1 || !n2 || n1.hop <= n2.hop;
        const srcRect = leftToRight ? r1 : r2;
        const dstRect = leftToRight ? r2 : r1;

        let x1, y1, x2, y2, d, lx, ly;
        if (sameCol) {{
          x1 = srcRect.right - baseRect.left;
          y1 = srcRect.top + srcRect.height / 2 - baseRect.top;
          x2 = dstRect.right - baseRect.left;
          y2 = dstRect.top + dstRect.height / 2 - baseRect.top;
          const bulge = 22 + (idx % 3) * 8;
          d = `M ${{x1}} ${{y1}} C ${{x1 + bulge}} ${{y1}}, ${{x2 + bulge}} ${{y2}}, ${{x2}} ${{y2}}`;
          lx = Math.max(x1, x2) + bulge * 0.6;
          ly = (y1 + y2) / 2;
        }} else {{
          x1 = srcRect.right - baseRect.left;
          y1 = srcRect.top + srcRect.height / 2 - baseRect.top;
          x2 = dstRect.left - baseRect.left - 3;
          y2 = dstRect.top + dstRect.height / 2 - baseRect.top;
          const dx = Math.max(24, Math.abs(x2 - x1) * 0.45);
          d = `M ${{x1}} ${{y1}} C ${{x1 + dx}} ${{y1}}, ${{x2 - dx}} ${{y2}}, ${{x2}} ${{y2}}`;
          lx = (x1 + x2) / 2;
          ly = (y1 + y2) / 2 - 4;
        }}

        const focusId = hoveredNodeId || selectedNodeId;
        const isSelTouch = focusId && (e.source === focusId || e.target === focusId);
        let stroke = 'var(--accent)';
        let marker = 'url(#arr-acc)';
        let dash = '';
        let op = sameCol ? 0.24 : 0.45;
        let sw = 1.35;

        if (currentFrame >= 4) {{
          if (e.status === 'used') {{
            stroke = 'var(--win)';
            marker = 'url(#arr-win)';
            op = sameCol ? 0.26 : 0.52;
            sw = e.cited ? 1.85 : 1.4;
          }} else if (e.status === 'pruned') {{
            stroke = 'var(--prune)';
            marker = 'url(#arr-prune)';
            dash = 'stroke-dasharray="5 4"';
            op = 0.26;
            sw = 1.2;
          }} else {{
            stroke = 'var(--ghost)';
            marker = 'url(#arr-ghost)';
            dash = 'stroke-dasharray="3 4"';
            op = 0.18;
            sw = 1.0;
          }}
        }}
        if (focusId) {{
          if (isSelTouch) {{
            op = 0.96;
            sw = 2.5;
          }} else {{
            op = 0.08;
            sw = 1.0;
          }}
        }}

        drawn += `<path d="${{d}}" fill="none" stroke="${{stroke}}" stroke-width="${{sw}}" ` +
          `opacity="${{op}}" ${{dash}} marker-end="${{marker}}">` +
          `<title>${{esc(e.relation_type)}}</title></path>`;
      }});

      svg.innerHTML = defs + drawn;
    }}

    function openNodeInspector(nid) {{
      const node = DATA.traversal.nodes.find(n => n.id === nid);
      if (!node) return;
      const box = document.getElementById('node-inspector');
      const statusTxt = node.status === 'used'
        ? '✓ 最终留用核心节点'
        : (node.status === 'pruned' ? '✂ ToT 反证剪枝节点' : '⚪ 检索遍历但未引用节点');
      document.getElementById('ni-title').textContent = (
        `节点 #${{node.read_order}} · ${{node.name}} [${{node.entity_type}}] · ` +
        `Hop ${{node.hop}} · ${{statusTxt}}`
      );
      const rels = DATA.traversal.edges.filter(e => e.source === nid || e.target === nid);
      const nameById = {{}};
      DATA.traversal.nodes.forEach(n => {{ nameById[n.id] = n.name; }});
      const relHtml = rels.length
        ? rels.slice(0, 6).map(r => {{
            const s = nameById[r.source] || r.source;
            const t = nameById[r.target] || r.target;
            const relTxt = `${{esc(s)}} --[${{esc(r.relation_type)}}]--> ${{esc(t)}}`;
            return `<span class="cite-chip rel-chip">🔗 ${{relTxt}}</span>`;
          }}).join(' ')
        : '<span style="color:var(--muted);">当前视图内无直连边</span>';
      const chHtml = (node.chunk_ids || []).map(cid =>
        `<button type="button" class="cite-chip chunk-btn" ` +
        `onclick="openEvidence('${{esc(cid)}}')">📄 ${{esc(cid)}}</button>`
      ).join(' ');

      document.getElementById('ni-body').innerHTML = `
        <div style="margin-bottom:6px; color:var(--fg);">${{esc(node.description || '来自知识图谱抽取实体。')}}</div>
        <div class="cite-box" style="margin-bottom:6px;">${{relHtml}}</div>
        <div class="cite-box">${{chHtml}}</div>
      `;
      box.classList.add('open');
    }}

    function closeNodeInspector() {{
      selectedNodeId = null;
      document.getElementById('node-inspector').classList.remove('open');
      renderHopGraph();
    }}

    function setFrame(frameNum) {{
      currentFrame = frameNum;
      highlightBranchFilter = null;
      document.querySelectorAll('[data-frame]').forEach(b => {{
        b.classList.toggle('active', Number(b.dataset.frame) === frameNum);
      }});
      renderHopGraph();
    }}

    function togglePlay() {{
      const btn = document.getElementById('btn-play');
      if (playTimer) {{
        clearInterval(playTimer);
        playTimer = null;
        btn.textContent = '▶ 自动播放';
        return;
      }}
      btn.textContent = '⏸ 停止播放';
      setFrame(1);
      playTimer = setInterval(() => {{
        if (currentFrame >= 5) {{
          clearInterval(playTimer);
          playTimer = null;
          btn.textContent = '▶ 重播 1→5';
          return;
        }}
        setFrame(currentFrame + 1);
      }}, 1500);
    }}

    function renderBranches() {{
      const grid = document.getElementById('branch-grid');
      const cols = Math.min(Math.max(DATA.branches.length, 1), 3);
      grid.style.gridTemplateColumns = `repeat(${{cols}}, minmax(0, 1fr))`;

      grid.innerHTML = DATA.branches.map(br => {{
        const isWin = br.status === 'selected';
        const isFocus = br.branch_id === activeBranchId;
        const gPct = Math.round(br.graph_score * 100);
        const sPct = Math.round(br.semantic_score * 100);
        const depthTag = br.depth ? `D${{br.depth}}` : '';
        const btTag = br.backtracked ? ' · ↩ 回溯' : '';
        const verdictTag = br.verdict ? ` [${{br.verdict}}]` : '';
        return `
          <button type="button"
            class="branch-card ${{isWin ? 'selected-status' : 'pruned-status'}} ${{isFocus ? 'active-focus' : ''}}"
            data-bid="${{esc(br.branch_id)}}">
            <div class="br-head">
              <span class="br-badge ${{isWin ? 'win' : 'prune'}}">
                ${{esc(br.branch_id)}} ${{depthTag}} · ${{isWin ? '✓ SELECTED' : '✂ PRUNED'}}${{btTag}}
              </span>
              <span class="br-score">${{br.score.toFixed(2)}}</span>
            </div>
            <div class="br-hyp">${{esc(br.hypothesis)}}${{esc(verdictTag)}}</div>
            <div class="br-target">🎯 目标节点：<strong>${{esc(br.target_label)}}</strong></div>
            <div class="bar-row">
              <span>G(图谱)</span>
              <div class="bar-track"><div class="bar-fill-g" style="width:${{gPct}}%"></div></div>
              <span>${{br.graph_score.toFixed(2)}}</span>
            </div>
            <div class="bar-row">
              <span>S(语义)</span>
              <div class="bar-track"><div class="bar-fill-s" style="width:${{sPct}}%"></div></div>
              <span>${{br.semantic_score.toFixed(2)}}</span>
            </div>
            ${{!isWin && br.prune_reason ? `<div class="br-prune-note">✂ ${{esc(br.prune_reason)}}</div>` : ''}}
          </button>
        `;
      }}).join('');

      grid.querySelectorAll('[data-bid]').forEach(btn => {{
        btn.addEventListener('click', () => {{
          activeBranchId = btn.dataset.bid;
          renderBranches();
          renderDetail();
          requestAnimationFrame(drawTreeLines);
        }});
      }});
      requestAnimationFrame(drawTreeLines);
    }}

    function drawTreeLines() {{
      const svg = document.getElementById('tree-svg');
      const rootEl = document.getElementById('root-box');
      const cards = Array.from(document.querySelectorAll('.branch-card'));
      if (!svg || !rootEl || !cards.length) return;
      const svgRect = svg.getBoundingClientRect();
      const rootRect = rootEl.getBoundingClientRect();
      const startX = rootRect.left + rootRect.width / 2 - svgRect.left;
      const endY = svgRect.height || 48;
      const midY = endY * 0.52;

      svg.innerHTML = cards.map(card => {{
        const bid = card.dataset.bid;
        const br = DATA.branches.find(b => b.branch_id === bid);
        const isWin = br && br.status === 'selected';
        const cRect = card.getBoundingClientRect();
        const endX = cRect.left + cRect.width / 2 - svgRect.left;
        const stroke = isWin ? 'var(--win)' : 'var(--prune)';
        const dash = isWin ? '' : 'stroke-dasharray="6 4"';
        const sw = isWin ? 2.5 : 1.6;
        const d = `M ${{startX}} 0 C ${{startX}} ${{midY}}, ${{endX}} ${{midY}}, ${{endX}} ${{endY}}`;
        const label = isWin ? '✓ 胜出路径' : '✂ 反证剪枝';
        return `
          <path d="${{d}}" fill="none" stroke="${{stroke}}" stroke-width="${{sw}}" ${{dash}} />
          <circle cx="${{endX}}" cy="${{endY - 2}}" r="3.5" fill="${{stroke}}" />
          <text x="${{(startX + endX) / 2}}" y="${{midY - 3}}" text-anchor="middle"
            font-size="10.5" font-weight="700" fill="${{stroke}}">${{label}}</text>
        `;
      }}).join('');
    }}

    function openEvidence(chunkId) {{
      const ch = DATA.chunks[chunkId];
      const drawer = document.getElementById('evidence-drawer');
      document.getElementById('ev-title').textContent = ch
        ? `📄 原文溯源切片：${{ch.chunk_id}}（来源：${{ch.source}} · 通道：${{ch.matched_via}}）`
        : `📄 原文切片：${{chunkId}}（未在证据池中找到全文）`;
      document.getElementById('ev-text').textContent = ch ? ch.text : '无可用文本内容。';
      drawer.classList.add('open');
      drawer.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}
    function closeEvidence() {{
      document.getElementById('evidence-drawer').classList.remove('open');
    }}

    function renderDetail() {{
      closeEvidence();
      const br = DATA.branches.find(b => b.branch_id === activeBranchId) || DATA.branches[0];
      if (!br) return;
      const isWin = br.status === 'selected';

      document.getElementById('detail-title').textContent = isWin
        ? `胜出因果链条解析（${{br.branch_id}}：${{br.target_label}}）`
        : `剪枝分支反证与 L2 折叠记录（${{br.branch_id}}：${{br.target_label}}）`;

      document.getElementById('detail-badges').innerHTML = `
        <span class="pill ${{isWin ? 'win' : ''}}">综合分 ${{br.score.toFixed(2)}}</span>
        <span class="pill">图谱 G=${{br.graph_score.toFixed(2)}}</span>
        <span class="pill">语义 S=${{br.semantic_score.toFixed(2)}}</span>
      `;

      const body = document.getElementById('detail-body');
      if (isWin) {{
        const stepsHtml = DATA.steps.length
          ? `<div class="causal-flow">` + DATA.steps.map((st, idx) => {{
              const arrow = idx > 0 ? `<div class="step-arrow">➔</div>` : '';
              const entChips = st.entities.map(e =>
                `<span class="cite-chip" title="${{esc(e.id)}}">🔹 ${{esc(e.label)}}</span>`
              ).join('');
              const relChips = st.relations.map(r =>
                `<span class="cite-chip rel-chip" title="${{esc(r.id)}}">🔗 ${{esc(r.label)}}</span>`
              ).join('');
              const chChips = st.chunks.map(cid =>
                `<button type="button" class="cite-chip chunk-btn" ` +
                `onclick="openEvidence('${{esc(cid)}}')">📄 ${{esc(cid)}}</button>`
              ).join('');
              const okCls = st.grounded ? 'ok-pill' : 'bad-pill';
              const okTxt = st.grounded ? '✓ GROUNDED' : '⚠ CHECK';
              return `${{arrow}}
                <div class="step-card">
                  <div>
                    <div class="step-top">
                      <span class="stage-pill">Step ${{st.step_id}} · ${{esc(stageLabel(st.stage))}}</span>
                      <span class="${{okCls}}">${{okTxt}}</span>
                    </div>
                    <div class="step-claim" style="margin-top:7px;">${{esc(st.claim)}}</div>
                  </div>
                  <div class="cite-box">${{entChips}}${{relChips}}${{chChips}}</div>
                </div>
              `;
            }}).join('') + `</div>`
          : `<div class="prune-box">当前模式（${{esc(DATA.paradigm)}}）未生成分步 CoT。</div>`;

        body.innerHTML = `
          ${{stepsHtml}}
          <div class="answer-box">
            <div class="answer-label">✅ 经图谱溯源验证的最终诊断 / 结论</div>
            <div class="answer-text">${{esc(DATA.answer)}}</div>
          </div>
        `;
      }} else {{
        const supRelHtml = br.supporting_relations.length
          ? br.supporting_relations.map(r =>
              `<div class="cite-chip rel-chip" style="margin-bottom:4px;">🔗 ${{esc(r.label)}}</div>`
            ).join('')
          : '<div style="color:var(--muted);">无显著强支持三元组</div>';
        const contraHtml = br.contradicting_evidences.length
          ? br.contradicting_evidences.map(c => `<div style="margin-bottom:4px;">• ${{esc(c)}}</div>`).join('')
          : `<div>• ${{esc(br.prune_reason || '综合图谱命中率与语义一致性低于主胜出假设')}}</div>`;
        const chHtml = br.supporting_chunks.length
          ? br.supporting_chunks.map(cid =>
              `<button type="button" class="cite-chip chunk-btn" ` +
              `onclick="openEvidence('${{esc(cid)}}')">📄 ${{esc(cid)}}</button>`
            ).join(' ')
          : '<span style="color:var(--muted);">无额外关联切片</span>';
        const pruneTitle = esc(br.prune_reason || '得分低于主假设');
        const foldedTxt = esc(
          br.folded_memory ||
          (DATA.pruned_memories.join('\\n') || '已按 L2 策略折叠并归档排除理由')
        );

        body.innerHTML = `
          <div class="prune-grid">
            <div class="prune-box">
              <h4>✂️ 剪枝反证依据</h4>
              <div style="color:var(--prune); font-weight:600; margin-bottom:6px;">${{pruneTitle}}</div>
              ${{contraHtml}}
            </div>
            <div class="prune-box">
              <h4>🔍 该分支独立查图命中的关联</h4>
              ${{supRelHtml}}
              <div style="margin-top:8px;">${{chHtml}}</div>
            </div>
            <div class="prune-box">
              <h4>🗜️ L2 状态折叠留痕 (CompactedMemory)</h4>
              <div style="font-family:ui-monospace,monospace; font-size:11.5px; white-space:pre-wrap;">
                ${{foldedTxt}}
              </div>
            </div>
          </div>
        `;
      }}
    }}

    document.querySelectorAll('[data-frame]').forEach(btn => {{
      btn.addEventListener('click', () => {{
        if (playTimer) {{
          clearInterval(playTimer);
          playTimer = null;
          document.getElementById('btn-play').textContent = '▶ 自动播放';
        }}
        setFrame(Number(btn.dataset.frame));
      }});
    }});
    document.getElementById('btn-play').addEventListener('click', togglePlay);
    document.getElementById('chk-only-used').addEventListener('change', (e) => {{
      onlyUsedMode = Boolean(e.target.checked);
      renderHopGraph();
    }});
    document.getElementById('theme-toggle').addEventListener('click', () => {{
      const root = document.documentElement;
      root.setAttribute('data-theme', root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    }});
    window.addEventListener('resize', () => {{
      drawHopEdges();
      drawTreeLines();
    }});

    initHeader();
    renderHopGraph();
    renderBranches();
    renderDetail();
  </script>
</body>
</html>
"""


def export_tot_tree_html(
    result: QueryResult,
    graph_store: BaseGraphStore,
    output_path: Path,
    vector_store: BaseVectorStore | None = None,
    *,
    title: str = "基于思维树的透明推理与因果链条解析",
) -> Path:
    """Write standalone offline interactive Node->Edge->Node + ToT HTML file to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_str = render_tot_tree_html(
        result,
        graph_store,
        vector_store=vector_store,
        title=title,
        embedded=False,
    )
    output_path.write_text(html_str, encoding="utf-8")
    return output_path

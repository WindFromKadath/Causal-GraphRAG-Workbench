"""Paper-faithful Multi-Depth Graph-ToT v2 (ToT NeurIPS 2023 + ToG/ToG-2 ICLR 2024/2025).

Implements genuine multi-level tree search over Knowledge Graph edges with:
- True `for depth in 1..max_depth` frontier expansion along REAL KG relations `(e_t --[r]--> e_{t+1})`
  accompanied by original source passages (ToG-2 bidirectional graph + text grounding).
- Explicit State Evaluator emitting discrete ToT verdicts (`sure` | `likely` | `impossible`) + score:
  * `sure` -> Sufficiency Early Stop (converges immediately without wasted depth)
  * `impossible` -> Immediate pruning + L2 counter-evidence folding
- Two canonical search strategies:
  * `bfs` (ToT-BFS Beam Search): retains Top-B `likely` branches per level and expands all at depth+1.
  * `dfs` (ToT-DFS + Backtracking): greedily deepens highest-scoring branch; if deep verification
    hits `impossible` or drops below `backtrack_threshold`, marks `backtracked=True` and pops back
    to expand the runner-up parent branch!
- Dual-model tiering: `planner_harness` (e.g. fast `qwen3.5:4b` or unified `qwen3.5:9b`) executes
  per-level edge pruning & state evaluation, while `synth_harness` (`qwen3.5:9b`) synthesizes the
  final grounded answer audited by `GroundingAudit`.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.interfaces import BaseGraphStore, BaseVectorStore
from llm_graphrag_kg.core.schema import Community, Entity, EvidenceItem, Relation
from llm_graphrag_kg.harness.context_compactor import ContextCompactor, estimate_tokens
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.reasoning.cot import audit_cot_grounding
from llm_graphrag_kg.reasoning.schema import (
    ReasoningResult,
    ThoughtBranch,
    ToTSynthesisOutput,
)
from llm_graphrag_kg.reasoning.tot import TOT_SYSTEM_PROMPT, _tokenize_terms

DEEP_TOT_EVAL_SYSTEM_PROMPT = (
    "You are a ToT + ToG-2 Knowledge Graph Tree Search evaluator. "
    "Given candidate next-hop KG edge expansions and their real source text passages, "
    "score each candidate honestly (0.0-1.0) and assign a discrete ToT verdict:\n"
    "- `sure`: this path + passage already provides sufficient verified evidence to answer the question.\n"
    "- `likely`: promising partial evidence; worth expanding another hop.\n"
    "- `impossible`: dead end, irrelevant, or contradicted by evidence; prune immediately.\n"
    "Return ONLY valid JSON."
)


class DeepToTExpandScoreItem(BaseModel):
    candidate_id: str
    verdict: Literal["sure", "likely", "impossible"] = "likely"
    semantic_score: float = Field(default=0.5, ge=0.0, le=1.0)
    hypothesis: str = ""
    prune_reason: str = ""


class DeepToTLevelEvalOutput(BaseModel):
    evaluations: list[DeepToTExpandScoreItem] = Field(default_factory=list)


def _model_label(harness: LLMHarness) -> str:
    return str(getattr(harness.llm, "model", type(harness.llm).__name__))


class DeepGraphToTEngine:
    """Multi-depth ToT-BFS / ToT-DFS+Backtracking engine with real ToG edge expansion."""

    def __init__(
        self,
        synth_harness: LLMHarness,
        *,
        planner_harness: LLMHarness | None = None,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        compactor: ContextCompactor | None = None,
        search_strategy: Literal["bfs", "dfs"] = "bfs",
        beam_width: int = 2,
        max_depth: int = 2,
        backtrack_threshold: float = 0.62,
        temperature: float = 0.1,
    ) -> None:
        self.synth_harness = synth_harness
        self.planner_harness = planner_harness or synth_harness
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.compactor = compactor or synth_harness.compactor
        self.search_strategy = "dfs" if search_strategy == "dfs" else "bfs"
        self.beam_width = max(2, min(beam_width, 4))
        self.max_depth = max(1, min(max_depth, 3))
        self.backtrack_threshold = backtrack_threshold
        self.temperature = temperature

    def run(
        self,
        *,
        question: str,
        seed_entity_ids: list[str],
        entities: list[Entity],
        relations: list[Relation],
        communities: list[Community],
        evidences: list[EvidenceItem],
    ) -> tuple[ReasoningResult, list[Entity], list[Relation], list[EvidenceItem]]:
        """Run multi-depth tree search (BFS Beam Search or DFS + Backtracking) & synthesize."""
        self.synth_harness.reset_ledger()
        if self.planner_harness is not self.synth_harness:
            self.planner_harness.reset_ledger()

        all_store_ents = {e.entity_id: e for e in self.graph_store.get_all_entities()}
        all_store_rels = self.graph_store.get_all_relations()
        incident_rels: dict[str, list[Relation]] = {}
        for r in all_store_rels:
            incident_rels.setdefault(r.source_id, []).append(r)
            incident_rels.setdefault(r.target_id, []).append(r)

        union_ents: dict[str, Entity] = {e.entity_id: e for e in entities}
        union_rels: dict[str, Relation] = {r.relation_id: r for r in relations}
        union_evs: dict[str, EvidenceItem] = {ev.chunk_id: ev for ev in evidences}
        q_terms = _tokenize_terms(question)

        # Anchor roots: seed entities (fallback to top retrieved entities)
        root_eids = [eid for eid in seed_entity_ids if eid in all_store_ents][:4]
        if not root_eids:
            root_eids = [e.entity_id for e in entities[:4] if e.entity_id in all_store_ents]

        all_branches: list[ThoughtBranch] = []
        pruned_memories: list[str] = []
        planner_calls = 0
        planner_seconds = 0.0
        planner_load_seconds = 0.0
        backtrack_count = 0
        early_stop_sure = False
        max_depth_reached = 0
        planner_model = _model_label(self.planner_harness)

        def _evaluate_candidates(
            cand_specs: list[dict[str, object]],
            *,
            depth: int,
            step_tag: str,
        ) -> list[ThoughtBranch]:
            nonlocal planner_calls, planner_seconds, planner_load_seconds
            if not cand_specs:
                return []

            blocks: list[str] = []
            for spec in cand_specs:
                cid = str(spec["candidate_id"])
                path_str = str(spec["path_str"])
                ev_txt = str(spec["evidence_preview"])
                blocks.append(
                    f"=== Candidate {cid} (depth={depth}) ===\n"
                    f"Path: {path_str}\n"
                    f"Passage Evidence: {ev_txt}"
                )

            prompt = (
                f"GRAPH_TOT_DEEP_EVAL_TASK (depth={depth}, strategy={self.search_strategy})\n"
                f"Question: {question}\n\n"
                + "\n\n".join(blocks)
                + "\n\nEvaluate EACH candidate above independently. Return strict JSON:\n"
                '{"evaluations": [{"candidate_id": "H1", "verdict": "sure|likely|impossible", '
                '"semantic_score": 0.85, "hypothesis": "<short <=12 words>", '
                '"prune_reason": "<short reason if impossible>"}]}'
            )

            t0 = time.perf_counter()
            eval_map: dict[str, DeepToTExpandScoreItem] = {}
            try:
                out, _ = self.planner_harness.run_structured(
                    prompt,
                    DeepToTLevelEvalOutput,
                    step_name=step_tag,
                    system_prompt=DEEP_TOT_EVAL_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=640,
                )
                eval_map = {item.candidate_id.strip(): item for item in out.evaluations}
            except ValueError:
                eval_map = {}
            planner_calls += 1
            planner_seconds += time.perf_counter() - t0
            tel = getattr(self.planner_harness.llm, "last_telemetry", None)
            if isinstance(tel, dict):
                planner_load_seconds += float(tel.get("load_seconds", 0.0))

            built: list[ThoughtBranch] = []
            for spec in cand_specs:
                cid = str(spec["candidate_id"])
                ev_item = eval_map.get(cid)
                g_score = float(spec["graph_score"])
                if ev_item is not None:
                    s_score = float(ev_item.semantic_score)
                    verdict = ev_item.verdict
                    hyp = ev_item.hypothesis.strip() or str(spec["path_str"])
                    p_reason = ev_item.prune_reason.strip()
                else:
                    s_score = g_score
                    verdict = "likely" if g_score >= 0.45 else "impossible"
                    hyp = str(spec["path_str"])
                    p_reason = ""

                bonus = 0.08 if verdict == "sure" else (-0.15 if verdict == "impossible" else 0.0)
                score = round(min(1.0, max(0.0, 0.40 * g_score + 0.60 * s_score + bonus)), 4)
                br = ThoughtBranch(
                    branch_id=cid,
                    parent_branch_id=str(spec["parent_branch_id"]),
                    hypothesis=hyp,
                    target_entity_id=str(spec["target_entity_id"]),
                    target_entity_name=str(spec["target_entity_name"]),
                    depth=depth,
                    path_entity_ids=list(spec["path_entity_ids"]),  # type: ignore[arg-type]
                    path_relation_ids=list(spec["path_relation_ids"]),  # type: ignore[arg-type]
                    explored_entity_ids=list(spec["path_entity_ids"]),  # type: ignore[arg-type]
                    supporting_relations=list(spec["path_relation_ids"]),  # type: ignore[arg-type]
                    supporting_chunks=list(spec["chunk_ids"]),  # type: ignore[arg-type]
                    deterministic_graph_score=g_score,
                    semantic_score=round(s_score, 4),
                    score=score,
                    verdict=verdict,
                    status="active",
                    prune_reason=p_reason,
                    evaluated_by_model=planner_model,
                )
                built.append(br)
            return built

        def _enumerate_children(
            parent_branch: ThoughtBranch | None,
            frontier_eids: list[str],
            *,
            depth: int,
            max_candidates: int,
        ) -> list[dict[str, object]]:
            """Propose top real KG edge transitions from frontier entities."""
            raw_cands: list[tuple[float, Entity, Relation, Entity]] = []
            visited_path = set(parent_branch.path_entity_ids) if parent_branch else set()
            seen_edges: set[tuple[str, str]] = set()

            for src_id in frontier_eids:
                src_ent = all_store_ents.get(src_id)
                if src_ent is None:
                    continue
                for rel in incident_rels.get(src_id, []):
                    nxt_id = rel.target_id if rel.source_id == src_id else rel.source_id
                    if nxt_id == src_id or nxt_id in visited_path:
                        continue
                    if (rel.relation_id, nxt_id) in seen_edges:
                        continue
                    nxt_ent = all_store_ents.get(nxt_id)
                    if nxt_ent is None:
                        continue
                    seen_edges.add((rel.relation_id, nxt_id))

                    hay = f"{nxt_ent.name} {nxt_ent.description} {rel.relation_type} {rel.description}".lower()
                    hits = sum(1 for t in q_terms if t in hay)
                    term_sc = min(1.0, hits / max(len(q_terms), 1)) if q_terms else 0.4
                    type_bonus = 0.25 if rel.relation_type != "CO_OCCURS_WITH" else 0.0
                    w_bonus = min(0.25, float(rel.weight) * 0.05)
                    det_sc = round(min(1.0, 0.55 * term_sc + type_bonus + w_bonus + 0.15), 4)
                    raw_cands.append((det_sc, src_ent, rel, nxt_ent))

            raw_cands.sort(key=lambda x: (-x[0], x[3].name))
            specs: list[dict[str, object]] = []
            used_targets: set[str] = set()
            prefix = f"{parent_branch.branch_id}." if parent_branch else "H"

            for det_sc, src_ent, rel, nxt_ent in raw_cands:
                if nxt_ent.entity_id in used_targets:
                    continue
                used_targets.add(nxt_ent.entity_id)
                union_ents[src_ent.entity_id] = src_ent
                union_ents[nxt_ent.entity_id] = nxt_ent
                union_rels[rel.relation_id] = rel

                cids = list(dict.fromkeys([*rel.source_chunk_ids[:2], *nxt_ent.source_chunk_ids[:2]]))[:3]
                previews: list[str] = []
                for cid in cids:
                    if cid not in union_evs:
                        ch = self.vector_store.get_chunk(cid)
                        if ch is not None:
                            union_evs[cid] = EvidenceItem(
                                chunk_id=ch.chunk_id,
                                doc_id=ch.doc_id,
                                source=ch.source,
                                text=ch.text,
                                score=0.9,
                                matched_via=f"tot_d{depth}",
                            )
                    if cid in union_evs:
                        previews.append(f"[{cid}] {union_evs[cid].text[:420]}")

                base_eids = list(parent_branch.path_entity_ids) if parent_branch else [src_ent.entity_id]
                base_rids = list(parent_branch.path_relation_ids) if parent_branch else []
                path_eids = [*base_eids, nxt_ent.entity_id]
                path_rids = [*base_rids, rel.relation_id]
                path_names = [
                    all_store_ents[e].name if e in all_store_ents else e for e in path_eids
                ]
                path_str = f"{' -> '.join(path_names)} (via {rel.relation_type})"

                idx_num = len(specs) + 1
                specs.append(
                    {
                        "candidate_id": f"{prefix}{idx_num}",
                        "parent_branch_id": parent_branch.branch_id if parent_branch else "",
                        "target_entity_id": nxt_ent.entity_id,
                        "target_entity_name": nxt_ent.name,
                        "path_entity_ids": path_eids,
                        "path_relation_ids": path_rids,
                        "path_str": path_str,
                        "chunk_ids": cids,
                        "evidence_preview": " | ".join(previews) if previews else "(no chunk text)",
                        "graph_score": det_sc,
                    }
                )
                if len(specs) >= max_candidates:
                    break
            return specs

        def _fold_pruned(br: ThoughtBranch, reason: str, *, backtracked: bool = False) -> ThoughtBranch:
            folded = self.compactor.fold_state_l2(
                label=f"{br.branch_id}(d={br.depth}): {br.target_entity_name}",
                raw_text=f"{br.hypothesis}. Reason: {reason}",
                status="pruned",
                entities=[br.target_entity_name] if br.target_entity_name else [],
                chunk_ids=br.supporting_chunks,
                llm=None,
            )
            line = folded.to_compact_line()
            pruned_memories.append(line)
            return br.model_copy(
                update={
                    "status": "pruned",
                    "backtracked": backtracked,
                    "prune_reason": reason,
                    "folded_memory": line,
                }
            )

        # =====================================================================
        # MULTI-DEPTH TREE SEARCH (ToT-BFS Beam Search OR ToT-DFS + Backtracking)
        # =====================================================================
        d1_specs = _enumerate_children(None, root_eids, depth=1, max_candidates=self.beam_width + 2)
        d1_branches = _evaluate_candidates(d1_specs, depth=1, step_tag="deep_tot_depth1_eval")
        d1_branches.sort(key=lambda b: (-b.score, b.branch_id))
        if d1_branches:
            max_depth_reached = 1

        winner: ThoughtBranch | None = None

        if self.search_strategy == "bfs":
            # --- ToT-BFS: Retain Top-B `likely` branches per level and expand deeper ---
            frontier: list[ThoughtBranch] = []
            for rank, br in enumerate(d1_branches):
                if br.verdict == "impossible":
                    all_branches.append(
                        _fold_pruned(br, br.prune_reason or "Depth-1 评估为 impossible，立即剪枝。")
                    )
                elif rank < self.beam_width:
                    frontier.append(br)
                    all_branches.append(br)
                    if br.verdict == "sure" and winner is None:
                        winner = br
                        early_stop_sure = True
                else:
                    all_branches.append(
                        _fold_pruned(br, f"超出 Depth-1 Beam 宽度 ({self.beam_width})。")
                    )

            for depth in range(2, self.max_depth + 1):
                if early_stop_sure or not frontier:
                    break
                next_specs: list[dict[str, object]] = []
                for parent in frontier:
                    next_specs.extend(
                        _enumerate_children(
                            parent,
                            [parent.target_entity_id],
                            depth=depth,
                            max_candidates=2,
                        )
                    )
                if not next_specs:
                    break
                child_branches = _evaluate_candidates(
                    next_specs[: self.beam_width + 2],
                    depth=depth,
                    step_tag=f"deep_tot_depth{depth}_eval",
                )
                child_branches.sort(key=lambda b: (-b.score, b.branch_id))
                max_depth_reached = depth
                next_frontier: list[ThoughtBranch] = []
                for rank, cbr in enumerate(child_branches):
                    if cbr.verdict == "impossible":
                        all_branches.append(
                            _fold_pruned(
                                cbr,
                                cbr.prune_reason or f"Depth-{depth} 评估为 impossible，剪枝。",
                            )
                        )
                    elif rank < self.beam_width:
                        next_frontier.append(cbr)
                        all_branches.append(cbr)
                        if cbr.verdict == "sure" and winner is None:
                            winner = cbr
                            early_stop_sure = True
                    else:
                        all_branches.append(
                            _fold_pruned(cbr, f"超出 Depth-{depth} Beam 宽度 ({self.beam_width})。")
                        )
                frontier = next_frontier

            if winner is None:
                active_pool = [b for b in all_branches if b.status != "pruned"]
                if active_pool:
                    active_pool.sort(key=lambda b: (-b.score, -b.depth, b.branch_id))
                    winner = active_pool[0]
        else:
            # --- ToT-DFS + True Backtracking ---
            # Keep stack of viable Depth-1 branches; deepen rank-0 first; backtrack if dead end!
            viable_d1: list[ThoughtBranch] = []
            for br in d1_branches:
                if br.verdict == "impossible":
                    all_branches.append(
                        _fold_pruned(br, br.prune_reason or "Depth-1 评估为 impossible，剪枝。")
                    )
                else:
                    viable_d1.append(br)
                    all_branches.append(br)

            for idx_d1, parent in enumerate(viable_d1[: self.beam_width]):
                if parent.verdict == "sure":
                    winner = parent
                    early_stop_sure = True
                    break
                if self.max_depth < 2:
                    winner = parent
                    break

                child_specs = _enumerate_children(
                    parent, [parent.target_entity_id], depth=2, max_candidates=2
                )
                if not child_specs:
                    if idx_d1 + 1 < len(viable_d1[: self.beam_width]):
                        backtrack_count += 1
                        # Mark dead-end leaf as backtracked and pop to next Depth-1 branch
                        all_branches = [
                            _fold_pruned(
                                b,
                                "无更深有效出边，触发 DFS 回溯切换次优分支。",
                                backtracked=True,
                            )
                            if b.branch_id == parent.branch_id
                            else b
                            for b in all_branches
                        ]
                        continue
                    winner = parent
                    break

                child_branches = _evaluate_candidates(
                    child_specs,
                    depth=2,
                    step_tag=f"deep_tot_dfs_{parent.branch_id}_d2",
                )
                child_branches.sort(key=lambda b: (-b.score, b.branch_id))
                max_depth_reached = 2
                best_child = child_branches[0]

                # Check if deep exploration succeeded or hit dead end -> Backtrack!
                need_backtrack = (
                    best_child.verdict == "impossible"
                    or best_child.score < self.backtrack_threshold
                ) and (idx_d1 + 1 < len(viable_d1[: self.beam_width]))

                if need_backtrack:
                    backtrack_count += 1
                    for cbr in child_branches:
                        all_branches.append(
                            _fold_pruned(
                                cbr,
                                cbr.prune_reason
                                or f"DFS 深挖得分 ({cbr.score:.2f} < {self.backtrack_threshold:.2f})，触发回溯。",
                                backtracked=True,
                            )
                        )
                    all_branches = [
                        _fold_pruned(
                            b,
                            f"子路径未达阈值 ({self.backtrack_threshold:.2f})，DFS 回溯切换下一候选。",
                            backtracked=True,
                        )
                        if b.branch_id == parent.branch_id
                        else b
                        for b in all_branches
                    ]
                    continue

                for rank, cbr in enumerate(child_branches):
                    if rank == 0 and cbr.verdict != "impossible":
                        all_branches.append(cbr)
                        winner = cbr
                        if cbr.verdict == "sure":
                            early_stop_sure = True
                    else:
                        all_branches.append(
                            _fold_pruned(
                                cbr,
                                cbr.prune_reason or f"低于 DFS 胜出子分支 ({best_child.score:.2f})。",
                            )
                        )
                if winner is not None:
                    break

            if winner is None:
                active_pool = [b for b in all_branches if b.status != "pruned"]
                pool = active_pool or all_branches
                if pool:
                    pool.sort(key=lambda b: (-b.score, -b.depth, b.branch_id))
                    winner = pool[0]

        # Finalize branch statuses: mark winner + its ancestor lineage as `selected`
        win_lineage: set[str] = set()
        if winner is not None:
            win_lineage.add(winner.branch_id)
            if winner.parent_branch_id:
                win_lineage.add(winner.parent_branch_id)

        final_branches: list[ThoughtBranch] = []
        for br in all_branches:
            if br.branch_id in win_lineage:
                final_branches.append(br.model_copy(update={"status": "selected", "prune_reason": ""}))
            elif br.status != "pruned":
                final_branches.append(
                    _fold_pruned(
                        br,
                        f"综合得分 ({br.score:.2f}) 低于最终胜出路径 ({winner.score if winner else 0:.2f})。",
                    )
                )
            else:
                final_branches.append(br)

        # =====================================================================
        # FINAL SYNTHESIS (Primary `synth_harness`, full ToG-2 passages + L2 pruned folds)
        # =====================================================================
        win_path_lines: list[str] = []
        path_ent_ids: list[str] = []
        if winner is not None:
            path_ent_ids = list(dict.fromkeys(winner.path_entity_ids))
            win_names = [
                f"[{e}] {all_store_ents[e].name}" if e in all_store_ents else f"[{e}]"
                for e in winner.path_entity_ids
            ]
            win_path_lines.append(
                f"Winning Multi-Hop Path [{winner.branch_id} | depth={winner.depth} | "
                f"verdict={winner.verdict} | score={winner.score:.2f}]: {' -> '.join(win_names)}"
            )
            for rid in winner.path_relation_ids:
                r = union_rels.get(rid)
                if r is not None:
                    sn = all_store_ents[r.source_id].name if r.source_id in all_store_ents else r.source_id
                    tn = all_store_ents[r.target_id].name if r.target_id in all_store_ents else r.target_id
                    win_path_lines.append(
                        f"- [{r.relation_id}] [{r.source_id}] {sn} --[{r.relation_type}]--> "
                        f"[{r.target_id}] {tn} chunks={r.source_chunk_ids[:2]}"
                    )

        # Explicit entity roster so synthesis steps cite exact [ent_...] IDs cleanly
        roster_ids = list(
            dict.fromkeys(
                [
                    *path_ent_ids,
                    *seed_entity_ids[:6],
                    *list(union_ents.keys())[:12],
                ]
            )
        )[:14]
        ent_roster_lines = [
            f"- [{eid}] {union_ents[eid].name} ({union_ents[eid].entity_type})"
            for eid in roster_ids
            if eid in union_ents
        ]

        # Combine winner's chunks + top global PPR/RRF passages (lossless ToG-2 preservation)
        seen_cids: set[str] = set()
        ev_lines: list[str] = []
        ordered_cids = [*(winner.supporting_chunks if winner else []), *(ev.chunk_id for ev in evidences[:6])]
        for cid in ordered_cids:
            if cid in seen_cids or cid not in union_evs:
                continue
            seen_cids.add(cid)
            ev = union_evs[cid]
            ev_lines.append(f"- [{ev.chunk_id} | {ev.source} | via={ev.matched_via}]: {ev.text}")
            if len(ev_lines) >= 7:
                break

        pruned_block = (
            "Pruned / Backtracked Branches (L2 Folded):\n"
            + "\n".join(f"- {m}" for m in pruned_memories[:4])
            if pruned_memories
            else ""
        )
        comm_block = (
            "Community Context:\n"
            + "\n".join(f"- [{c.title}] {c.summary[:180]}" for c in communities[:2])
            if communities
            else ""
        )
        synth_context = "\n\n".join(
            s
            for s in (
                "\n".join(win_path_lines) if win_path_lines else "",
                "Grounded Entities (cite exact [ent_...] IDs):\n" + "\n".join(ent_roster_lines)
                if ent_roster_lines
                else "",
                "Full Grounded Source Passages:\n" + "\n".join(ev_lines) if ev_lines else "",
                comm_block,
                pruned_block,
            )
            if s
        )
        raw_tokens = estimate_tokens(synth_context)

        synth_instruction = (
            "GRAPH_TOT_SYNTH_TASK\n"
            "Synthesize an accurate, grounded answer in the EXACT SAME LANGUAGE as the question "
            "following the Winning Multi-Hop Path and full source passages above. "
            "Cite ONLY exact entity_ids, relation_ids, and chunk_ids present above.\n"
            "Return strict JSON:\n"
            '{"selected_branch_id": "H1.1", '
            '"steps": [{"step_id": 1, "stage": "anchor|graph_traverse|evidence_check|synthesis", '
            '"claim": "<concise step>", "cited_entity_ids": ["ent_..."], '
            '"cited_relation_ids": ["rel_..."], "cited_chunk_ids": ["doc_...::chunk_..."]}], '
            '"excluded_summary": ["<short exclusion>"], '
            '"final_answer": "<grounded answer citing chunk IDs>", "confidence": 0.9}'
        )
        synth_prompt = LLMHarness.compile_prompt(
            context_text=synth_context,
            question=question,
            task_instruction=synth_instruction,
        )

        t_s0 = time.perf_counter()
        synth_load_seconds = 0.0
        try:
            synth_out, _ = self.synth_harness.run_structured(
                synth_prompt,
                ToTSynthesisOutput,
                step_name=f"deep_tot_{self.search_strategy}_synthesize",
                system_prompt=TOT_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=1536,
            )
            synth_calls = 1
        except ValueError:
            synth_out = None
            synth_calls = 0
        synth_seconds = time.perf_counter() - t_s0
        s_tel = getattr(self.synth_harness.llm, "last_telemetry", None)
        if isinstance(s_tel, dict):
            synth_load_seconds += float(s_tel.get("load_seconds", 0.0))

        all_union_ents = list(union_ents.values())
        all_union_rels = list(union_rels.values())
        all_union_evs = list(union_evs.values())

        if synth_out is not None:
            audited_steps, audit = audit_cot_grounding(
                synth_out.steps,
                entities=all_union_ents,
                relations=all_union_rels,
                evidences=all_union_evs,
            )
            final_answer = synth_out.final_answer
            confidence = synth_out.confidence
        else:
            audited_steps, audit = audit_cot_grounding(
                [],
                entities=all_union_ents,
                relations=all_union_rels,
                evidences=all_union_evs,
            )
            final_answer = winner.hypothesis if winner else "Insufficient grounded graph evidence."
            confidence = winner.score if winner else 0.5

        synth_summary = self.synth_harness.ledger.summary()
        merged_ledger: dict[str, object] = {
            **synth_summary,
            "total_calls": planner_calls + synth_calls,
            "planner_model": planner_model,
            "planner_calls": planner_calls,
            "planner_seconds": round(planner_seconds, 3),
            "planner_load_seconds": round(planner_load_seconds, 3),
            "synth_model": _model_label(self.synth_harness),
            "synth_calls": synth_calls,
            "synth_seconds": round(synth_seconds, 3),
            "synth_load_seconds": round(synth_load_seconds, 3),
            "search_strategy": self.search_strategy,
            "max_depth_reached": max_depth_reached,
            "backtrack_count": backtrack_count,
            "early_stop_sure": early_stop_sure,
        }

        paradigm_tag = f"tot_{self.search_strategy}"
        res = ReasoningResult(
            question=question,
            paradigm=paradigm_tag,
            anchor_entities=list(winner.path_entity_ids) if winner else root_eids,
            final_answer=final_answer,
            confidence=confidence,
            steps=audited_steps,
            branches=final_branches,
            pruned_memories=pruned_memories,
            grounding_audit=audit,
            compaction={
                "raw_tokens_est": raw_tokens,
                "compacted_tokens_est": raw_tokens,
                "compression_ratio": 0.0,
                "applied_levels": [
                    f"deep_tot_{self.search_strategy}_d{max_depth_reached}",
                    "L2_pruned_branch_folding",
                ],
            },
            trace_ledger=merged_ledger,
        )
        return res, all_union_ents, all_union_rels, all_union_evs

"""Explicit Controllable Graph-ToT Engine (Think-on-Graph + L2 Compaction, Stage 2.3)."""

from __future__ import annotations

import re

from llm_graphrag_kg.core.interfaces import BaseGraphStore, BaseVectorStore
from llm_graphrag_kg.core.schema import Community, Entity, EvidenceItem, Relation
from llm_graphrag_kg.harness.context_compactor import ContextCompactor, estimate_tokens
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.reasoning.cot import audit_cot_grounding
from llm_graphrag_kg.reasoning.schema import (
    ReasoningResult,
    ThoughtBranch,
    ToTBranchEvaluationOutput,
    ToTBranchProposalItem,
    ToTBranchProposalOutput,
    ToTSynthesisOutput,
)

TOT_SYSTEM_PROMPT = (
    "You are a disciplined, concise, controllable Graph-ToT (Tree-of-Thought on Knowledge Graph) "
    "diagnostic reasoning engine. "
    "Compare competing hypotheses strictly against retrieved Knowledge Graph relations and "
    "source text chunks. Cite ONLY exact entity_ids, relation_ids, and chunk_ids present in "
    "the provided evidence packs. Be concise and match the question's language. "
    "Return ONLY valid JSON matching the requested schema."
)


def _tokenize_terms(text: str) -> set[str]:
    lowered = text.lower()
    en = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
    return en | bigrams


class GraphToTEngine:
    """Budget-controlled Graph-ToT: Branch -> Ground -> Score & Prune -> L2 Fold -> Synthesize."""

    def __init__(
        self,
        harness: LLMHarness,
        *,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        compactor: ContextCompactor | None = None,
        enable_compact: bool = True,
        beam_width: int = 3,
        max_depth: int = 2,
        max_llm_calls: int = 6,
        temperature: float = 0.1,
        synthesis_mode: str = "competitive",  # competitive (ToT prune) | additive (multi_branch union)
        min_branch_score: float = 0.25,
    ) -> None:
        self.harness = harness
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.compactor = compactor or harness.compactor
        self.enable_compact = enable_compact
        self.beam_width = max(2, min(beam_width, 5))
        self.max_depth = max(1, min(max_depth, 3))
        self.max_llm_calls = max(3, max_llm_calls)
        self.temperature = temperature
        self.synthesis_mode = (
            "additive" if synthesis_mode.strip().lower() == "additive" else "competitive"
        )
        self.min_branch_score = min_branch_score

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
        """Execute 4-phase controllable Graph-ToT and return trace + enriched subgraph."""
        # Enforce ToT call budget guardrail on the harness
        prev_max_calls = self.harness.max_calls
        self.harness.max_calls = min(prev_max_calls, self.max_llm_calls)
        self.harness.reset_ledger()

        try:
            return self._execute_tot(
                question=question,
                seed_entity_ids=seed_entity_ids,
                entities=entities,
                relations=relations,
                communities=communities,
                evidences=evidences,
            )
        finally:
            self.harness.max_calls = prev_max_calls

    def _execute_tot(
        self,
        *,
        question: str,
        seed_entity_ids: list[str],
        entities: list[Entity],
        relations: list[Relation],
        communities: list[Community],
        evidences: list[EvidenceItem],
    ) -> tuple[ReasoningResult, list[Entity], list[Relation], list[EvidenceItem]]:
        all_store_entities = self.graph_store.get_all_entities()
        store_ent_by_id = {e.entity_id: e for e in all_store_entities}
        store_ent_by_name: dict[str, Entity] = {}
        for ent in all_store_entities:
            store_ent_by_name[ent.name.strip().lower()] = ent
            for alias in ent.aliases:
                store_ent_by_name[alias.strip().lower()] = ent

        # Initial compact neighborhood summary for Branch Proposal
        ent_map = {e.entity_id: e for e in entities}
        candidate_lines = [
            f"- [{e.entity_id}] {e.name} ({e.entity_type})"
            for e in entities[:16]
        ]
        rel_lines = []
        for r in relations[:16]:
            sn = ent_map[r.source_id].name if r.source_id in ent_map else r.source_id
            tn = ent_map[r.target_id].name if r.target_id in ent_map else r.target_id
            rel_lines.append(f"- [{r.relation_id}] {sn} --[{r.relation_type}]--> {tn}")

        seed_context = "\n\n".join(
            s
            for s in (
                "Candidate KG Entities:\n" + "\n".join(candidate_lines) if candidate_lines else "",
                "Initial Subgraph Relations:\n" + "\n".join(rel_lines) if rel_lines else "",
                "Community Context:\n"
                + "\n".join(f"- [{c.title}] {c.summary[:120]}" for c in communities[:2])
                if communities
                else "",
            )
            if s
        ) or "(Empty graph context)"

        raw_initial_tokens = estimate_tokens(seed_context) + sum(
            estimate_tokens(ev.text) for ev in evidences
        )

        # =====================================================================
        # PHASE 1: BRANCH (Competitive hypotheses OR complementary multi-facet angles)
        # =====================================================================
        if self.synthesis_mode == "additive":
            branch_instruction = (
                "GRAPH_MULTI_BRANCH_FACET_TASK\n"
                f"Decompose the question into {self.beam_width} COMPLEMENTARY facets/sub-topics "
                f"(H1..H{self.beam_width}) that together cover all dimensions of the question. "
                "Keep each `hypothesis` (facet summary) and `rationale` under 15 words. "
                "Map each facet to an exact `target_entity_id` from Candidate KG Entities.\n"
                "Return strict JSON:\n"
                '{"branches": [{"branch_id": "H1", "hypothesis": "<facet focus>", '
                '"target_entity_id": "ent_...", "target_entity_name": "<name>", '
                '"rationale": "<short why>"}]}'
            )
        else:
            branch_instruction = (
                "GRAPH_TOT_BRANCH_TASK\n"
                f"Propose {self.beam_width} distinct candidate hypotheses (H1..H{self.beam_width}), "
                "keeping each `hypothesis` and `rationale` under 15 words. "
                "Map each to an exact `target_entity_id` from Candidate KG Entities.\n"
                "Return strict JSON:\n"
                '{"branches": [{"branch_id": "H1", "hypothesis": "<short claim>", '
                '"target_entity_id": "ent_...", "target_entity_name": "<name>", '
                '"rationale": "<short why>"}]}'
            )
        branch_prompt = LLMHarness.compile_prompt(
            context_text=seed_context,
            question=question,
            task_instruction=branch_instruction,
        )
        try:
            proposal_out, _ = self.harness.run_structured(
                branch_prompt,
                ToTBranchProposalOutput,
                step_name="graph_tot_branch",
                system_prompt=TOT_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=640,
            )
            proposals = proposal_out.branches[: self.beam_width]
        except ValueError:
            proposals = []

        # Deterministic fallback if LLM returned < 2 proposals
        if len(proposals) < 2 and entities:
            used_ids = {p.target_entity_id for p in proposals}
            # Prioritize FaultMode / RootCause / Concept / Equipment entities as hypotheses
            priority_types = {"FaultMode", "RootCause", "FailureMode", "RemediationSOP", "Concept"}
            fallback_candidates = sorted(
                entities,
                key=lambda e: (0 if e.entity_type in priority_types else 1, e.name),
            )
            for ent in fallback_candidates:
                if ent.entity_id in used_ids:
                    continue
                idx = len(proposals) + 1
                proposals.append(
                    ToTBranchProposalItem(
                        branch_id=f"H{idx}",
                        hypothesis=f"候选根因/目标实体：{ent.name}（{ent.entity_type}）",
                        target_entity_id=ent.entity_id,
                        target_entity_name=ent.name,
                    )
                )
                used_ids.add(ent.entity_id)
                if len(proposals) >= min(self.beam_width, len(entities)):
                    break

        # =====================================================================
        # PHASE 2: GROUND (Independent per-branch k-hop KG lookup + deterministic scoring)
        # =====================================================================
        q_terms = _tokenize_terms(question)
        seed_set = set(seed_entity_ids)
        union_entities: dict[str, Entity] = {e.entity_id: e for e in entities}
        union_relations: dict[str, Relation] = {r.relation_id: r for r in relations}
        union_evidences: dict[str, EvidenceItem] = {ev.chunk_id: ev for ev in evidences}

        grounded_branches: list[ThoughtBranch] = []
        branch_evidence_packs: dict[str, str] = {}
        branch_raw_texts: dict[str, str] = {}

        for idx, prop in enumerate(proposals, start=1):
            bid = prop.branch_id.strip() or f"H{idx}"
            resolved_ent = self._resolve_entity(
                prop.target_entity_id,
                prop.target_entity_name,
                prop.hypothesis,
                store_ent_by_id,
                store_ent_by_name,
            )

            b_ents: list[Entity] = []
            b_rels: list[Relation] = []
            b_cids: list[str] = []
            if resolved_ent is not None:
                b_ents, b_rels = self.graph_store.get_k_hop_subgraph(
                    [resolved_ent.entity_id], k_hops=min(self.max_depth, 2)
                )
                for e in b_ents:
                    union_entities[e.entity_id] = e
                    b_cids.extend(e.source_chunk_ids)
                for r in b_rels:
                    union_relations[r.relation_id] = r
                    b_cids.extend(r.source_chunk_ids)

            b_evidences: list[EvidenceItem] = []
            for cid in dict.fromkeys(b_cids):
                if cid in union_evidences:
                    b_evidences.append(union_evidences[cid])
                else:
                    ch = self.vector_store.get_chunk(cid)
                    if ch is not None:
                        ev_item = EvidenceItem(
                            chunk_id=ch.chunk_id,
                            doc_id=ch.doc_id,
                            source=ch.source,
                            text=ch.text,
                            score=0.9,
                            matched_via=f"tot_{bid}",
                        )
                        union_evidences[cid] = ev_item
                        b_evidences.append(ev_item)
                if len(b_evidences) >= 4:
                    break

            # Deterministic graph support score: seed overlap + query term coverage + relation density
            det_score = self._compute_deterministic_branch_score(
                resolved_ent=resolved_ent,
                hypothesis=prop.hypothesis,
                q_terms=q_terms,
                seed_set=seed_set,
                branch_entities=b_ents,
                branch_relations=b_rels,
                branch_evidences=b_evidences,
            )

            # Build compact auditable evidence pack for this branch
            pack_text, raw_pack = self._format_branch_evidence_pack(
                branch_id=bid,
                hypothesis=prop.hypothesis,
                resolved_ent=resolved_ent,
                branch_entities=b_ents,
                branch_relations=b_rels,
                branch_evidences=b_evidences,
                q_terms=q_terms,
            )
            branch_evidence_packs[bid] = pack_text
            branch_raw_texts[bid] = raw_pack

            grounded_branches.append(
                ThoughtBranch(
                    branch_id=bid,
                    hypothesis=prop.hypothesis,
                    target_entity_id=resolved_ent.entity_id if resolved_ent else "",
                    target_entity_name=resolved_ent.name if resolved_ent else prop.target_entity_name,
                    depth=min(self.max_depth, 2),
                    explored_entity_ids=[e.entity_id for e in b_ents[:18]],
                    supporting_relations=[r.relation_id for r in b_rels[:8]],
                    supporting_chunks=[ev.chunk_id for ev in b_evidences[:4]],
                    deterministic_graph_score=det_score,
                    score=det_score,
                    status="active",
                )
            )

        # =====================================================================
        # PHASE 3: SCORE & PRUNE (Competitive ToT) OR NON-PRUNING RETENTION (Additive Multi-Branch)
        # =====================================================================
        if self.synthesis_mode == "additive":
            return self._execute_additive_synthesis(
                question=question,
                initial_entities=entities,
                initial_evidences=evidences,
                communities=communities,
                grounded_branches=grounded_branches,
                branch_evidence_packs=branch_evidence_packs,
                union_entities=union_entities,
                union_relations=union_relations,
                union_evidences=union_evidences,
                raw_initial_tokens=raw_initial_tokens,
                seed_context=seed_context,
            )

        combined_packs = "\n\n".join(branch_evidence_packs.values())
        eval_instruction = (
            "GRAPH_TOT_EVAL_TASK\n"
            "Score each branch (0.0-1.0) against its evidence pack. Keep `contradicting_evidences` "
            "and `prune_reason` very concise (<= 15 words each). Cite ONLY exact IDs from above.\n"
            "Return strict JSON:\n"
            '{"evaluations": [{"branch_id": "H1", "semantic_score": 0.9, '
            '"supporting_relations": ["rel_..."], "supporting_chunks": ["doc_...::chunk_..."], '
            '"contradicting_evidences": ["..."], "prune_reason": ""}]}'
        )
        eval_prompt = LLMHarness.compile_prompt(
            context_text=combined_packs,
            question=question,
            task_instruction=eval_instruction,
        )

        eval_by_bid = {}
        if grounded_branches and self.harness.ledger.total_calls < self.harness.max_calls:
            try:
                eval_out, _ = self.harness.run_structured(
                    eval_prompt,
                    ToTBranchEvaluationOutput,
                    step_name="graph_tot_score_prune",
                    system_prompt=TOT_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=768,
                )
                eval_by_bid = {item.branch_id.strip(): item for item in eval_out.evaluations}
            except ValueError:
                eval_by_bid = {}

        valid_rel_set = set(union_relations.keys())
        valid_chunk_set = set(union_evidences.keys())

        scored_branches: list[ThoughtBranch] = []
        for br in grounded_branches:
            ev_item = eval_by_bid.get(br.branch_id)
            if ev_item is not None:
                sem_score = float(ev_item.semantic_score)
                # Keep only verified real relation/chunk IDs
                verified_rels = [r for r in ev_item.supporting_relations if r in valid_rel_set]
                verified_chunks = [c for c in ev_item.supporting_chunks if c in valid_chunk_set]
                contras = list(ev_item.contradicting_evidences)
                prune_reason = ev_item.prune_reason.strip()
            else:
                sem_score = br.deterministic_graph_score
                verified_rels = br.supporting_relations
                verified_chunks = br.supporting_chunks
                contras = []
                prune_reason = ""

            final_score = round(0.45 * br.deterministic_graph_score + 0.55 * sem_score, 4)
            scored_branches.append(
                br.model_copy(
                    update={
                        "supporting_relations": verified_rels or br.supporting_relations[:5],
                        "supporting_chunks": verified_chunks or br.supporting_chunks[:3],
                        "contradicting_evidences": contras,
                        "semantic_score": round(sem_score, 4),
                        "score": final_score,
                        "prune_reason": prune_reason,
                    }
                )
            )

        scored_branches.sort(key=lambda b: (-b.score, b.branch_id))

        final_branches: list[ThoughtBranch] = []
        pruned_memories: list[str] = []
        winner: ThoughtBranch | None = None

        for rank, br in enumerate(scored_branches):
            if rank == 0:
                selected_br = br.model_copy(update={"status": "selected", "prune_reason": ""})
                winner = selected_br
                final_branches.append(selected_br)
            else:
                reason = (
                    br.prune_reason
                    or ("; ".join(br.contradicting_evidences) if br.contradicting_evidences else "")
                    or f"综合证据得分 ({br.score:.2f}) 低于胜出假设 ({scored_branches[0].score:.2f})。"
                )
                # L2 state folding on pruned branch (preserving exact metrics/entities/chunk_ids)
                fold_source = (
                    f"Hypothesis: {br.hypothesis}. Counter-evidence/Reason: {reason}. "
                    f"Context: {branch_raw_texts.get(br.branch_id, '')[:600]}"
                )
                folded = self.compactor.fold_state_l2(
                    label=f"{br.branch_id}: {br.target_entity_name or br.hypothesis[:24]}",
                    raw_text=fold_source,
                    status="pruned",
                    entities=[br.target_entity_name] if br.target_entity_name else [],
                    chunk_ids=br.supporting_chunks,
                    llm=None,  # deterministic zero-latency L2 fold to preserve budget
                )
                folded_line = folded.to_compact_line()
                pruned_memories.append(folded_line)
                final_branches.append(
                    br.model_copy(
                        update={
                            "status": "pruned",
                            "prune_reason": reason,
                            "folded_memory": folded_line,
                        }
                    )
                )

        # =====================================================================
        # PHASE 4: COMPACT & SYNTHESIZE + DETERMINISTIC GROUNDING AUDIT (ToG-2 Full-Passage Fix)
        # =====================================================================
        winner_pack = branch_evidence_packs.get(winner.branch_id, seed_context) if winner else seed_context
        winner_chunks_seen = set(winner.supporting_chunks) if winner else set()
        extra_global_ev_lines = [
            f"- [{ev.chunk_id} | {ev.source}]: {ev.text}"
            for ev in evidences[:5]
            if ev.chunk_id not in winner_chunks_seen
        ][:3]
        extra_ev_block = (
            "Additional Grounded Source Passages (from PPR/RRF retrieval):\n"
            + "\n".join(extra_global_ev_lines)
            if extra_global_ev_lines
            else ""
        )
        pruned_block = (
            "Pruned Competing Hypotheses (L2 Folded Counter-Evidences):\n"
            + "\n".join(f"- {line}" for line in pruned_memories)
            if pruned_memories
            else ""
        )
        synth_context = "\n\n".join(
            s
            for s in (
                f"WINNING BRANCH ({winner.branch_id if winner else 'N/A'}):\n{winner_pack}",
                extra_ev_block,
                pruned_block,
            )
            if s
        )
        compacted_synth_tokens = estimate_tokens(synth_context)

        synth_instruction = (
            "GRAPH_TOT_SYNTH_TASK\n"
            "Synthesize a concise grounded answer (in the question's language) from the WINNING BRANCH "
            "and briefly state why pruned hypotheses were excluded. Keep each step <= 20 words and "
            "cite ONLY exact entity_ids, relation_ids, and chunk_ids present above.\n"
            "Return strict JSON:\n"
            '{"selected_branch_id": "H1", '
            '"steps": [{"step_id": 1, "stage": "anchor|graph_traverse|evidence_check|synthesis", '
            '"claim": "<short step>", "cited_entity_ids": ["ent_..."], '
            '"cited_relation_ids": ["rel_..."], "cited_chunk_ids": ["doc_...::chunk_..."]}], '
            '"excluded_summary": ["<short exclusion>"], '
            '"final_answer": "<concise structured answer citing chunk IDs and exclusions>", '
            '"confidence": 0.9}'
        )
        synth_prompt = LLMHarness.compile_prompt(
            context_text=synth_context,
            question=question,
            task_instruction=synth_instruction,
        )

        synth_out: ToTSynthesisOutput | None = None
        if self.harness.ledger.total_calls < self.harness.max_calls:
            try:
                synth_out, _ = self.harness.run_structured(
                    synth_prompt,
                    ToTSynthesisOutput,
                    step_name="graph_tot_synthesize",
                    system_prompt=TOT_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=1024,
                )
            except ValueError:
                synth_out = None

        all_union_ents = list(union_entities.values())
        all_union_rels = list(union_relations.values())
        all_union_evs = list(union_evidences.values())

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
            # Graceful guardrail fallback if budget exhausted
            audited_steps, audit = audit_cot_grounding(
                [],
                entities=all_union_ents,
                relations=all_union_rels,
                evidences=all_union_evs,
            )
            win_desc = winner.hypothesis if winner else "当前检索子图证据不足"
            final_answer = f"[Graph-ToT 预算收敛] 最高分假设：{win_desc}。" + (
                "\n已剪枝假设：" + "；".join(pruned_memories) if pruned_memories else ""
            )
            confidence = winner.score if winner else 0.5

        ratio = round(max(0.0, 1.0 - compacted_synth_tokens / max(raw_initial_tokens, 1)), 4)
        compaction_meta: dict[str, object] = {
            "raw_tokens_est": raw_initial_tokens,
            "compacted_tokens_est": compacted_synth_tokens,
            "compression_ratio": ratio,
            "raw_entity_count": len(all_union_ents),
            "kept_entity_count": len(final_branches),
            "raw_relation_count": len(all_union_rels),
            "kept_relation_count": len(winner.supporting_relations) if winner else 0,
            "dropped_nodes": max(0, len(all_union_ents) - len(entities)),
            "applied_levels": [
                "L0_branch_subgraph_pruning",
                "L1_extractive_evidence",
                "L2_pruned_branch_folding",
            ],
        }

        result = ReasoningResult(
            question=question,
            paradigm="tot",
            anchor_entities=[b.target_entity_id for b in final_branches if b.target_entity_id],
            final_answer=final_answer,
            confidence=confidence,
            steps=audited_steps,
            branches=final_branches,
            pruned_memories=pruned_memories,
            grounding_audit=audit,
            compaction=compaction_meta,
            trace_ledger=self.harness.ledger.summary(),
        )
        return result, all_union_ents, all_union_rels, all_union_evs

    @staticmethod
    def _resolve_entity(
        target_entity_id: str,
        target_entity_name: str,
        hypothesis: str,
        ent_by_id: dict[str, Entity],
        ent_by_name: dict[str, Entity],
    ) -> Entity | None:
        cleaned_id = target_entity_id.strip()
        if cleaned_id in ent_by_id:
            return ent_by_id[cleaned_id]
        cleaned_name = target_entity_name.strip().lower()
        if cleaned_name in ent_by_name:
            return ent_by_name[cleaned_name]
        # Substring fallback against entity names
        hyp_low = f"{cleaned_name} {hypothesis.lower()}"
        for name_key in sorted(ent_by_name.keys(), key=len, reverse=True):
            if len(name_key) >= 2 and name_key in hyp_low:
                return ent_by_name[name_key]
        return None

    @staticmethod
    def _compute_deterministic_branch_score(
        *,
        resolved_ent: Entity | None,
        hypothesis: str,
        q_terms: set[str],
        seed_set: set[str],
        branch_entities: list[Entity],
        branch_relations: list[Relation],
        branch_evidences: list[EvidenceItem],
    ) -> float:
        if resolved_ent is None and not branch_entities:
            return 0.1
        b_eids = {e.entity_id for e in branch_entities}
        seed_overlap = len(b_eids & seed_set) / max(len(seed_set), 1) if seed_set else 0.3

        haystack = " ".join(
            [
                hypothesis,
                *(f"{e.name} {e.description}" for e in branch_entities[:12]),
                *(ev.text[:240] for ev in branch_evidences[:3]),
            ]
        ).lower()
        term_hits = sum(1 for t in q_terms if t in haystack)
        term_cov = min(1.0, term_hits / max(len(q_terms), 1)) if q_terms else 0.5
        rel_density = min(1.0, len(branch_relations) / 6.0)
        return round(min(1.0, 0.45 * seed_overlap + 0.40 * term_cov + 0.15 * rel_density), 4)

    def _format_branch_evidence_pack(
        self,
        *,
        branch_id: str,
        hypothesis: str,
        resolved_ent: Entity | None,
        branch_entities: list[Entity],
        branch_relations: list[Relation],
        branch_evidences: list[EvidenceItem],
        q_terms: set[str],
    ) -> tuple[str, str]:
        ent_map = {e.entity_id: e for e in branch_entities}
        ent_label = (
            f"[{resolved_ent.entity_id}] {resolved_ent.name} ({resolved_ent.entity_type})"
            if resolved_ent
            else "(Unresolved Entity)"
        )
        rel_lines = []
        for r in branch_relations[:6]:
            sn = ent_map[r.source_id].name if r.source_id in ent_map else r.source_id
            tn = ent_map[r.target_id].name if r.target_id in ent_map else r.target_id
            cids = f" chunks={r.source_chunk_ids[:2]}" if r.source_chunk_ids else ""
            rel_lines.append(f"  - [{r.relation_id}] {sn} --[{r.relation_type}]--> {tn}{cids}")

        ev_lines = []
        raw_texts = []
        # Stage 3.2 ToG-2 fix: remove destructive 220-char truncation; preserve complete key sentences
        for ev in branch_evidences[:3]:
            raw_texts.append(ev.text)
            if self.enable_compact and len(ev.text) > 900:
                snippet = self.compactor._extract_key_sentences(
                    ev.text, anchor_terms=q_terms, max_sentences=4
                )
            else:
                snippet = ev.text.strip()
            ev_lines.append(f"  - [{ev.chunk_id}]: {snippet}")

        pack = (
            f"=== Branch {branch_id}: {hypothesis} ===\n"
            f"Target Entity: {ent_label}\n"
            "Grounded Relations:\n"
            + ("\n".join(rel_lines) if rel_lines else "  - (none)")
            + "\nGrounded Evidences:\n"
            + ("\n".join(ev_lines) if ev_lines else "  - (none)")
        )
        return pack, " ".join(raw_texts)

    def _execute_additive_synthesis(
        self,
        *,
        question: str,
        initial_entities: list[Entity],
        initial_evidences: list[EvidenceItem],
        communities: list[Community],
        grounded_branches: list[ThoughtBranch],
        branch_evidence_packs: dict[str, str],
        union_entities: dict[str, Entity],
        union_relations: dict[str, Relation],
        union_evidences: dict[str, EvidenceItem],
        raw_initial_tokens: int,
        seed_context: str,
    ) -> tuple[ReasoningResult, list[Entity], list[Relation], list[EvidenceItem]]:
        """Stage 3.2 Non-Pruning Multi-Branch Additive Synthesis (2 LLM calls total).

        Retains ALL complementary branches above `min_branch_score` and merges their full
        grounded evidence union with top PPR/RRF passages so multi-facet / summary queries
        never lose complementary information.
        """
        ranked = sorted(grounded_branches, key=lambda b: (-b.deterministic_graph_score, b.branch_id))
        final_branches: list[ThoughtBranch] = []
        retained_packs: list[str] = []
        retained_chunk_ids: list[str] = []

        for rank, br in enumerate(ranked):
            keep = rank == 0 or br.deterministic_graph_score >= self.min_branch_score
            if keep:
                kept_br = br.model_copy(
                    update={
                        "status": "selected",
                        "semantic_score": br.deterministic_graph_score,
                        "score": br.deterministic_graph_score,
                        "prune_reason": "",
                    }
                )
                final_branches.append(kept_br)
                if br.branch_id in branch_evidence_packs:
                    retained_packs.append(branch_evidence_packs[br.branch_id])
                retained_chunk_ids.extend(br.supporting_chunks)
            else:
                final_branches.append(
                    br.model_copy(
                        update={
                            "status": "pruned",
                            "prune_reason": (
                                f"图谱支持度 ({br.deterministic_graph_score:.2f}) "
                                f"低于多分支聚合阈值 ({self.min_branch_score:.2f})。"
                            ),
                        }
                    )
                )

        # Merge full union of top global PPR/RRF passages + branch-discovered passages (lossless!)
        seen_cids: set[str] = set()
        merged_ev_lines: list[str] = []
        # Prioritize initial top RRF/PPR passages first, then unique branch passages
        ordered_evs: list[EvidenceItem] = [*initial_evidences[:6]]
        for cid in retained_chunk_ids:
            if cid in union_evidences:
                ordered_evs.append(union_evidences[cid])
        for ev in ordered_evs:
            if ev.chunk_id in seen_cids:
                continue
            seen_cids.add(ev.chunk_id)
            merged_ev_lines.append(f"- [{ev.chunk_id} | {ev.source} | via={ev.matched_via}]: {ev.text}")
            if len(merged_ev_lines) >= 8:
                break

        comm_block = (
            "Community Summaries:\n"
            + "\n".join(f"- [{c.title}] {c.summary[:220]}" for c in communities[:3])
            if communities
            else ""
        )
        branches_block = (
            "Complementary Facet Branches (All Retained — Non-Pruning Union):\n"
            + "\n\n".join(retained_packs)
            if retained_packs
            else seed_context
        )
        evidences_block = (
            "Full Deduplicated Source Passages (Union of PPR/RRF + All Branches):\n"
            + "\n".join(merged_ev_lines)
            if merged_ev_lines
            else ""
        )

        # Keep concise graph overview + full deduplicated passages within context budget
        synth_context = "\n\n".join(s for s in (branches_block, comm_block, evidences_block) if s)
        # If total exceeds compactor budget, drop only excess redundant branch lines, never full passages
        if estimate_tokens(synth_context) > self.compactor.max_context_tokens + 1200:
            compact_branch_summary = "\n".join(
                f"- [{b.branch_id}] {b.hypothesis} (target={b.target_entity_name}, "
                f"rels={b.supporting_relations[:4]}, chunks={b.supporting_chunks[:3]})"
                for b in final_branches
                if b.status == "selected"
            )
            synth_context = "\n\n".join(
                s
                for s in (
                    "Retained Facet Branches:\n" + compact_branch_summary,
                    comm_block,
                    evidences_block,
                )
                if s
            )

        compacted_synth_tokens = estimate_tokens(synth_context)

        synth_instruction = (
            "GRAPH_MULTI_BRANCH_SYNTH_TASK\n"
            "Synthesize a comprehensive, accurate, grounded answer in the EXACT SAME LANGUAGE "
            "as the question by integrating ALL complementary facts across ALL retained branches "
            "and source passages above. Do NOT omit secondary factors or details present in the "
            "passages. Cite ONLY exact entity_ids, relation_ids, and chunk_ids present above.\n"
            "Return strict JSON:\n"
            '{"selected_branch_id": "ALL_RETAINED", '
            '"steps": [{"step_id": 1, "stage": "anchor|graph_traverse|evidence_check|synthesis", '
            '"claim": "<grounded facet synthesis>", "cited_entity_ids": ["ent_..."], '
            '"cited_relation_ids": ["rel_..."], "cited_chunk_ids": ["doc_...::chunk_..."]}], '
            '"excluded_summary": [], '
            '"final_answer": "<thorough yet focused answer integrating all grounded evidence>", '
            '"confidence": 0.9}'
        )
        synth_prompt = LLMHarness.compile_prompt(
            context_text=synth_context,
            question=question,
            task_instruction=synth_instruction,
        )

        synth_out: ToTSynthesisOutput | None = None
        if self.harness.ledger.total_calls < self.harness.max_calls:
            try:
                synth_out, _ = self.harness.run_structured(
                    synth_prompt,
                    ToTSynthesisOutput,
                    step_name="graph_multi_branch_synthesize",
                    system_prompt=TOT_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=1536,
                )
            except ValueError:
                synth_out = None

        all_union_ents = list(union_entities.values())
        all_union_rels = list(union_relations.values())
        all_union_evs = list(union_evidences.values())

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
            facets = "; ".join(b.hypothesis for b in final_branches if b.status == "selected")
            final_answer = f"[Multi-Branch Fallback] Retained facets: {facets}"
            confidence = 0.6

        ratio = round(max(0.0, 1.0 - compacted_synth_tokens / max(raw_initial_tokens, 1)), 4)
        kept_cnt = sum(1 for b in final_branches if b.status == "selected")
        compaction_meta: dict[str, object] = {
            "raw_tokens_est": raw_initial_tokens,
            "compacted_tokens_est": compacted_synth_tokens,
            "compression_ratio": ratio,
            "raw_entity_count": len(all_union_ents),
            "kept_entity_count": kept_cnt,
            "raw_relation_count": len(all_union_rels),
            "kept_relation_count": sum(
                len(b.supporting_relations) for b in final_branches if b.status == "selected"
            ),
            "dropped_nodes": max(0, len(all_union_ents) - len(initial_entities)),
            "applied_levels": [
                "L0_multi_branch_union",
                "L1_full_passage_preservation",
            ],
        }

        result = ReasoningResult(
            question=question,
            paradigm="multi_branch",
            anchor_entities=[b.target_entity_id for b in final_branches if b.target_entity_id],
            final_answer=final_answer,
            confidence=confidence,
            steps=audited_steps,
            branches=final_branches,
            pruned_memories=[],
            grounding_audit=audit,
            compaction=compaction_meta,
            trace_ledger=self.harness.ledger.summary(),
        )
        return result, all_union_ents, all_union_rels, all_union_evs

"""Paper-faithful Iterative Graph-CoT v2 engine (Graph-CoT ACL 2024 + PoG AAAI 2025).

Executes a genuine multi-turn Thought -> Graph-Action -> Observation loop where the model
actively inspects KG entities, traverses real outgoing/incoming edges, fetches source chunks
on demand, self-corrects on dead ends, and converges to a GroundingAudit-verified answer.

Supports dual-model execution:
- `planner_harness` (e.g., fast `qwen3.5:4b` OR unified `qwen3.5:9b`) drives intermediate
  Thought -> Action tool selection.
- `synth_harness` (primary `qwen3.5:9b`) produces the final grounded synthesis.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient, BaseGraphStore, BaseVectorStore
from llm_graphrag_kg.core.schema import Community, Entity, EvidenceItem, Relation
from llm_graphrag_kg.harness.context_compactor import ContextCompactor, estimate_tokens
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.pipelines.hipporag import SemanticGraphIndex
from llm_graphrag_kg.reasoning.cot import COT_SYSTEM_PROMPT, audit_cot_grounding
from llm_graphrag_kg.reasoning.schema import ReasoningResult, ReasoningStep

ITERATIVE_COT_SYSTEM_PROMPT = (
    "You are a paper-faithful Iterative Graph-CoT agent reasoning over a Knowledge Graph. "
    "At each iteration, write a concise `thought`, then pick ONE graph tool action:\n"
    "- RetrieveNode(arg='<keywords>'): search KG entities semantically/lexically.\n"
    "- InspectEntity(arg='<entity_id>'): inspect an entity's description, source chunks, and degree.\n"
    "- ExpandNeighbors(arg='<entity_id>'): list real KG relations and neighbor entities.\n"
    "- FetchChunk(arg='<chunk_id>'): read the full original text of a specific source chunk.\n"
    "- Finish(arg='<brief readiness summary>'): stop exploring when evidence is sufficient.\n"
    "Cite ONLY exact IDs observed so far. Return ONLY valid JSON."
)


class GraphCoTActionOutput(BaseModel):
    """Single Thought -> Action step emitted by the iterative Graph-CoT planner."""

    thought: str
    action: Literal["RetrieveNode", "InspectEntity", "ExpandNeighbors", "FetchChunk", "Finish"]
    arg: str = ""
    cited_entity_ids: list[str] = Field(default_factory=list)
    cited_relation_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)


class IterativeSynthesisOutput(BaseModel):
    """Final grounded synthesis grounded on all collected Thought-Action-Observation evidence."""

    steps: list[ReasoningStep] = Field(default_factory=list)
    final_answer: str
    confidence: float = Field(default=0.88, ge=0.0, le=1.0)


def _model_label(harness: LLMHarness) -> str:
    return str(getattr(harness.llm, "model", type(harness.llm).__name__))


class IterativeGraphCoTEngine:
    """Multi-turn Thought -> Graph-Action -> Observation engine with dual-model telemetry."""

    def __init__(
        self,
        synth_harness: LLMHarness,
        *,
        planner_harness: LLMHarness | None = None,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        embedder: BaseEmbeddingClient | None = None,
        semantic_index: SemanticGraphIndex | None = None,
        compactor: ContextCompactor | None = None,
        max_iters: int = 3,
        temperature: float = 0.1,
    ) -> None:
        self.synth_harness = synth_harness
        self.planner_harness = planner_harness or synth_harness
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.semantic_index = semantic_index
        self.compactor = compactor or synth_harness.compactor
        self.max_iters = max(1, min(max_iters, 5))
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
        """Execute interactive Graph-CoT tool loop and synthesize grounded answer."""
        self.synth_harness.reset_ledger()
        if self.planner_harness is not self.synth_harness:
            self.planner_harness.reset_ledger()

        all_store_ents = {e.entity_id: e for e in self.graph_store.get_all_entities()}
        all_store_rels = self.graph_store.get_all_relations()
        incident_rels: dict[str, list[Relation]] = {}
        for r in all_store_rels:
            incident_rels.setdefault(r.source_id, []).append(r)
            incident_rels.setdefault(r.target_id, []).append(r)

        observed_ents: dict[str, Entity] = {e.entity_id: e for e in entities}
        observed_rels: dict[str, Relation] = {r.relation_id: r for r in relations}
        observed_evs: dict[str, EvidenceItem] = {ev.chunk_id: ev for ev in evidences}
        fetched_chunk_ids: set[str] = set()

        seed_summary = "\n".join(
            f"- [{e.entity_id}] {e.name} ({e.entity_type})"
            + (f" chunks={e.source_chunk_ids[:2]}" if e.source_chunk_ids else "")
            for e in entities[:10]
        ) or "(No initial seed entities)"
        init_ev_summary = "\n".join(
            f"- [{ev.chunk_id} | {ev.source}]: {ev.text[:260]}..."
            for ev in evidences[:4]
        ) or "(No initial passages)"

        scratchpad: list[str] = []
        action_steps: list[ReasoningStep] = []
        planner_calls = 0
        planner_seconds = 0.0
        planner_load_seconds = 0.0
        self_corrections = 0

        for step_idx in range(1, self.max_iters + 1):
            history_block = "\n\n".join(scratchpad) if scratchpad else "(No tool calls yet)"
            step_prompt = (
                f"GRAPH_COT_ITER_ACTION_TASK (Iteration {step_idx}/{self.max_iters})\n"
                f"Question: {question}\n\n"
                f"Initial Seed Entities:\n{seed_summary}\n\n"
                f"Initial Top Passages (previews; call FetchChunk to read full text):\n"
                f"{init_ev_summary}\n\n"
                f"Previous Thought-Action-Observation History:\n{history_block}\n\n"
                "Choose the single most informative next graph tool action, or choose `Finish` "
                "if you already have enough grounded facts and passages to answer accurately.\n"
                "Return strict JSON:\n"
                '{"thought": "<1-2 concise sentences>", '
                '"action": "RetrieveNode|InspectEntity|ExpandNeighbors|FetchChunk|Finish", '
                '"arg": "<entity_id or chunk_id or search query>", '
                '"cited_entity_ids": ["ent_..."], "cited_relation_ids": ["rel_..."], '
                '"cited_chunk_ids": ["doc_...::chunk_..."]}'
            )

            t0 = time.perf_counter()
            try:
                act_out, _ = self.planner_harness.run_structured(
                    step_prompt,
                    GraphCoTActionOutput,
                    step_name=f"graph_cot_iter_{step_idx}",
                    system_prompt=ITERATIVE_COT_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=420,
                )
            except ValueError:
                break
            dt = time.perf_counter() - t0
            planner_calls += 1
            planner_seconds += dt
            tel = getattr(self.planner_harness.llm, "last_telemetry", None)
            if isinstance(tel, dict):
                planner_load_seconds += float(tel.get("load_seconds", 0.0))

            obs_text, obs_eids, obs_rids, obs_cids, missed = self._execute_tool(
                action=act_out.action,
                arg=act_out.arg,
                all_store_ents=all_store_ents,
                incident_rels=incident_rels,
                observed_ents=observed_ents,
                observed_rels=observed_rels,
                observed_evs=observed_evs,
                fetched_chunk_ids=fetched_chunk_ids,
            )
            if missed:
                self_corrections += 1

            step_eids = list(dict.fromkeys([*act_out.cited_entity_ids, *obs_eids]))[:6]
            step_rids = list(dict.fromkeys([*act_out.cited_relation_ids, *obs_rids]))[:6]
            step_cids = list(dict.fromkeys([*act_out.cited_chunk_ids, *obs_cids]))[:4]
            action_steps.append(
                ReasoningStep(
                    step_id=step_idx,
                    stage="synthesis" if act_out.action == "Finish" else "tool_call",
                    claim=f"[{act_out.action}({act_out.arg[:48]})] {act_out.thought}",
                    cited_entity_ids=step_eids,
                    cited_relation_ids=step_rids,
                    cited_chunk_ids=step_cids,
                    tool_action=act_out.action,
                    tool_args={"arg": act_out.arg},
                    observation=obs_text[:360],
                )
            )
            scratchpad.append(
                f"Iter {step_idx} Thought: {act_out.thought}\n"
                f"Iter {step_idx} Action: {act_out.action}({act_out.arg})\n"
                f"Iter {step_idx} Observation:\n{obs_text}"
            )
            if act_out.action == "Finish":
                break

        # Final synthesis on synth_harness using ALL fetched + top retrieved full passages
        all_ents_list = list(observed_ents.values())
        all_rels_list = list(observed_rels.values())
        all_evs_list = list(observed_evs.values())

        # Prioritize explicitly fetched chunks, then top initial evidences
        ordered_evs: list[EvidenceItem] = []
        seen_cids: set[str] = set()
        for cid in fetched_chunk_ids:
            if cid in observed_evs and cid not in seen_cids:
                ordered_evs.append(observed_evs[cid])
                seen_cids.add(cid)
        for ev in evidences[:6]:
            if ev.chunk_id not in seen_cids:
                ordered_evs.append(ev)
                seen_cids.add(ev.chunk_id)
        for ev in all_evs_list:
            if len(ordered_evs) >= 8:
                break
            if ev.chunk_id not in seen_cids:
                ordered_evs.append(ev)
                seen_cids.add(ev.chunk_id)

        ev_lines = [
            f"- [{ev.chunk_id} | {ev.source} | via={ev.matched_via}]: {ev.text}"
            for ev in ordered_evs[:8]
        ]
        comm_lines = [f"- [{c.title}] {c.summary[:180]}" for c in communities[:2]]
        synth_context = "\n\n".join(
            s
            for s in (
                "Interactive Graph-CoT Tool Trace:\n" + "\n\n".join(scratchpad) if scratchpad else "",
                "Community Context:\n" + "\n".join(comm_lines) if comm_lines else "",
                "Full Grounded Source Passages:\n" + "\n".join(ev_lines) if ev_lines else "",
            )
            if s
        )
        raw_tokens = estimate_tokens(synth_context)

        synth_instruction = (
            "GRAPH_COT_ITER_SYNTH_TASK\n"
            "Using ONLY the verified Graph-CoT tool trace and full source passages above, "
            "produce a concise, accurate, grounded final answer in the EXACT SAME LANGUAGE "
            "as the question. Cite exact chunk_ids/entity_ids/relation_ids.\n"
            "Return strict JSON:\n"
            '{"steps": [{"step_id": 1, "stage": "anchor|graph_traverse|evidence_check|synthesis", '
            '"claim": "<concise step>", "cited_entity_ids": ["ent_..."], '
            '"cited_relation_ids": ["rel_..."], "cited_chunk_ids": ["doc_...::chunk_..."]}], '
            '"final_answer": "<grounded answer>", "confidence": 0.9}'
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
                IterativeSynthesisOutput,
                step_name="graph_cot_iter_synthesize",
                system_prompt=COT_SYSTEM_PROMPT,
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

        # Combine iterative tool trace steps + final synthesis step(s) for auditable replay
        combined_steps = list(action_steps)
        if synth_out is not None:
            offset = len(combined_steps)
            for st in synth_out.steps:
                combined_steps.append(st.model_copy(update={"step_id": offset + st.step_id}))
            final_answer = synth_out.final_answer
            confidence = synth_out.confidence
        else:
            final_answer = (
                action_steps[-1].claim if action_steps else "Insufficient grounded graph evidence."
            )
            confidence = 0.6

        audited_steps, audit = audit_cot_grounding(
            combined_steps,
            entities=all_ents_list,
            relations=all_rels_list,
            evidences=all_evs_list,
        )

        # Merge ledgers so total_calls reflects planner + synth accurately
        synth_summary = self.synth_harness.ledger.summary()
        total_calls = planner_calls + synth_calls
        merged_ledger: dict[str, object] = {
            **synth_summary,
            "total_calls": total_calls,
            "planner_model": _model_label(self.planner_harness),
            "planner_calls": planner_calls,
            "planner_seconds": round(planner_seconds, 3),
            "planner_load_seconds": round(planner_load_seconds, 3),
            "synth_model": _model_label(self.synth_harness),
            "synth_calls": synth_calls,
            "synth_seconds": round(synth_seconds, 3),
            "synth_load_seconds": round(synth_load_seconds, 3),
            "iterations": len(action_steps),
            "self_corrections": self_corrections,
        }

        res = ReasoningResult(
            question=question,
            paradigm="cot_iter",
            anchor_entities=list(seed_entity_ids[:6]),
            final_answer=final_answer,
            confidence=confidence,
            steps=audited_steps,
            grounding_audit=audit,
            compaction={
                "raw_tokens_est": raw_tokens,
                "compacted_tokens_est": raw_tokens,
                "compression_ratio": 0.0,
                "applied_levels": ["iterative_on_demand_tool_fetch"],
            },
            trace_ledger=merged_ledger,
        )
        return res, all_ents_list, all_rels_list, all_evs_list

    def _execute_tool(
        self,
        *,
        action: str,
        arg: str,
        all_store_ents: dict[str, Entity],
        incident_rels: dict[str, list[Relation]],
        observed_ents: dict[str, Entity],
        observed_rels: dict[str, Relation],
        observed_evs: dict[str, EvidenceItem],
        fetched_chunk_ids: set[str],
    ) -> tuple[str, list[str], list[str], list[str], bool]:
        """Run one deterministic KG tool call and return (observation, eids, rids, cids, missed)."""
        cleaned = arg.strip().strip("[]'\"")
        if action == "Finish":
            return f"Finished exploration: {cleaned or 'ready to synthesize.'}", [], [], [], False

        if action == "RetrieveNode":
            hits = self.graph_store.find_entities_by_query(cleaned, top_k=5)
            if self.embedder is not None and self.semantic_index is not None and self.semantic_index.is_ready:
                q_vec = self.embedder.embed([cleaned])[0]
                for eid, _sc in self.semantic_index.search_entities(q_vec, top_k=4):
                    ent_obj = all_store_ents.get(eid)
                    if ent_obj is not None and all(h.entity_id != eid for h in hits):
                        hits.append(ent_obj)
            if not hits:
                return (
                    f"[SELF-CORRECT] No entities matched `{cleaned}`. "
                    "Try broader terms or ExpandNeighbors on an existing seed.",
                    [],
                    [],
                    [],
                    True,
                )
            lines = []
            eids = []
            for e in hits[:5]:
                observed_ents[e.entity_id] = e
                eids.append(e.entity_id)
                lines.append(
                    f"- [{e.entity_id}] {e.name} ({e.entity_type}): {e.description[:120]} "
                    f"chunks={e.source_chunk_ids[:2]}"
                )
            return "\n".join(lines), eids, [], [], False

        if action == "InspectEntity":
            ent = self._resolve_ent(cleaned, all_store_ents)
            if ent is None:
                return (
                    f"[SELF-CORRECT] Entity `{cleaned}` not found. Use an exact [ent_...] ID or call RetrieveNode.",
                    [],
                    [],
                    [],
                    True,
                )
            observed_ents[ent.entity_id] = ent
            rels = incident_rels.get(ent.entity_id, [])
            cids = list(ent.source_chunk_ids[:3])
            for cid in cids:
                ch = self.vector_store.get_chunk(cid)
                if ch is not None and cid not in observed_evs:
                    observed_evs[cid] = EvidenceItem(
                        chunk_id=ch.chunk_id,
                        doc_id=ch.doc_id,
                        source=ch.source,
                        text=ch.text,
                        score=0.9,
                        matched_via="cot_tool_inspect",
                    )
            obs = (
                f"Entity [{ent.entity_id}] {ent.name} ({ent.entity_type})\n"
                f"Description: {ent.description[:220]}\n"
                f"Degree: {len(rels)} relations | Source Chunks: {cids}"
            )
            return obs, [ent.entity_id], [], cids, False

        if action == "ExpandNeighbors":
            ent = self._resolve_ent(cleaned, all_store_ents)
            if ent is None:
                return (
                    f"[SELF-CORRECT] Cannot expand `{cleaned}`: entity not found. Use RetrieveNode first.",
                    [],
                    [],
                    [],
                    True,
                )
            observed_ents[ent.entity_id] = ent
            rels = sorted(
                incident_rels.get(ent.entity_id, []),
                key=lambda r: (0 if r.relation_type != "CO_OCCURS_WITH" else 1, -float(r.weight)),
            )[:8]
            if not rels:
                return f"Entity [{ent.entity_id}] {ent.name} has no incident edges.", [ent.entity_id], [], [], True
            lines = []
            eids = [ent.entity_id]
            rids = []
            cids = []
            for r in rels:
                observed_rels[r.relation_id] = r
                rids.append(r.relation_id)
                other_id = r.target_id if r.source_id == ent.entity_id else r.source_id
                other = all_store_ents.get(other_id)
                if other is not None:
                    observed_ents[other.entity_id] = other
                    eids.append(other.entity_id)
                src_n = all_store_ents[r.source_id].name if r.source_id in all_store_ents else r.source_id
                tgt_n = all_store_ents[r.target_id].name if r.target_id in all_store_ents else r.target_id
                cids.extend(r.source_chunk_ids[:1])
                lines.append(
                    f"- [{r.relation_id}] {src_n} --[{r.relation_type}]--> {tgt_n} "
                    f"(neighbor=[{other_id}], chunks={r.source_chunk_ids[:2]})"
                )
            return "\n".join(lines), eids, rids, list(dict.fromkeys(cids))[:4], False

        if action == "FetchChunk":
            ch = self.vector_store.get_chunk(cleaned)
            if ch is None:
                return (
                    f"[SELF-CORRECT] Chunk `{cleaned}` not found. Pick a chunk_id from observed entities/passages.",
                    [],
                    [],
                    [],
                    True,
                )
            fetched_chunk_ids.add(ch.chunk_id)
            observed_evs[ch.chunk_id] = EvidenceItem(
                chunk_id=ch.chunk_id,
                doc_id=ch.doc_id,
                source=ch.source,
                text=ch.text,
                score=1.0,
                matched_via="cot_tool_fetch",
            )
            return f"[{ch.chunk_id} | {ch.source}]: {ch.text}", [], [], [ch.chunk_id], False

        return f"[SELF-CORRECT] Unknown action `{action}`.", [], [], [], True

    @staticmethod
    def _resolve_ent(raw: str, all_store_ents: dict[str, Entity]) -> Entity | None:
        if raw in all_store_ents:
            return all_store_ents[raw]
        low = raw.lower()
        for ent in all_store_ents.values():
            if ent.name.strip().lower() == low:
                return ent
            if any(a.strip().lower() == low for a in ent.aliases):
                return ent
        return None

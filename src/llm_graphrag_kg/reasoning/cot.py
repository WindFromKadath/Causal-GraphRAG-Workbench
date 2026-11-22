"""Explicit Graph-CoT engine with deterministic GroundingAudit (Stage 2.2)."""

from __future__ import annotations

from llm_graphrag_kg.core.schema import Community, Entity, EvidenceItem, Relation
from llm_graphrag_kg.harness.context_compactor import ContextCompactor
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.reasoning.schema import (
    GraphCoTOutput,
    GroundingAudit,
    ReasoningResult,
    ReasoningStep,
)

COT_SYSTEM_PROMPT = (
    "You are an auditable Graph-CoT reasoning engine. "
    "You MUST follow an explicit 4-stage reasoning chain over the provided Knowledge Graph "
    "and source text evidences: "
    "(1) `anchor`: identify query anchor entities; "
    "(2) `graph_traverse`: cite exact retrieved relation IDs connecting entities; "
    "(3) `evidence_check`: verify thresholds/rules/details against cited chunk IDs; "
    "(4) `synthesis`: produce the grounded final answer in Chinese (or the question's language). "
    "NEVER invent entity_id, relation_id, or chunk_id values that do not appear in the context. "
    "Return ONLY valid JSON matching the schema."
)


def audit_cot_grounding(
    steps: list[ReasoningStep],
    *,
    entities: list[Entity],
    relations: list[Relation],
    evidences: list[EvidenceItem],
    min_grounding_score: float = 0.75,
) -> tuple[list[ReasoningStep], GroundingAudit]:
    """Deterministically verify every step's entity/relation/chunk citations against retrieved facts."""
    valid_entity_ids = {e.entity_id for e in entities}
    entity_name_to_id: dict[str, str] = {}
    for ent in entities:
        entity_name_to_id[ent.name.strip().lower()] = ent.entity_id
        for alias in ent.aliases:
            entity_name_to_id[alias.strip().lower()] = ent.entity_id

    valid_relation_ids = {r.relation_id for r in relations}
    valid_chunk_ids: set[str] = {ev.chunk_id for ev in evidences}
    for ent in entities:
        valid_chunk_ids.update(ent.source_chunk_ids)
    for rel in relations:
        valid_chunk_ids.update(rel.source_chunk_ids)

    audited_steps: list[ReasoningStep] = []
    valid_e_used: list[str] = []
    valid_r_used: list[str] = []
    valid_c_used: list[str] = []
    ungrounded_all: list[str] = []
    grounded_steps = 0
    prior_grounded = False

    for idx, step in enumerate(steps, start=1):
        notes: list[str] = []
        resolved_eids: list[str] = []
        resolved_rids: list[str] = []
        resolved_cids: list[str] = []
        bad_citations: list[str] = []

        for raw_e in step.cited_entity_ids:
            cleaned = str(raw_e).strip()
            if not cleaned:
                continue
            if cleaned in valid_entity_ids:
                resolved_eids.append(cleaned)
                valid_e_used.append(cleaned)
            elif cleaned.lower() in entity_name_to_id:
                mapped = entity_name_to_id[cleaned.lower()]
                resolved_eids.append(mapped)
                valid_e_used.append(mapped)
            else:
                bad_citations.append(f"entity:{cleaned}")
                notes.append(f"Ungrounded entity citation: {cleaned}")

        for raw_r in step.cited_relation_ids:
            cleaned = str(raw_r).strip()
            if not cleaned:
                continue
            if cleaned in valid_relation_ids:
                resolved_rids.append(cleaned)
                valid_r_used.append(cleaned)
            else:
                bad_citations.append(f"relation:{cleaned}")
                notes.append(f"Ungrounded relation citation: {cleaned}")

        for raw_c in step.cited_chunk_ids:
            cleaned = str(raw_c).strip().strip("[]")
            if not cleaned:
                continue
            if cleaned in valid_chunk_ids:
                resolved_cids.append(cleaned)
                valid_c_used.append(cleaned)
            else:
                bad_citations.append(f"chunk:{cleaned}")
                notes.append(f"Ungrounded chunk citation: {cleaned}")

        ungrounded_all.extend(bad_citations)
        has_valid_citation = bool(resolved_eids or resolved_rids or resolved_cids)
        is_synthesis_carry = step.stage.strip().lower() == "synthesis" and prior_grounded

        # If the retrieved graph itself was empty, allow synthesis without fake IDs
        empty_graph_fallback = not (valid_entity_ids or valid_relation_ids or valid_chunk_ids)

        is_grounded = (not bad_citations) and (
            has_valid_citation or is_synthesis_carry or empty_graph_fallback
        )
        if not is_grounded and not notes:
            notes.append("Step cites no verified entity_id, relation_id, or chunk_id.")

        if is_grounded:
            grounded_steps += 1
            prior_grounded = True

        audited_steps.append(
            step.model_copy(
                update={
                    "step_id": step.step_id or idx,
                    "cited_entity_ids": list(dict.fromkeys(resolved_eids)),
                    "cited_relation_ids": list(dict.fromkeys(resolved_rids)),
                    "cited_chunk_ids": list(dict.fromkeys(resolved_cids)),
                    "grounded": is_grounded,
                    "grounding_notes": notes,
                }
            )
        )

    total = len(audited_steps)
    score = round(grounded_steps / total, 4) if total else 0.0
    ungrounded_unique = list(dict.fromkeys(ungrounded_all))
    passed = total > 0 and score >= min_grounding_score and not ungrounded_unique

    audit = GroundingAudit(
        total_steps=total,
        grounded_steps=grounded_steps,
        grounding_score=score,
        valid_cited_entities=list(dict.fromkeys(valid_e_used)),
        valid_cited_relations=list(dict.fromkeys(valid_r_used)),
        valid_cited_chunks=list(dict.fromkeys(valid_c_used)),
        ungrounded_citations=ungrounded_unique,
        passed=passed,
    )
    return audited_steps, audit


def build_auditable_cot_context(
    *,
    question: str,
    seed_entity_ids: list[str],
    entities: list[Entity],
    relations: list[Relation],
    communities: list[Community],
    evidences: list[EvidenceItem],
    compactor: ContextCompactor | None = None,
    enable_compact: bool = True,
) -> tuple[str, dict[str, object] | None]:
    """Render an ID-explicit compacted context so CoT steps can cite exact KG/chunk IDs."""
    ent_map = {e.entity_id: e for e in entities}
    compaction_meta: dict[str, object] | None = None

    if enable_compact and compactor is not None:
        compacted = compactor.compact_graphrag_context(
            question=question,
            seed_entity_ids=seed_entity_ids,
            entities=entities,
            relations=relations,
            communities=communities,
            evidences=evidences,
        )
        compaction_meta = compacted.report.model_dump()
        kept_e_cap = max(compactor.max_entities, 1)
        kept_r_cap = max(compactor.max_relations, 1)
        evidences_block = compacted.evidences_block
        communities_block = compacted.communities_block
    else:
        kept_e_cap = 24
        kept_r_cap = 30
        communities_block = (
            "Community Summaries:\n"
            + "\n".join(f"- [{c.title}] {c.summary[:200]}" for c in communities[:3])
            if communities
            else ""
        )
        evidences_block = (
            "Source Text Evidences:\n"
            + "\n".join(
                f"- [{ev.chunk_id} | {ev.source} | via={ev.matched_via}]: {ev.text}"
                for ev in evidences[:6]
            )
            if evidences
            else ""
        )

    seed_set = set(seed_entity_ids)
    ranked_ents = sorted(
        entities,
        key=lambda e: (0 if e.entity_id in seed_set else 1, e.name),
    )[:kept_e_cap]
    ranked_rels = sorted(
        relations,
        key=lambda r: (
            0
            if (r.source_id in seed_set and r.target_id in seed_set)
            else (1 if (r.source_id in seed_set or r.target_id in seed_set) else 2),
            -r.weight,
            r.relation_id,
        ),
    )[:kept_r_cap]

    ent_lines = [
        f"- [{e.entity_id}] {e.name} ({e.entity_type})"
        + (f": {e.description[:85]}" if e.description else "")
        for e in ranked_ents
    ]
    rel_lines = []
    for r in ranked_rels:
        s_name = ent_map[r.source_id].name if r.source_id in ent_map else r.source_id
        t_name = ent_map[r.target_id].name if r.target_id in ent_map else r.target_id
        cids = f" chunks={r.source_chunk_ids[:2]}" if r.source_chunk_ids else ""
        rel_lines.append(
            f"- [{r.relation_id}] {s_name} --[{r.relation_type}]--> {t_name}{cids}"
        )

    sections = []
    if ent_lines:
        sections.append("Auditable KG Entities (cite exact [ent_...] IDs):\n" + "\n".join(ent_lines))
    if rel_lines:
        sections.append("Auditable KG Relations (cite exact [rel_...] IDs):\n" + "\n".join(rel_lines))
    if communities_block:
        sections.append(communities_block)
    if evidences_block:
        sections.append(evidences_block)

    return "\n\n".join(sections) if sections else "(Empty retrieved context)", compaction_meta


class GraphCoTEngine:
    """Executes explicit 4-stage Graph-CoT via LLMHarness with GroundingAudit."""

    def __init__(
        self,
        harness: LLMHarness,
        *,
        compactor: ContextCompactor | None = None,
        enable_compact: bool = True,
        min_grounding_score: float = 0.75,
        self_correct_grounding: bool = True,
        temperature: float = 0.1,
    ) -> None:
        self.harness = harness
        self.compactor = compactor or harness.compactor
        self.enable_compact = enable_compact
        self.min_grounding_score = min_grounding_score
        self.self_correct_grounding = self_correct_grounding
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
    ) -> ReasoningResult:
        """Run Explicit Graph-CoT and deterministically audit step-level grounding."""
        context_text, compaction_meta = build_auditable_cot_context(
            question=question,
            seed_entity_ids=seed_entity_ids,
            entities=entities,
            relations=relations,
            communities=communities,
            evidences=evidences,
            compactor=self.compactor,
            enable_compact=self.enable_compact,
        )

        task_instruction = (
            "GRAPH_COT_TASK\n"
            "Produce an explicit 4-step Graph-CoT reasoning chain in strict JSON:\n"
            "{\n"
            '  "anchor_entities": ["<entity_name_or_id>"],\n'
            '  "steps": [\n'
            "    {\n"
            '      "step_id": 1,\n'
            '      "stage": "anchor | graph_traverse | evidence_check | synthesis",\n'
            '      "claim": "concise factual step in Chinese",\n'
            '      "cited_entity_ids": ["ent_..."],\n'
            '      "cited_relation_ids": ["rel_..."],\n'
            '      "cited_chunk_ids": ["doc_...::chunk_..."]\n'
            "    }\n"
            "  ],\n"
            '  "final_answer": "grounded structured answer citing chunk IDs",\n'
            '  "confidence": 0.9\n'
            "}\n"
            "Rule: Every cited ID MUST copy an exact ID from the Auditable KG / Evidences above."
        )

        prompt = LLMHarness.compile_prompt(
            context_text=context_text,
            question=question,
            task_instruction=task_instruction,
        )

        self.harness.reset_ledger()
        cot_out, _ = self.harness.run_structured(
            prompt,
            GraphCoTOutput,
            step_name="graph_cot_primary",
            system_prompt=COT_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=1536,
        )

        audited_steps, audit = audit_cot_grounding(
            cot_out.steps,
            entities=entities,
            relations=relations,
            evidences=evidences,
            min_grounding_score=self.min_grounding_score,
        )

        # Optional single-pass grounding self-correction when fabricated IDs are detected
        if (
            self.self_correct_grounding
            and not audit.passed
            and (entities or relations or evidences)
            and self.harness.ledger.total_calls < self.harness.max_calls
        ):
            repair_prompt = (
                f"{prompt}\n\n"
                "--- GROUNDING AUDIT FEEDBACK ---\n"
                f"Your previous draft failed grounding verification (score={audit.grounding_score}).\n"
                f"Ungrounded / fabricated citations detected: {audit.ungrounded_citations}\n"
                "Rewrite the JSON so EVERY step cites ONLY exact entity_ids, relation_ids, "
                "and chunk_ids present in the context above. Remove any unsupported claims."
            )
            try:
                repaired_out, _ = self.harness.run_structured(
                    repair_prompt,
                    GraphCoTOutput,
                    step_name="graph_cot_grounding_repair",
                    system_prompt=COT_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=1536,
                )
                repaired_steps, repaired_audit = audit_cot_grounding(
                    repaired_out.steps,
                    entities=entities,
                    relations=relations,
                    evidences=evidences,
                    min_grounding_score=self.min_grounding_score,
                )
                repaired_audit.repair_triggered = True
                # Keep repaired output if grounding score improved or ungrounded count dropped
                if (
                    repaired_audit.grounding_score > audit.grounding_score
                    or len(repaired_audit.ungrounded_citations) < len(audit.ungrounded_citations)
                ):
                    cot_out = repaired_out
                    audited_steps = repaired_steps
                    audit = repaired_audit
                else:
                    audit.repair_triggered = True
            except ValueError:
                audit.repair_triggered = True

        return ReasoningResult(
            question=question,
            paradigm="cot",
            anchor_entities=cot_out.anchor_entities,
            final_answer=cot_out.final_answer,
            confidence=cot_out.confidence,
            steps=audited_steps,
            grounding_audit=audit,
            compaction=compaction_meta,
            trace_ledger=self.harness.ledger.summary(),
        )

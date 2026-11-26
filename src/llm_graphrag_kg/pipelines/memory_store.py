"""Verified Self-Evolving Episodic Memory Store (Stage 3.4, A-MEM + IndustryAssetEQA).

Implements a strict 4-stage anti-poisoning write-back gate before persisting any reasoning
conclusion back into long-term episodic memory or promoting it to a HyperEdge:
  1. Grounding Gate: `grounding_score >= min_grounding` (default 1.0), 0 ungrounded citations, >=1 chunk
  2. Confidence & Substance Gate: `confidence >= min_confidence`, non-refusal, substantive length
  3. Contradiction / Conflict NLI Gate: blocks polarity inversions or conflicting conclusions
  4. Auditable Provenance Persist: UTC ISO `valid_from` + `provenance_run_id` -> `verified_memories.jsonl`
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from llm_graphrag_kg.core.interfaces import BaseGraphStore
from llm_graphrag_kg.core.schema import VerifiedMemoryRecord
from llm_graphrag_kg.eval.llm_judge import detect_refusal
from llm_graphrag_kg.pipelines.hypergraph import create_hyperedge

_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|neither|unrelated|cannot|does\s+not|do\s+not|is\s+not|are\s+not)\b"
    r"|不是|并非|无关|不可能|排除|不会导致|未引起",
    flags=re.IGNORECASE,
)


def _content_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", text.lower()))


class WriteBackGateDecision(BaseModel):
    """Structured audit verdict from the 4-stage anti-poisoning write-back gate."""

    accepted: bool
    gate_failed: str = ""
    reason: str
    record: VerifiedMemoryRecord | None = None
    promoted_hyperedge_id: str | None = None


def check_memory_conflict(
    *,
    conclusion: str,
    condition_entity_ids: list[str],
    target_entity_ids: list[str],
    existing_records: list[VerifiedMemoryRecord],
) -> tuple[bool, str]:
    """Deterministic NLI-lite contradiction & poisoning detector against existing verified memories.

    Returns `(has_conflict, reason)`.
    """
    cand_neg = bool(_NEGATION_RE.search(conclusion))
    cand_conds = set(condition_entity_ids)
    cand_targets = set(target_entity_ids)
    cand_toks = _content_tokens(conclusion)

    for rec in existing_records:
        if rec.status != "verified":
            continue
        rec_conds = set(rec.condition_entity_ids)
        rec_targets = set(rec.target_entity_ids)
        rec_neg = bool(_NEGATION_RE.search(rec.conclusion))
        rec_toks = _content_tokens(rec.conclusion)

        # Case 1: Same target + overlapping conditions (or high lexical overlap), opposite polarity!
        shared_targets = cand_targets & rec_targets
        cond_overlap = (
            len(cand_conds & rec_conds) / max(1, len(cand_conds | rec_conds))
            if (cand_conds and rec_conds)
            else 0.0
        )
        tok_overlap = (
            len(cand_toks & rec_toks) / max(1, min(len(cand_toks), len(rec_toks)))
            if (cand_toks and rec_toks)
            else 0.0
        )

        if cand_neg != rec_neg and (
            (shared_targets and (cond_overlap >= 0.5 or tok_overlap >= 0.45))
            or tok_overlap >= 0.65
        ):
            return (
                True,
                f"Polarity contradiction with verified memory `{rec.memory_id}` "
                f"(valid_from={rec.valid_from}).",
            )

        # Case 2: Identical multi-condition signature (>=2 conditions) claiming a disjoint exclusive target
        if (
            len(cand_conds) >= 2
            and cand_conds == rec_conds
            and cand_targets
            and rec_targets
            and not shared_targets
            and not cand_neg
            and not rec_neg
        ):
            return (
                True,
                f"Competing root-cause conflict on identical conditions with `{rec.memory_id}` "
                f"(existing targets={sorted(rec_targets)} vs candidate={sorted(cand_targets)}).",
            )

    return False, ""


class VerifiedMemoryStore:
    """Append-only auditable store for verified episodic reasoning memories (`verified_memories.jsonl`)."""

    FILENAME = "verified_memories.jsonl"

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir
        self._records: list[VerifiedMemoryRecord] = []

    @property
    def file_path(self) -> Path | None:
        return (self.storage_dir / self.FILENAME) if self.storage_dir is not None else None

    def load(self, storage_dir: Path | None = None) -> list[VerifiedMemoryRecord]:
        if storage_dir is not None:
            self.storage_dir = storage_dir
        self._records = []
        fp = self.file_path
        if fp is not None and fp.is_file():
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._records.append(VerifiedMemoryRecord.model_validate_json(line))
        return list(self._records)

    def get_all(self) -> list[VerifiedMemoryRecord]:
        return list(self._records)

    def evaluate_and_write_back(
        self,
        *,
        question: str,
        answer: str,
        grounding_score: float | None,
        ungrounded_citations: list[str] | None = None,
        confidence: float = 0.0,
        condition_entity_ids: list[str] | None = None,
        target_entity_ids: list[str] | None = None,
        source_chunk_ids: list[str] | None = None,
        reasoning_paradigm: str = "auto",
        min_grounding: float = 1.0,
        min_confidence: float = 0.75,
        enable_conflict_check: bool = True,
        promote_to_hyperedge: bool = True,
        graph_store: BaseGraphStore | None = None,
        provenance_run_id: str | None = None,
    ) -> WriteBackGateDecision:
        """Run all 4 anti-poisoning gates and persist ONLY verified, non-conflicting memories."""
        ungrounded = list(ungrounded_citations or [])
        cids = list(dict.fromkeys(source_chunk_ids or []))
        cond_ids = list(dict.fromkeys(condition_entity_ids or []))
        tgt_ids = list(dict.fromkeys(target_entity_ids or []))

        # Gate 1: Strict Grounding & Provenance Gate
        if grounding_score is None:
            return WriteBackGateDecision(
                accepted=False,
                gate_failed="GATE_1_GROUNDING",
                reason="Missing GroundingAudit score (write-back requires explicit grounded reasoning).",
            )
        if grounding_score < min_grounding or ungrounded:
            return WriteBackGateDecision(
                accepted=False,
                gate_failed="GATE_1_GROUNDING",
                reason=(
                    f"GroundingAudit below threshold ({grounding_score:.2f} < {min_grounding:.2f}) "
                    f"or ungrounded citations present ({len(ungrounded)})."
                ),
            )
        if not cids:
            return WriteBackGateDecision(
                accepted=False,
                gate_failed="GATE_1_GROUNDING",
                reason="No verifiable source_chunk_ids attached; ghost memory write-back blocked.",
            )

        # Gate 2: Confidence & Non-Refusal Substance Gate
        clean_ans = answer.strip()
        if len(clean_ans) < 20 or detect_refusal(clean_ans):
            return WriteBackGateDecision(
                accepted=False,
                gate_failed="GATE_2_SUBSTANCE",
                reason="Refusal, abstention, or too-short answer rejected from long-term memory.",
            )
        if confidence < min_confidence:
            return WriteBackGateDecision(
                accepted=False,
                gate_failed="GATE_2_CONFIDENCE",
                reason=f"Reasoning confidence ({confidence:.2f}) < threshold ({min_confidence:.2f}).",
            )

        # Gate 3: Contradiction / Conflict NLI Gate
        if enable_conflict_check:
            has_conflict, conflict_reason = check_memory_conflict(
                conclusion=clean_ans,
                condition_entity_ids=cond_ids,
                target_entity_ids=tgt_ids,
                existing_records=self._records,
            )
            if has_conflict:
                return WriteBackGateDecision(
                    accepted=False,
                    gate_failed="GATE_3_CONFLICT_NLI",
                    reason=conflict_reason,
                )

        # Deduplicate exact question+conclusion signature
        sig_hash = hashlib.sha1(
            f"{question.strip().lower()}|{clean_ans[:240].lower()}".encode()
        ).hexdigest()[:12]
        mem_id = f"mem_{sig_hash}"
        for existing in self._records:
            if existing.memory_id == mem_id:
                return WriteBackGateDecision(
                    accepted=True,
                    reason=f"Verified memory `{mem_id}` already recorded (idempotent reuse).",
                    record=existing,
                )

        # Gate 4: Auditable Provenance Write-Back (+ optional HyperEdge promotion)
        now_iso = datetime.now(UTC).replace(microsecond=0).isoformat()
        run_id = provenance_run_id or f"run_{uuid.uuid4().hex[:10]}"
        record = VerifiedMemoryRecord(
            memory_id=mem_id,
            question=question.strip()[:400],
            conclusion=clean_ans[:800],
            condition_entity_ids=cond_ids,
            target_entity_ids=tgt_ids,
            source_chunk_ids=cids[:8],
            grounding_score=round(float(grounding_score), 4),
            confidence=round(float(confidence), 4),
            reasoning_paradigm=reasoning_paradigm,
            valid_from=now_iso,
            provenance_run_id=run_id,
            conflict_checked=enable_conflict_check,
            status="verified",
        )
        self._records.append(record)
        fp = self.file_path
        if fp is not None:
            fp.parent.mkdir(parents=True, exist_ok=True)
            with fp.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")

        promoted_hid: str | None = None
        if (
            promote_to_hyperedge
            and graph_store is not None
            and len(cond_ids) >= 2
            and tgt_ids
            and hasattr(graph_store, "upsert_hyperedge")
        ):
            he = create_hyperedge(
                cond_ids[:5],
                tgt_ids[0],
                relation_type="VERIFIED_JOINT_DIAGNOSIS",
                description=f"[Verified Memory {mem_id}] {clean_ans[:200]}",
                confidence=record.confidence,
                weight=1.0,
                source_chunk_ids=cids[:6],
                metadata={
                    "from_verified_memory": mem_id,
                    "valid_from": now_iso,
                    "provenance_run_id": run_id,
                },
            )
            graph_store.upsert_hyperedge(he)
            promoted_hid = he.hyperedge_id
            if self.storage_dir is not None:
                graph_store.save(self.storage_dir)

        return WriteBackGateDecision(
            accepted=True,
            reason=(
                f"Verified memory `{mem_id}` persisted (grounding={record.grounding_score:.2f}, "
                f"conf={record.confidence:.2f}, valid_from={now_iso})."
            ),
            record=record,
            promoted_hyperedge_id=promoted_hid,
        )

    def retrieve_relevant(
        self,
        question: str,
        seed_entity_ids: list[str],
        *,
        top_k: int = 3,
    ) -> list[VerifiedMemoryRecord]:
        """Recall verified historical episodic memories relevant to the active query."""
        if not self._records:
            return []
        q_toks = _content_tokens(question)
        seed_set = set(seed_entity_ids)
        scored: list[tuple[float, VerifiedMemoryRecord]] = []
        for rec in self._records:
            if rec.status != "verified":
                continue
            ent_pool = set(rec.condition_entity_ids) | set(rec.target_entity_ids)
            ent_hits = len(seed_set & ent_pool)
            rec_toks = _content_tokens(f"{rec.question} {rec.conclusion}")
            tok_hits = len(q_toks & rec_toks) / max(1, len(q_toks)) if q_toks else 0.0
            score = ent_hits * 0.65 + tok_hits * 1.0
            if score >= 0.35:
                scored.append((score, rec))
        scored.sort(key=lambda item: (-item[0], item[1].valid_from))
        return [rec for _, rec in scored[:top_k]]


def format_recalled_memories_context(records: list[VerifiedMemoryRecord]) -> str:
    """Format recalled verified memories as a traceable context block."""
    if not records:
        return ""
    lines = ["Recalled Verified Episodic Memories (A-MEM, Audited):"]
    for r in records:
        cids = ", ".join(r.source_chunk_ids[:3])
        lines.append(
            f"- [{r.memory_id} | valid_from={r.valid_from} | grounding={r.grounding_score:.2f} | "
            f"chunks=[{cids}]]: {r.conclusion[:260]}"
        )
    return "\n".join(lines)

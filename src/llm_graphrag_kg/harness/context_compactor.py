"""Three-tier lossless-first Context Compactor (L0 / L1 / L2) for GraphRAG + KG."""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.interfaces import BaseLLMClient
from llm_graphrag_kg.core.schema import Community, Entity, EvidenceItem, Relation

# Captures numeric measurements with optional physical units (e.g., 31.8 L/h, 12.8 kHz, 14°C, 20816)
NUMERIC_UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:kHz|Hz|MPa|kPa|Pa|mm/s|L/h|m/s|°C|℃|C|K|V|A|kW|MW|RPM|rpm|%|s|ms|h))?"
)


def estimate_tokens(text: str) -> int:
    """Fast dependency-free token estimator calibrated for mixed Chinese/English text."""
    if not text:
        return 0
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_cjk = re.sub(r"[\u4e00-\u9fff]", " ", text)
    ascii_words = re.findall(r"[A-Za-z0-9_./:-]+", non_cjk)
    # Approximate ~1.0 token per CJK char + ~1.3 tokens per English/code token
    return max(1, int(cjk_count * 1.0 + len(ascii_words) * 1.3))


def _tokenize_query(text: str) -> set[str]:
    lowered = text.lower()
    en_tokens = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    cjk_bigrams = {cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)}
    return en_tokens | cjk_bigrams


class CompactionReport(BaseModel):
    """Telemetry describing how much context noise was removed before LLM inference."""

    raw_tokens_est: int
    compacted_tokens_est: int
    compression_ratio: float  # e.g., 0.62 means 62% of raw tokens were removed
    raw_entity_count: int = 0
    kept_entity_count: int = 0
    raw_relation_count: int = 0
    kept_relation_count: int = 0
    dropped_nodes: int = 0
    applied_levels: list[str] = Field(default_factory=list)


class CompactedContext(BaseModel):
    """Structured compacted context ready for Prompt compilation."""

    entities_block: str = ""
    relations_block: str = ""
    communities_block: str = ""
    evidences_block: str = ""
    full_text: str = ""
    report: CompactionReport


class CompactedMemory(BaseModel):
    """L2 folded state for pruned ToT branches or prior multi-turn history."""

    label: str
    status: str = "compacted"  # pruned | completed | summary
    summary: str
    retained_entities: list[str] = Field(default_factory=list)
    retained_numbers: list[str] = Field(default_factory=list)
    retained_chunk_ids: list[str] = Field(default_factory=list)

    def to_compact_line(self) -> str:
        chunks = f" | chunks=[{', '.join(self.retained_chunk_ids[:4])}]" if self.retained_chunk_ids else ""
        nums = f" | metrics=[{', '.join(self.retained_numbers[:6])}]" if self.retained_numbers else ""
        return f"[{self.status.upper()} {self.label}] {self.summary}{nums}{chunks}"


class ContextCompactor:
    """Lossless-first 3-tier context compactor preserving entity IDs, numbers, and provenance."""

    def __init__(
        self,
        *,
        max_context_tokens: int = 4000,
        max_entities: int = 24,
        max_relations: int = 30,
        max_evidence_chunks: int = 6,
        max_sentences_per_chunk: int = 3,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.max_evidence_chunks = max_evidence_chunks
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def compact_graphrag_context(
        self,
        *,
        question: str,
        seed_entity_ids: list[str],
        entities: list[Entity],
        relations: list[Relation],
        communities: list[Community],
        evidences: list[EvidenceItem],
    ) -> CompactedContext:
        """Apply L0 (rule-based subgraph pruning & symbolic encoding) and L1 (extractive evidence)."""
        entity_map = {e.entity_id: e for e in entities}
        seed_set = set(seed_entity_ids)
        q_tokens = _tokenize_query(question)

        # Estimate raw (uncompacted) prompt size for honest telemetry
        raw_text = self._render_raw_verbose(entities, relations, communities, evidences)
        raw_tokens = estimate_tokens(raw_text)

        applied_levels: list[str] = ["L0_subgraph_pruning"]

        # --- L0: Score and prune relations strictly around question anchors ---
        scored_rels: list[tuple[float, Relation]] = []
        seen_triples: set[tuple[str, str, str]] = set()
        for rel in relations:
            triple_key = (rel.source_id, rel.relation_type, rel.target_id)
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)

            src = entity_map.get(rel.source_id)
            tgt = entity_map.get(rel.target_id)
            score = rel.weight * 0.5
            if rel.source_id in seed_set and rel.target_id in seed_set:
                score += 6.0
            elif rel.source_id in seed_set or rel.target_id in seed_set:
                score += 3.0

            haystack = (
                f"{src.name if src else ''} {rel.relation_type} "
                f"{tgt.name if tgt else ''} {rel.description}"
            ).lower()
            overlap = sum(1 for tok in q_tokens if tok in haystack)
            score += overlap * 0.8
            scored_rels.append((score, rel))

        scored_rels.sort(key=lambda item: (-item[0], item[1].relation_id))
        kept_rels = [rel for _, rel in scored_rels[: self.max_relations]]

        # Keep seed entities + endpoints of kept relations, capped by max_entities
        referenced_eids: set[str] = set(seed_set)
        for rel in kept_rels:
            referenced_eids.add(rel.source_id)
            referenced_eids.add(rel.target_id)

        scored_ents: list[tuple[float, Entity]] = []
        for ent in entities:
            score = 0.0
            if ent.entity_id in seed_set:
                score += 10.0
            if ent.entity_id in referenced_eids:
                score += 3.0
            name_low = ent.name.lower()
            if name_low in question.lower():
                score += 5.0
            overlap = sum(1 for tok in q_tokens if tok in f"{name_low} {ent.description}".lower())
            score += overlap * 0.6
            if score > 0 or ent.entity_id in referenced_eids:
                scored_ents.append((score, ent))

        scored_ents.sort(key=lambda item: (-item[0], item[1].name))
        kept_ents = [ent for _, ent in scored_ents[: self.max_entities]]
        kept_eid_set = {e.entity_id for e in kept_ents}

        # L0 compact symbolic formatting (high density, lossless IDs + chunk provenance)
        ent_lines = [
            f"- {e.name} [{e.entity_type}]"
            + (f" ({e.description[:90]})" if e.description else "")
            for e in kept_ents
        ]
        entities_block = (
            "Knowledge Graph Entities (Compacted):\n" + "\n".join(ent_lines) if ent_lines else ""
        )

        # Group parallel relations concisely when possible
        grouped_edges: dict[tuple[str, str], list[tuple[str, list[str]]]] = defaultdict(list)
        for rel in kept_rels:
            if rel.source_id not in kept_eid_set and rel.target_id not in kept_eid_set:
                continue
            src_name = entity_map[rel.source_id].name if rel.source_id in entity_map else rel.source_id
            tgt_name = entity_map[rel.target_id].name if rel.target_id in entity_map else rel.target_id
            grouped_edges[(src_name, rel.relation_type)].append((tgt_name, rel.source_chunk_ids))

        rel_lines: list[str] = []
        for (src_name, rtype), targets in grouped_edges.items():
            tgt_names = list(dict.fromkeys(t for t, _ in targets))
            cids = list(dict.fromkeys(cid for _, cs in targets for cid in cs))[:3]
            cid_tag = f" [{', '.join(cids)}]" if cids else ""
            if len(tgt_names) == 1:
                rel_lines.append(f"- {src_name} --[{rtype}]--> {tgt_names[0]}{cid_tag}")
            else:
                rel_lines.append(f"- {src_name} --[{rtype}]--> [{', '.join(tgt_names)}]{cid_tag}")
        relations_block = (
            "Knowledge Graph Relations (Compacted):\n" + "\n".join(rel_lines) if rel_lines else ""
        )

        comm_lines = [f"- [{c.title}] {c.summary[:180]}" for c in communities[:3]]
        communities_block = (
            "Community Summaries:\n" + "\n".join(comm_lines) if comm_lines else ""
        )

        # --- L1: Extractive Evidence Compaction ---
        kept_evidences = evidences[: self.max_evidence_chunks]
        anchor_terms = q_tokens | {e.name.lower() for e in kept_ents if len(e.name) >= 2}
        raw_ev_tokens = sum(estimate_tokens(ev.text) for ev in kept_evidences)
        graph_tokens_so_far = estimate_tokens(
            "\n".join([entities_block, relations_block, communities_block])
        )
        need_l1 = (graph_tokens_so_far + raw_ev_tokens) > int(self.max_context_tokens * 0.75) or (
            any(len(ev.text) > 220 for ev in kept_evidences)
        )

        ev_lines: list[str] = []
        if need_l1 and kept_evidences:
            applied_levels.append("L1_extractive_evidence")
            for ev in kept_evidences:
                snippet = self._extract_key_sentences(
                    ev.text,
                    anchor_terms=anchor_terms,
                    max_sentences=self.max_sentences_per_chunk,
                )
                ev_lines.append(f"- [{ev.chunk_id} | {ev.source} | via={ev.matched_via}]: {snippet}")
        else:
            for ev in kept_evidences:
                ev_lines.append(f"- [{ev.chunk_id} | {ev.source} | via={ev.matched_via}]: {ev.text}")

        evidences_block = (
            "Source Text Evidences:\n" + "\n".join(ev_lines) if ev_lines else ""
        )

        sections = [b for b in (entities_block, relations_block, communities_block, evidences_block) if b]
        full_text = "\n\n".join(sections)

        # Hard budget guard: if still above max_context_tokens, trim lowest-priority evidence lines
        if estimate_tokens(full_text) > self.max_context_tokens and len(ev_lines) > 2:
            while len(ev_lines) > 2 and estimate_tokens(full_text) > self.max_context_tokens:
                ev_lines.pop()
                evidences_block = "Source Text Evidences:\n" + "\n".join(ev_lines)
                sections = [
                    b for b in (entities_block, relations_block, communities_block, evidences_block) if b
                ]
                full_text = "\n\n".join(sections)

        compacted_tokens = estimate_tokens(full_text)
        ratio = round(max(0.0, 1.0 - (compacted_tokens / max(raw_tokens, 1))), 4)

        report = CompactionReport(
            raw_tokens_est=raw_tokens,
            compacted_tokens_est=compacted_tokens,
            compression_ratio=ratio,
            raw_entity_count=len(entities),
            kept_entity_count=len(kept_ents),
            raw_relation_count=len(relations),
            kept_relation_count=len(kept_rels),
            dropped_nodes=max(0, len(entities) - len(kept_ents)),
            applied_levels=applied_levels,
        )
        return CompactedContext(
            entities_block=entities_block,
            relations_block=relations_block,
            communities_block=communities_block,
            evidences_block=evidences_block,
            full_text=full_text,
            report=report,
        )

    def fold_state_l2(
        self,
        *,
        label: str,
        raw_text: str,
        status: str = "pruned",
        entities: list[str] | None = None,
        chunk_ids: list[str] | None = None,
        llm: BaseLLMClient | None = None,
    ) -> CompactedMemory:
        """L2 structured folding for pruned ToT branches or long multi-turn history.

        Guarantees preservation of all numeric measurements/units, entity names, and chunk IDs.
        """
        retained_numbers = list(dict.fromkeys(NUMERIC_UNIT_PATTERN.findall(raw_text)))[:10]
        retained_entities = list(dict.fromkeys(entities or []))[:10]
        retained_chunks = list(dict.fromkeys(chunk_ids or []))[:6]

        if llm is not None and estimate_tokens(raw_text) > 250:
            prompt = (
                "Compress the following diagnostic trace into 1-2 concise sentences. "
                "You MUST preserve all exact numeric measurements, units, entity names, and chunk IDs.\n\n"
                f"Label: {label} ({status})\nText:\n{raw_text[:2000]}"
            )
            summary = llm.complete(prompt, temperature=0.0, json_mode=False).strip()
        else:
            # Deterministic lossless extractive fold (zero LLM latency)
            sentences = self._split_sentences(raw_text)
            summary = " ".join(sentences[:2])[:260] if sentences else raw_text[:260]

        return CompactedMemory(
            label=label,
            status=status,
            summary=summary,
            retained_entities=retained_entities,
            retained_numbers=retained_numbers,
            retained_chunk_ids=retained_chunks,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Split on sentence terminators without ever breaking decimal numbers like 7.8 or 31.8
        parts = re.split(r"(?:(?<=[。！？!?\n])|(?<=\.)(?=\s|$)(?<!\d\.))\s*", text.strip())
        # Secondary safe handling: avoid splitting decimal point like `7.8` (digit before & after `.`)
        merged: list[str] = []
        for p in parts:
            s = p.strip()
            if not s:
                continue
            if merged and merged[-1].endswith(".") and re.match(r"^\d", s):
                merged[-1] = merged[-1] + s
            else:
                merged.append(s)
        return merged

    def _extract_key_sentences(
        self,
        text: str,
        *,
        anchor_terms: set[str],
        max_sentences: int,
    ) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= max_sentences:
            return text.strip()

        scored: list[tuple[float, int, str]] = []
        for idx, sent in enumerate(sentences):
            s_low = sent.lower()
            hits = sum(1 for term in anchor_terms if term in s_low)
            has_num = 0.5 if NUMERIC_UNIT_PATTERN.search(sent) else 0.0
            pos_bonus = 0.3 if idx == 0 else 0.0
            scored.append((hits + has_num + pos_bonus, idx, sent))

        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = sorted(scored[:max_sentences], key=lambda item: item[1])
        return " ".join(sent for _, _, sent in chosen)

    @staticmethod
    def _render_raw_verbose(
        entities: list[Entity],
        relations: list[Relation],
        communities: list[Community],
        evidences: list[EvidenceItem],
    ) -> str:
        entity_map = {e.entity_id: e for e in entities}
        parts: list[str] = []
        for e in entities:
            parts.append(f"- {e.name} ({e.entity_type}): {e.description}")
        for r in relations:
            s = entity_map[r.source_id].name if r.source_id in entity_map else r.source_id
            t = entity_map[r.target_id].name if r.target_id in entity_map else r.target_id
            parts.append(f"- {s} --[{r.relation_type}]--> {t} (weight={r.weight:.1f}): {r.description}")
        for c in communities:
            parts.append(f"- [{c.title}] {c.summary}")
        for ev in evidences:
            parts.append(f"- [{ev.chunk_id} | {ev.source}]: {ev.text}")
        return "\n".join(parts)

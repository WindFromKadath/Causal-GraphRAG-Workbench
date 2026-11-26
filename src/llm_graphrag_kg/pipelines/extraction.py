"""LLM-driven entity & relation extraction with deterministic normalization and deduplication."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.interfaces import BaseLLMClient
from llm_graphrag_kg.core.schema import Entity, Relation, TextChunk
from llm_graphrag_kg.harness import LLMHarness

EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert Knowledge Graph extraction engine. "
    "Extract concise, accurate entities and directed relations strictly grounded in the input text. "
    "Respond ONLY with valid JSON."
)


def make_entity_id(name: str) -> str:
    canonical = re.sub(r"\s+", " ", name.strip()).lower()
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"ent_{digest}"


def make_relation_id(source_id: str, relation_type: str, target_id: str) -> str:
    canonical = f"{source_id}|{relation_type.strip().upper()}|{target_id}"
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"rel_{digest}"


def parse_json_payload(raw_text: str) -> dict[str, Any]:
    """Robustly parse JSON returned by LLMs, stripping markdown code fences if present."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"entities": [], "relations": []}
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


class KnowledgeExtractor:
    """Extracts normalized `Entity` and `Relation` objects from text chunks."""

    def __init__(
        self,
        llm: BaseLLMClient,
        *,
        entity_types: list[str] | None = None,
        temperature: float = 0.1,
        max_entities_per_chunk: int = 15,
    ) -> None:
        self.llm = llm
        self.entity_types = entity_types or [
            "Concept",
            "Technology",
            "Method",
            "Component",
            "Organization",
            "Person",
            "Document",
            "Metric",
        ]
        self.temperature = temperature
        self.max_entities_per_chunk = max_entities_per_chunk

    def extract_from_chunk(self, chunk: TextChunk) -> tuple[list[Entity], list[Relation]]:
        prompt = self._build_prompt(chunk.text)
        raw = self.llm.complete(
            prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=True,
        )
        payload = parse_json_payload(raw)

        entities_by_id: dict[str, Entity] = {}
        name_to_id: dict[str, str] = {}

        for item in payload.get("entities", [])[: self.max_entities_per_chunk]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            etype = str(item.get("entity_type", "Concept")).strip() or "Concept"
            desc = str(item.get("description", "")).strip()
            eid = make_entity_id(name)
            name_to_id[name.lower()] = eid

            if eid not in entities_by_id:
                entities_by_id[eid] = Entity(
                    entity_id=eid,
                    name=name,
                    entity_type=etype,
                    description=desc,
                    source_chunk_ids=[chunk.chunk_id],
                )

        relations_by_id: dict[str, Relation] = {}
        for item in payload.get("relations", []):
            if not isinstance(item, dict):
                continue
            src_name = str(item.get("source", "")).strip()
            tgt_name = str(item.get("target", "")).strip()
            rel_type = str(item.get("relation_type", "RELATED_TO")).strip().upper() or "RELATED_TO"
            rel_type = re.sub(r"\s+", "_", rel_type)
            desc = str(item.get("description", "")).strip()
            if not src_name or not tgt_name or src_name.lower() == tgt_name.lower():
                continue

            src_id = name_to_id.get(src_name.lower()) or make_entity_id(src_name)
            tgt_id = name_to_id.get(tgt_name.lower()) or make_entity_id(tgt_name)

            # Ensure endpoint entities exist even if the LLM omitted one from `entities`
            for endpoint_id, endpoint_name in ((src_id, src_name), (tgt_id, tgt_name)):
                if endpoint_id not in entities_by_id:
                    entities_by_id[endpoint_id] = Entity(
                        entity_id=endpoint_id,
                        name=endpoint_name,
                        entity_type="Concept",
                        description="",
                        source_chunk_ids=[chunk.chunk_id],
                    )

            rel_id = make_relation_id(src_id, rel_type, tgt_id)
            if rel_id not in relations_by_id:
                relations_by_id[rel_id] = Relation(
                    relation_id=rel_id,
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=rel_type,
                    description=desc,
                    weight=1.0,
                    source_chunk_ids=[chunk.chunk_id],
                )

        return list(entities_by_id.values()), list(relations_by_id.values())

    def _build_prompt(self, text: str) -> str:
        allowed_types = ", ".join(self.entity_types)
        return (
            "ENTITY_EXTRACTION_TASK\n"
            f"Allowed entity_type values: [{allowed_types}].\n"
            f"Extract up to {self.max_entities_per_chunk} important entities and directed relations "
            "from the text below.\n"
            "Return strictly a JSON object with this exact schema:\n"
            "{\n"
            '  "entities": [{"name": "...", "entity_type": "...", "description": "..."}],\n'
            '  "relations": [{"source": "...", "target": "...", "relation_type": "UPPER_SNAKE", '
            '"description": "..."}]\n'
            "}\n\n"
            "TEXT_START\n"
            f"{text}\n"
            "TEXT_END"
        )


_GENERIC_REL_REPLACEMENTS = {
    "CO_OCCURS_WITH": "ASSOCIATED_WITH",
    "COOCCURS_WITH": "ASSOCIATED_WITH",
    "RELATED": "ASSOCIATED_WITH",
}


class OpenIEEntityItem(BaseModel):
    name: str
    entity_type: str = "Concept"
    description: str = ""


class OpenIERelationItem(BaseModel):
    source: str
    target: str
    predicate: str
    description: str = ""


class OpenIEChunkOutput(BaseModel):
    entities: list[OpenIEEntityItem] = Field(default_factory=list)
    relations: list[OpenIERelationItem] = Field(default_factory=list)


class OpenIEStats(BaseModel):
    total_chunks_processed: int = 0
    resumed_from_checkpoint: int = 0
    extracted_entities: int = 0
    extracted_relations: int = 0
    semantic_predicate_ratio: float = 1.0
    top_predicates: dict[str, int] = Field(default_factory=dict)


def normalize_openie_predicate(raw_pred: str) -> str:
    """Normalize a free-form predicate into canonical UPPER_SNAKE_CASE."""
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", raw_pred.strip()).strip("_").upper()
    if not cleaned:
        return "ASSOCIATED_WITH"
    return _GENERIC_REL_REPLACEMENTS.get(cleaned, cleaned)


class SemanticOpenIEExtractor:
    """Harness-backed directional Semantic OpenIE triple extractor with JSONL checkpointing."""

    def __init__(
        self,
        llm: BaseLLMClient,
        *,
        entity_types: list[str] | None = None,
        domain: str = "general",
        temperature: float = 0.05,
        max_entities_per_chunk: int = 12,
        max_relations_per_chunk: int = 14,
    ) -> None:
        self.llm = llm
        self.harness = LLMHarness(
            llm, json_repair_retries=1, max_calls=5000, max_wall_seconds=3600.0
        )
        self.domain = domain
        self.entity_types = entity_types or [
            "Concept",
            "DiseaseConcept",
            "Treatment",
            "DiagnosticTest",
            "RiskFactor",
            "Component",
            "Equipment",
            "FaultMode",
            "Person",
            "Organization",
            "Location",
            "Event",
        ]
        self.temperature = temperature
        self.max_entities_per_chunk = max_entities_per_chunk
        self.max_relations_per_chunk = max_relations_per_chunk

    def extract_chunk(self, chunk: TextChunk) -> tuple[list[Entity], list[Relation]]:
        """Extract typed entities and directional semantic triples from a single chunk."""
        allowed = ", ".join(self.entity_types)
        prompt = (
            "SEMANTIC_OPENIE_TASK\n"
            f"Domain: {self.domain}\n"
            f"Allowed entity_type values: [{allowed}].\n"
            f"Extract up to {self.max_entities_per_chunk} canonical entities and up to "
            f"{self.max_relations_per_chunk} DIRECTIONAL semantic triples from the passage.\n"
            "Rules:\n"
            "- Prefer specific directional UPPER_SNAKE predicates such as TREATS, CAUSES, "
            "DIAGNOSES, INDICATES, PREVENTS, PART_OF, CONTRAINDICATED_FOR, LOCATED_IN, "
            "CHARACTERIZED_BY, INTERACTS_WITH, CAUSED_BY, PERFORMED_FOR.\n"
            "- NEVER output generic CO_OCCURS_WITH.\n"
            "- Return strict JSON: "
            '{"entities": [{"name": "...", "entity_type": "...", "description": "..."}], '
            '"relations": [{"source": "...", "target": "...", "predicate": "TREATS", '
            '"description": "..."}]}\n\n'
            f"TEXT_START\n{chunk.text[:2400]}\nTEXT_END"
        )
        try:
            parsed, _ = self.harness.run_structured(
                prompt,
                OpenIEChunkOutput,
                step_name="semantic_openie",
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=1024,
            )
        except ValueError:
            return [], []

        entities_by_id: dict[str, Entity] = {}
        name_to_id: dict[str, str] = {}
        for item in parsed.entities[: self.max_entities_per_chunk]:
            name = re.sub(r"\s+", " ", item.name.strip())
            if len(name) < 2:
                continue
            eid = make_entity_id(name)
            name_to_id[name.lower()] = eid
            if eid not in entities_by_id:
                entities_by_id[eid] = Entity(
                    entity_id=eid,
                    name=name,
                    entity_type=item.entity_type.strip() or "Concept",
                    description=item.description.strip(),
                    source_chunk_ids=[chunk.chunk_id],
                )

        relations_by_id: dict[str, Relation] = {}
        for rel in parsed.relations[: self.max_relations_per_chunk]:
            src_name = re.sub(r"\s+", " ", rel.source.strip())
            tgt_name = re.sub(r"\s+", " ", rel.target.strip())
            if len(src_name) < 2 or len(tgt_name) < 2 or src_name.lower() == tgt_name.lower():
                continue
            pred = normalize_openie_predicate(rel.predicate)
            src_id = name_to_id.get(src_name.lower()) or make_entity_id(src_name)
            tgt_id = name_to_id.get(tgt_name.lower()) or make_entity_id(tgt_name)
            if src_id == tgt_id:
                continue

            for endpoint_id, endpoint_name in ((src_id, src_name), (tgt_id, tgt_name)):
                if endpoint_id not in entities_by_id:
                    entities_by_id[endpoint_id] = Entity(
                        entity_id=endpoint_id,
                        name=endpoint_name,
                        entity_type=self.entity_types[0] if self.entity_types else "Concept",
                        description="",
                        source_chunk_ids=[chunk.chunk_id],
                    )

            rid = make_relation_id(src_id, pred, tgt_id)
            if rid not in relations_by_id:
                relations_by_id[rid] = Relation(
                    relation_id=rid,
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=pred,
                    description=rel.description.strip()
                    or f"{src_name} --[{pred}]--> {tgt_name} (chunk {chunk.chunk_id})",
                    weight=1.0,
                    source_chunk_ids=[chunk.chunk_id],
                )

        return list(entities_by_id.values()), list(relations_by_id.values())

    def extract_chunks_checkpointed(
        self,
        chunks: list[TextChunk],
        *,
        checkpoint_path: Path | None = None,
        resume: bool = True,
        max_chunks: int = 0,
        progress_every: int = 10,
    ) -> tuple[list[Entity], list[Relation], OpenIEStats]:
        """Extract semantic triples across chunks with resumable JSONL checkpointing."""
        target_chunks = chunks[:max_chunks] if max_chunks > 0 else chunks
        cached: dict[str, tuple[list[Entity], list[Relation]]] = {}
        if checkpoint_path is not None and resume and checkpoint_path.is_file():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = str(rec.get("chunk_id", ""))
                ents = [Entity.model_validate(e) for e in rec.get("entities", [])]
                rels = [Relation.model_validate(r) for r in rec.get("relations", [])]
                cached[cid] = (ents, rels)

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        merged_ents: dict[str, Entity] = {}
        merged_rels: dict[str, Relation] = {}
        resumed_count = 0

        ckpt_fh = (
            checkpoint_path.open("a", encoding="utf-8") if checkpoint_path is not None else None
        )
        try:
            for idx, ch in enumerate(target_chunks, start=1):
                if ch.chunk_id in cached:
                    ents, rels = cached[ch.chunk_id]
                    resumed_count += 1
                else:
                    ents, rels = self.extract_chunk(ch)
                    if ckpt_fh is not None:
                        ckpt_fh.write(
                            json.dumps(
                                {
                                    "chunk_id": ch.chunk_id,
                                    "entities": [e.model_dump() for e in ents],
                                    "relations": [r.model_dump() for r in rels],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        ckpt_fh.flush()

                for e in ents:
                    if e.entity_id not in merged_ents:
                        merged_ents[e.entity_id] = e
                    else:
                        ex = merged_ents[e.entity_id]
                        for sc in e.source_chunk_ids:
                            if sc not in ex.source_chunk_ids:
                                ex.source_chunk_ids.append(sc)
                        if not ex.description and e.description:
                            ex.description = e.description

                for r in rels:
                    if r.relation_id not in merged_rels:
                        merged_rels[r.relation_id] = r
                    else:
                        rx = merged_rels[r.relation_id]
                        rx.weight += r.weight
                        for sc in r.source_chunk_ids:
                            if sc not in rx.source_chunk_ids:
                                rx.source_chunk_ids.append(sc)

                if progress_every > 0 and (idx % progress_every == 0 or idx == len(target_chunks)):
                    print(
                        f"[SemanticOpenIE | {self.domain}] {idx}/{len(target_chunks)} chunks | "
                        f"entities={len(merged_ents)} | semantic_rels={len(merged_rels)}",
                        flush=True,
                    )
        finally:
            if ckpt_fh is not None:
                ckpt_fh.close()

        pred_counts: dict[str, int] = {}
        non_generic = 0
        for r in merged_rels.values():
            pred_counts[r.relation_type] = pred_counts.get(r.relation_type, 0) + 1
            if r.relation_type not in {"CO_OCCURS_WITH", "RELATED_TO"}:
                non_generic += 1

        total_rels = len(merged_rels)
        stats = OpenIEStats(
            total_chunks_processed=len(target_chunks),
            resumed_from_checkpoint=resumed_count,
            extracted_entities=len(merged_ents),
            extracted_relations=total_rels,
            semantic_predicate_ratio=round(non_generic / total_rels, 4) if total_rels else 1.0,
            top_predicates=dict(sorted(pred_counts.items(), key=lambda x: -x[1])[:12]),
        )
        return list(merged_ents.values()), list(merged_rels.values()), stats

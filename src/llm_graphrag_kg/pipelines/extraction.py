"""LLM-driven entity & relation extraction with deterministic normalization and deduplication."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from llm_graphrag_kg.core.interfaces import BaseLLMClient
from llm_graphrag_kg.core.schema import Entity, Relation, TextChunk

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

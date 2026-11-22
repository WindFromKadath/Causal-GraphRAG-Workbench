"""Adapter and 4-suite stress-test evaluation harness for IBM FailureSensorIQ (Apache-2.0).

Supports:
1. Ontology-First structured KG construction from `failuresensoriq.mapped.jsonl`
   (or directly from `failuresensoriq_standard__all.jsonl`):
       Equipment --[HAS_FAILURE_MODE]--> FailureMode (asset-scoped)
       FailureMode --[INSTANCE_OF]--> FailureCategory (cross-asset analogy)
       FailureMode --[MONITORED_BY]--> SignalChannel (normalized via Entity Resolution)
2. Metadata-free natural-language asset entity linking for perturbed & multi-answer suites.
3. High-throughput concise JSON MCQA evaluation with JSONL checkpoint/resume across all 4 suites:
   - L1: standard single-choice (`failuresensoriq_standard__all.jsonl`)
   - L2: 10-option high-distractor (`failuresensoriq_standard__all_10_options.jsonl`)
   - L3: 5-choose-2 multi-answer (`failuresensoriq_standard__all_multi_answers.jsonl`)
   - L4: perturbed option labels & no metadata (`failuresensoriq_perturbed__perturbed_simple.jsonl`)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient, BaseLLMClient
from llm_graphrag_kg.core.schema import Entity, GraphAuditStats, Relation, TextChunk
from llm_graphrag_kg.pipelines.community import CommunityDetector
from llm_graphrag_kg.pipelines.extraction import make_entity_id, make_relation_id
from llm_graphrag_kg.providers.factory import (
    create_embedding_client,
    create_graph_store,
    create_llm_client,
    create_vector_store,
)
from llm_graphrag_kg.viz.audit_report import compute_audit_stats

# Explicit, auditable Entity Resolution map fixing real typos/duplicates found in FailureSensorIQ.
SENSOR_CANONICAL_MAP: dict[str, str] = {
    "temparature": "temperature",
    "coast down": "coast down time",
    "compresor pressure/ pressure ratio": "compressor pressure/ pressure ratio",
    (
        "dielecric frequency response (dfr)/ polarization and de-polarization current (pdc)/ "
        "recovery voltage method (rvm)"
    ): (
        "dielectric frequency response (dfr)/ polarization and de-polarization current (pdc)/ "
        "recovery voltage method (rvm)"
    ),
}

DEFAULT_STRESS_SUITES: list[tuple[str, str]] = [
    ("L1_standard_all", "failuresensoriq_standard__all.jsonl"),
    ("L2_10_options", "failuresensoriq_standard__all_10_options.jsonl"),
    ("L3_multi_answers", "failuresensoriq_standard__all_multi_answers.jsonl"),
    ("L4_perturbed_simple", "failuresensoriq_perturbed__perturbed_simple.jsonl"),
]


def normalize_sensor_name(raw_name: str) -> tuple[str, bool]:
    """Normalize a FailureSensorIQ sensor label; returns (canonical_name, was_merged)."""
    cleaned = re.sub(r"\s+", " ", raw_name.strip()).lower()
    canonical = SENSOR_CANONICAL_MAP.get(cleaned, cleaned)
    return canonical, canonical != cleaned


def normalize_option_id(raw_id: str) -> str:
    """Normalize option identifiers such as `'A'`, `'P)'`, or `'b.'` to `'A'`, `'P'`, `'B'`."""
    return re.sub(r"[^A-Za-z0-9]", "", raw_id.strip()).upper()


def detect_asset_from_question(question: str, known_assets: list[str]) -> str:
    """Metadata-free asset entity linking by longest normalized substring match."""
    q_lower = re.sub(r"\s+", " ", question.strip()).lower()
    for asset in sorted(known_assets, key=len, reverse=True):
        a_norm = asset.strip().lower()
        if a_norm and a_norm in q_lower:
            return a_norm
    return ""


def parse_option_answers(
    raw_response: str,
    valid_ids: list[str],
    expected_count: int = 1,
) -> list[str]:
    """Extract 1..N normalized option IDs from JSON or free-text LLM output."""
    norm_valid = [normalize_option_id(v) for v in valid_ids]
    valid_set = set(norm_valid)
    text = raw_response.strip()
    if not text:
        return []

    found: list[str] = []

    def _collect(candidates: list[str]) -> None:
        for item in candidates:
            norm = normalize_option_id(str(item))
            if norm in valid_set and norm not in found:
                found.append(norm)

    # 1. Structured JSON extraction: {"answers": ["B", "E"]} or {"answer": "C"}
    cleaned = text
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        s_idx, e_idx = cleaned.find("{"), cleaned.rfind("}")
        if s_idx != -1 and e_idx > s_idx:
            cleaned = cleaned[s_idx : e_idx + 1]
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            ans = obj.get("answers", obj.get("answer"))
            if isinstance(ans, list):
                _collect([str(x) for x in ans])
            elif isinstance(ans, str):
                _collect(re.findall(r"[A-Za-z0-9]+", ans))
    except json.JSONDecodeError:
        pass

    # 2. Explicit "Answer(s): ..." fallback
    if len(found) < expected_count:
        m = re.search(
            r"(?:final\s+)?answers?\s*[:：]\s*([^\n]+)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            _collect(re.findall(r"[A-Za-z0-9]+", m.group(1)))

    # 3. Line-start / standalone valid token fallback
    if len(found) < expected_count:
        tokens = re.findall(r"\b([A-Za-z])\b", text)
        _collect(tokens)

    return found[:expected_count]


def compute_set_f1(predicted: list[str], gold: list[str]) -> float:
    """Compute F1 score between predicted and gold option sets."""
    p_set = set(predicted)
    g_set = set(gold)
    if not p_set and not g_set:
        return 1.0
    if not p_set or not g_set:
        return 0.0
    tp = len(p_set & g_set)
    return round((2.0 * tp) / (len(p_set) + len(g_set)), 4)


class IngestStats(BaseModel):
    """Statistics produced when ingesting FailureSensorIQ into the framework."""

    source_records: int
    gold_triples_loaded: int
    asset_count: int
    failure_mode_count: int
    failure_category_count: int
    sensor_count: int
    er_normalized_sensor_mentions: int
    audit: GraphAuditStats


class EvalQuestionOutcome(BaseModel):
    """Per-question evaluation outcome across one baseline mode."""

    question_id: int
    asset: str
    asset_linked: bool = True
    relevancy: str
    question_type: str
    gold_option: str
    predicted_option: str
    gold_options: list[str] = Field(default_factory=list)
    predicted_options: list[str] = Field(default_factory=list)
    is_correct: bool
    f1_score: float = 0.0
    latency_seconds: float = 0.0
    raw_response: str
    seed_entities: list[str] = Field(default_factory=list)
    expanded_nodes: int = 0


class ModeMetrics(BaseModel):
    """Aggregated accuracy, F1, empty-rate, and latency metrics for one evaluation mode."""

    mode: str
    total: int
    correct: int
    accuracy: float
    macro_f1: float = 0.0
    empty_responses: int = 0
    asset_linked_count: int = 0
    avg_latency_seconds: float = 0.0
    by_relevancy: dict[str, dict[str, float | int]] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    """Complete side-by-side benchmark report for a single dataset suite."""

    suite_name: str = "failuresensoriq"
    total_questions: int
    seed: int
    llm_provider: str
    llm_model: str
    concise_json: bool = True
    modes: list[ModeMetrics]
    details: dict[str, list[EvalQuestionOutcome]] = Field(default_factory=dict)


class StressSuiteReport(BaseModel):
    """Aggregated 4-suite stress-test report across all FailureSensorIQ difficulty tiers."""

    limit_per_suite: int
    seed: int
    llm_provider: str
    llm_model: str
    concise_json: bool
    suites: list[EvaluationSummary]


def _extract_triples_from_mapped(
    jsonl_path: Path,
) -> tuple[int, dict[str, dict[str, set[str]]], dict[str, set[str]], int]:
    """Parse `failuresensoriq.mapped.jsonl` -> asset -> failure_mode -> {canonical_sensors}."""
    mapping: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    sensor_aliases: dict[str, set[str]] = defaultdict(set)
    total_rows = 0
    er_fixes = 0

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)
            if not (row.get("gold") and row.get("relevant")):
                continue
            asset = re.sub(r"\s+", " ", str(row["asset"]).strip()).lower()
            if row.get("anchor_class") == "FailureMode":
                fm = re.sub(r"\s+", " ", str(row["anchor"]).strip()).lower()
                raw_sensor = str(row["option"]).strip()
            else:
                raw_sensor = str(row["anchor"]).strip()
                fm = re.sub(r"\s+", " ", str(row["option"]).strip()).lower()

            canonical_sensor, merged = normalize_sensor_name(raw_sensor)
            if merged:
                er_fixes += 1
                sensor_aliases[canonical_sensor].add(raw_sensor.strip().lower())
            mapping[asset][fm].add(canonical_sensor)

    return total_rows, mapping, sensor_aliases, er_fixes


def _extract_triples_from_standard_qa(
    jsonl_path: Path,
) -> tuple[int, dict[str, dict[str, set[str]]], dict[str, set[str]], int]:
    """Parse raw `failuresensoriq_standard__all.jsonl` directly if mapped file is unavailable."""
    mapping: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    sensor_aliases: dict[str, set[str]] = defaultdict(set)
    total_rows = 0
    er_fixes = 0

    fm_pattern = re.compile(
        r"(?:failure event|failure mode|has|when)\s+([a-z0-9()/\-\s]+?)\s+(?:occurs|in asset|should)",
        flags=re.IGNORECASE,
    )
    sensor_pattern = re.compile(
        r"sensor\s+([a-z0-9()/\-\s]+?)\s+(?:is|can|detects|monitors)",
        flags=re.IGNORECASE,
    )

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)
            if row.get("question_type") != "mcp1_positive":
                continue
            asset = re.sub(r"\s+", " ", str(row.get("asset_name", "")).strip()).lower()
            options: list[str] = list(row.get("options", []))
            correct_flags: list[bool] = list(row.get("correct", []))
            gold_options = [opt for opt, ok in zip(options, correct_flags, strict=False) if ok]
            if not asset or not gold_options:
                continue

            relevancy = str(row.get("relevancy", ""))
            question = str(row.get("question", ""))
            if relevancy == "relevant_sensors_for_failure_mode":
                m = fm_pattern.search(question)
                if not m:
                    continue
                fm = re.sub(r"\s+", " ", m.group(1).strip()).lower()
                for raw_s in gold_options:
                    canonical_s, merged = normalize_sensor_name(raw_s)
                    if merged:
                        er_fixes += 1
                        sensor_aliases[canonical_s].add(raw_s.strip().lower())
                    mapping[asset][fm].add(canonical_s)
            elif relevancy == "relevant_failure_modes_for_sensor":
                m = sensor_pattern.search(question)
                if not m:
                    continue
                raw_s = m.group(1).strip()
                canonical_s, merged = normalize_sensor_name(raw_s)
                if merged:
                    er_fixes += 1
                    sensor_aliases[canonical_s].add(raw_s.strip().lower())
                for fm_opt in gold_options:
                    fm = re.sub(r"\s+", " ", fm_opt.strip()).lower()
                    mapping[asset][fm].add(canonical_s)

    return total_rows, mapping, sensor_aliases, er_fixes


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_failuresensoriq_kg(
    source_path: Path,
    config: AppConfig,
    *,
    docs_dir: Path | None = None,
    storage_dir: Path | None = None,
    summarize_communities: bool = False,
    llm: BaseLLMClient | None = None,
    embedder: BaseEmbeddingClient | None = None,
) -> IngestStats:
    """Build an Ontology-First, ER-cleaned Knowledge Graph and vector index from FailureSensorIQ."""
    if not source_path.is_file():
        raise FileNotFoundError(f"FailureSensorIQ source file not found: {source_path}")

    first_line = ""
    with source_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                first_line = line.strip()
                break
    sample_obj = json.loads(first_line)
    if "anchor" in sample_obj and "option" in sample_obj:
        total_rows, mapping, sensor_aliases, er_fixes = _extract_triples_from_mapped(source_path)
    else:
        total_rows, mapping, sensor_aliases, er_fixes = _extract_triples_from_standard_qa(source_path)

    out_docs_dir = docs_dir or config.input_path
    out_storage_dir = storage_dir or config.storage_path
    out_docs_dir.mkdir(parents=True, exist_ok=True)
    out_storage_dir.mkdir(parents=True, exist_ok=True)

    graph_store = create_graph_store()
    vector_store = create_vector_store()
    active_embedder = embedder or create_embedding_client(config)
    active_llm = llm or (create_llm_client(config) if summarize_communities else None)

    chunks: list[TextChunk] = []
    manifest_docs: list[dict[str, str]] = []
    unique_categories: set[str] = set()
    unique_sensors: set[str] = set()
    gold_triple_count = 0

    for asset in sorted(mapping.keys()):
        fm_map = mapping[asset]
        slug = _slugify(asset)
        doc_file = out_docs_dir / f"{slug}.md"
        try:
            rel_source = doc_file.resolve().relative_to(config.project_root.resolve()).as_posix()
        except ValueError:
            rel_source = f"failuresensoriq/{slug}.md"

        doc_hash = hashlib.sha1(rel_source.encode("utf-8")).hexdigest()[:8]
        doc_id = f"fsiq_{doc_hash}"
        manifest_docs.append(
            {"doc_id": doc_id, "source": rel_source, "title": f"FailureSensorIQ Asset: {asset}"}
        )

        header = (
            f"# Industrial Asset: {asset}\n\n"
            f"ISO-aligned failure mode and sensor monitoring specification for `{asset}` "
            f"({len(fm_map)} failure modes).\n\n"
        )
        doc_parts: list[str] = [header]
        cursor = len(header)

        asset_eid = make_entity_id(f"equipment::{asset}")
        asset_chunk_ids: list[str] = []

        for idx, fm in enumerate(sorted(fm_map.keys())):
            sensors = sorted(fm_map[fm])
            sensor_list_str = ", ".join(sensors)
            section = (
                f"## {asset} :: {fm}\n"
                f"- Equipment: {asset}\n"
                f"- FailureMode: {fm}\n"
                f"- Relevant Monitoring Sensors ({len(sensors)}): {sensor_list_str}.\n"
                f"- Diagnostic Rule: In `{asset}`, failure mode `{fm}` is monitored by sensors: "
                f"{sensor_list_str}.\n\n"
            )
            start_char = cursor
            end_char = cursor + len(section)
            doc_parts.append(section)
            cursor = end_char

            chunk_id = f"{doc_id}::chunk_{idx:04d}"
            asset_chunk_ids.append(chunk_id)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source=rel_source,
                    index=idx,
                    text=section.strip(),
                    start_char=start_char,
                    end_char=end_char,
                    metadata={"asset": asset, "failure_mode": fm},
                )
            )

            fm_scoped_name = f"{asset} :: {fm}"
            fm_eid = make_entity_id(f"failure_mode::{fm_scoped_name}")
            graph_store.upsert_entity(
                Entity(
                    entity_id=fm_eid,
                    name=fm_scoped_name,
                    entity_type="FailureMode",
                    description=f"Failure mode `{fm}` on `{asset}`; monitored by: {sensor_list_str}.",
                    aliases=[fm],
                    source_chunk_ids=[chunk_id],
                    metadata={"asset": asset, "failure_category": fm},
                )
            )

            unique_categories.add(fm)
            cat_eid = make_entity_id(f"failure_category::{fm}")
            graph_store.upsert_entity(
                Entity(
                    entity_id=cat_eid,
                    name=fm,
                    entity_type="FailureCategory",
                    description=f"Canonical industrial failure category `{fm}` shared across assets.",
                    source_chunk_ids=[chunk_id],
                )
            )

            graph_store.upsert_relation(
                Relation(
                    relation_id=make_relation_id(asset_eid, "HAS_FAILURE_MODE", fm_eid),
                    source_id=asset_eid,
                    target_id=fm_eid,
                    relation_type="HAS_FAILURE_MODE",
                    description=f"{asset} can experience failure mode `{fm}`.",
                    source_chunk_ids=[chunk_id],
                )
            )

            graph_store.upsert_relation(
                Relation(
                    relation_id=make_relation_id(fm_eid, "INSTANCE_OF", cat_eid),
                    source_id=fm_eid,
                    target_id=cat_eid,
                    relation_type="INSTANCE_OF",
                    description=f"`{fm_scoped_name}` is an asset-specific instance of `{fm}`.",
                    source_chunk_ids=[chunk_id],
                )
            )

            for sensor in sensors:
                unique_sensors.add(sensor)
                s_eid = make_entity_id(f"signal_channel::{sensor}")
                graph_store.upsert_entity(
                    Entity(
                        entity_id=s_eid,
                        name=sensor,
                        entity_type="SignalChannel",
                        description=f"Industrial monitoring sensor/channel `{sensor}`.",
                        aliases=sorted(sensor_aliases.get(sensor, set())),
                        source_chunk_ids=[chunk_id],
                    )
                )
                graph_store.upsert_relation(
                    Relation(
                        relation_id=make_relation_id(fm_eid, "MONITORED_BY", s_eid),
                        source_id=fm_eid,
                        target_id=s_eid,
                        relation_type="MONITORED_BY",
                        description=f"In {asset}, `{fm}` is monitored by `{sensor}`.",
                        source_chunk_ids=[chunk_id],
                    )
                )
                gold_triple_count += 1

        graph_store.upsert_entity(
            Entity(
                entity_id=asset_eid,
                name=asset,
                entity_type="Equipment",
                description=f"Industrial equipment class `{asset}` with {len(fm_map)} ISO failure modes.",
                source_chunk_ids=asset_chunk_ids[:8],
            )
        )
        doc_file.write_text("".join(doc_parts), encoding="utf-8")

    if chunks:
        vectors = active_embedder.embed([c.text for c in chunks])
        vector_store.add_chunks(chunks, vectors)

    detector = CommunityDetector(
        resolution=config.community.resolution,
        seed=config.community.seed,
        summarize_with_llm=summarize_communities,
    )
    detector.detect_and_summarize(graph_store, llm=active_llm)

    graph_store.save(out_storage_dir)
    vector_store.save(out_storage_dir)
    (out_storage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "FailureSensorIQ (IBM Research, Apache-2.0)",
                "doc_count": len(manifest_docs),
                "chunk_count": len(chunks),
                "er_normalized_sensor_mentions": er_fixes,
                "documents": manifest_docs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = compute_audit_stats(graph_store, vector_store, doc_count=len(manifest_docs))
    return IngestStats(
        source_records=total_rows,
        gold_triples_loaded=gold_triple_count,
        asset_count=len(mapping),
        failure_mode_count=sum(len(v) for v in mapping.values()),
        failure_category_count=len(unique_categories),
        sensor_count=len(unique_sensors),
        er_normalized_sensor_mentions=er_fixes,
        audit=audit,
    )


def _format_mcqa_options(options: list[str], option_ids: list[str]) -> str:
    return "\n".join(
        f"{normalize_option_id(oid)}. {opt}"
        for oid, opt in zip(option_ids, options, strict=False)
    )


def _infer_relevancy_label(row: dict[str, Any]) -> str:
    existing = str(row.get("relevancy", "")).strip()
    if existing:
        return existing
    q = str(row.get("question", "")).lower()
    is_neg = any(
        kw in q for kw in ("not ", "irrelevant", "least relevant", "eliminate", "except", "unrelated")
    )
    asks_sensor = any(kw in q for kw in ("which sensor", "which sensors", "sensor from", "sensor out"))
    polarity = "irrelevant" if is_neg else "relevant"
    target = "sensors_for_failure_mode" if asks_sensor else "failure_modes_for_sensor"
    return f"{polarity}_{target}"


def _build_graphrag_mcqa_context(
    row: dict[str, Any],
    resolved_asset: str,
    graph_store: Any,
    vector_store: Any,
    embedder: BaseEmbeddingClient,
) -> tuple[str, list[str], int]:
    """Retrieve structured asset-exact KG subgraph + provenance chunks for an MCQA question."""
    question = str(row.get("question", ""))
    options = [str(o) for o in row.get("options", [])]

    asset_eid = make_entity_id(f"equipment::{resolved_asset}") if resolved_asset else ""
    seed_ids: list[str] = []
    if asset_eid and graph_store.get_entity(asset_eid) is not None:
        seed_ids.append(asset_eid)

    q_and_opts = f"{question} {' '.join(options)}".lower()
    all_entities = graph_store.get_all_entities()
    for ent in all_entities:
        if ent.entity_type == "FailureMode":
            if resolved_asset and ent.metadata.get("asset") != resolved_asset:
                continue
            fm_raw = str(ent.metadata.get("failure_category", "")).lower()
            if fm_raw and fm_raw in q_and_opts:
                seed_ids.append(ent.entity_id)
        elif ent.entity_type == "SignalChannel":
            names = [ent.name.lower(), *(a.lower() for a in ent.aliases)]
            if any(n and n in q_and_opts for n in names):
                seed_ids.append(ent.entity_id)

    seed_ids = list(dict.fromkeys(seed_ids))
    expanded_entities, sub_relations = graph_store.get_k_hop_subgraph(seed_ids, k_hops=1)

    ent_map = {e.entity_id: e for e in expanded_entities}
    asset_fm_ids = {
        e.entity_id
        for e in expanded_entities
        if e.entity_type == "FailureMode"
        and (not resolved_asset or e.metadata.get("asset") == resolved_asset)
    }

    fact_lines: list[str] = []
    fm_to_sensors: dict[str, list[str]] = defaultdict(list)
    for rel in sub_relations:
        if rel.relation_type == "MONITORED_BY" and rel.source_id in asset_fm_ids:
            fm_ent = ent_map.get(rel.source_id)
            s_ent = ent_map.get(rel.target_id)
            if fm_ent and s_ent:
                fm_label = str(fm_ent.metadata.get("failure_category", fm_ent.name))
                alias_note = f" (aliases: {', '.join(s_ent.aliases)})" if s_ent.aliases else ""
                fm_to_sensors[fm_label].append(f"{s_ent.name}{alias_note}")

    for fm_label in sorted(fm_to_sensors.keys()):
        sensors_joined = ", ".join(sorted(set(fm_to_sensors[fm_label])))
        fact_lines.append(
            f"- Asset `{resolved_asset or 'unknown'}` | FailureMode `{fm_label}` "
            f"--[MONITORED_BY]--> [{sensors_joined}]"
        )

    q_vec = embedder.embed([f"{resolved_asset} {question}"])[0]
    chunk_hits = [
        chunk
        for chunk, _ in vector_store.search(q_vec, top_k=6)
        if not resolved_asset or chunk.metadata.get("asset") == resolved_asset
    ][:3]

    sections = [
        f"Structured Knowledge Graph Facts for Asset `{resolved_asset or 'unknown'}` "
        "(ISO Complete Monitored-By List):\n"
        + ("\n".join(fact_lines) if fact_lines else "- (no asset triples matched)")
    ]
    if chunk_hits:
        sections.append(
            "Source Provenance Chunks:\n"
            + "\n".join(f"- [{c.chunk_id}] {c.text}" for c in chunk_hits)
        )

    seed_names = [ent_map[eid].name for eid in seed_ids if eid in ent_map]
    return "\n\n".join(sections), seed_names, len(expanded_entities)


def _call_llm_complete(
    llm: BaseLLMClient,
    prompt: str,
    *,
    system_prompt: str,
    temperature: float,
    json_mode: bool,
    max_tokens: int | None,
) -> str:
    """Invoke `llm.complete` passing `max_tokens` only when supported by the provider."""
    sig = inspect.signature(llm.complete)
    kwargs: dict[str, Any] = {
        "system_prompt": system_prompt,
        "temperature": temperature,
        "json_mode": json_mode,
    }
    if max_tokens is not None and "max_tokens" in sig.parameters:
        kwargs["max_tokens"] = max_tokens
    return llm.complete(prompt, **kwargs)


def _sample_stratified_rows(
    rows: list[dict[str, Any]],
    known_assets: list[str],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        asset = str(row.get("asset_name", "")).strip().lower()
        if not asset:
            asset = detect_asset_from_question(str(row.get("question", "")), known_assets) or "unknown"
        rel = _infer_relevancy_label(row)
        by_bucket[f"{rel}|{asset}"].append(row)

    buckets = sorted(by_bucket.keys())
    rng.shuffle(buckets)
    for b in buckets:
        rng.shuffle(by_bucket[b])

    sampled: list[dict[str, Any]] = []
    round_idx = 0
    target = min(limit, len(rows))
    while len(sampled) < target:
        progressed = False
        for b in buckets:
            if round_idx < len(by_bucket[b]):
                sampled.append(by_bucket[b][round_idx])
                progressed = True
                if len(sampled) >= target:
                    break
        if not progressed:
            break
        round_idx += 1
    return sampled


def evaluate_failuresensoriq(
    questions_path: Path,
    config: AppConfig,
    *,
    suite_name: str = "failuresensoriq",
    storage_dir: Path | None = None,
    output_dir: Path | None = None,
    limit: int = 20,
    seed: int = 42,
    modes: list[str] | None = None,
    concise_json: bool = True,
    resume: bool = True,
    llm: BaseLLMClient | None = None,
    embedder: BaseEmbeddingClient | None = None,
) -> tuple[EvaluationSummary, Path]:
    """Run quantitative MCQA benchmark comparing `llm_only`, `naive_rag`, and `graphrag_kg`."""
    if not questions_path.is_file():
        raise FileNotFoundError(f"FailureSensorIQ questions file not found: {questions_path}")

    selected_modes = modes or ["llm_only", "naive_rag", "graphrag_kg"]
    src_storage = storage_dir or config.storage_path
    out_dir = output_dir or config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_store = create_graph_store()
    vector_store = create_vector_store()
    known_assets: list[str] = []
    if src_storage.is_dir() and (src_storage / "graph.json").is_file():
        graph_store.load(src_storage)
        vector_store.load(src_storage)
        known_assets = [
            e.name for e in graph_store.get_all_entities() if e.entity_type == "Equipment"
        ]
    elif any(m in {"naive_rag", "graphrag_kg"} for m in selected_modes):
        graph_store.load(src_storage)
        vector_store.load(src_storage)

    active_llm = llm or create_llm_client(config)
    active_embedder = embedder or create_embedding_client(config)

    rows: list[dict[str, Any]] = []
    with questions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    sampled = _sample_stratified_rows(rows, known_assets, limit=limit, seed=seed)

    details: dict[str, list[EvalQuestionOutcome]] = {m: [] for m in selected_modes}
    mode_metrics: list[ModeMetrics] = []

    for mode in selected_modes:
        ckpt_path = out_dir / f"{suite_name}_{mode}.checkpoint.jsonl"
        cached: dict[int, EvalQuestionOutcome] = {}
        if resume and ckpt_path.is_file():
            with ckpt_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        item = EvalQuestionOutcome(**json.loads(line))
                        cached[item.question_id] = item

        rel_totals: dict[str, int] = defaultdict(int)
        rel_correct: dict[str, int] = defaultdict(int)

        with ckpt_path.open("a", encoding="utf-8") as ckpt_fh:
            for idx_row, row in enumerate(sampled):
                qid = int(row.get("id", idx_row))
                if qid in cached:
                    outcome = cached[qid]
                    details[mode].append(outcome)
                    rel_totals[outcome.relevancy] += 1
                    if outcome.is_correct:
                        rel_correct[outcome.relevancy] += 1
                    continue

                options = [str(o) for o in row["options"]]
                raw_option_ids = [str(oid) for oid in row["option_ids"]]
                norm_option_ids = [normalize_option_id(oid) for oid in raw_option_ids]
                correct_flags = [bool(c) for c in row["correct"]]
                gold_list = sorted(
                    oid for oid, ok in zip(norm_option_ids, correct_flags, strict=False) if ok
                )
                expected_count = max(1, len(gold_list))

                options_block = _format_mcqa_options(options, raw_option_ids)
                question = str(row["question"])
                explicit_asset = re.sub(r"\s+", " ", str(row.get("asset_name", "")).strip()).lower()
                resolved_asset = explicit_asset or detect_asset_from_question(question, known_assets)
                relevancy = _infer_relevancy_label(row)

                if concise_json:
                    if expected_count == 1:
                        mcqa_system = (
                            "You are an industrial reliability & fault diagnosis expert. "
                            'Return ONLY a compact JSON object {"answers": ["<ID>"]} containing '
                            "exactly 1 option ID from the given choices. No explanation."
                        )
                        instruction = (
                            'Select the single best option ID and return strictly JSON: {"answers": ["<ID>"]}.'
                        )
                    else:
                        mcqa_system = (
                            "You are an industrial reliability & fault diagnosis expert. "
                            f'Return ONLY a compact JSON object {{"answers": ["<ID1>", ...]}} '
                            f"containing exactly {expected_count} distinct option IDs. No explanation."
                        )
                        instruction = (
                            f"Select all {expected_count} correct option IDs and return strictly JSON: "
                            '{"answers": ["<ID1>", "<ID2>"]}.'
                        )
                else:
                    mcqa_system = (
                        "You are an industrial reliability & fault diagnosis expert answering a "
                        "multiple-choice question. End your response with `Answer: <ID>`."
                    )
                    instruction = (
                        f"Select exactly {expected_count} option ID(s) and end with `Answer: <ID>`."
                    )

                seed_names: list[str] = []
                expanded_nodes = 0

                if mode == "llm_only":
                    prompt = f"Question: {question}\n\nOptions:\n{options_block}\n\n{instruction}"
                elif mode == "naive_rag":
                    q_vec = active_embedder.embed([f"{resolved_asset} {question}"])[0]
                    hits = vector_store.search(q_vec, top_k=4)
                    ctx = "\n".join(f"- [{c.chunk_id}] {c.text}" for c, _ in hits)
                    prompt = (
                        f"Retrieved Text Chunks:\n{ctx}\n\n"
                        f"Question: {question}\n\nOptions:\n{options_block}\n\n{instruction}"
                    )
                else:
                    kg_ctx, seed_names, expanded_nodes = _build_graphrag_mcqa_context(
                        row, resolved_asset, graph_store, vector_store, active_embedder
                    )
                    prompt = (
                        f"{kg_ctx}\n\n"
                        f"Question: {question}\n\nOptions:\n{options_block}\n\n"
                        "Note: The KG list above enumerates the ISO-relevant sensors for each listed "
                        "failure mode on this asset. If the question asks for IRRELEVANT / NOT-relevant "
                        f"option(s), select the option(s) NOT linked in the KG.\n{instruction}"
                    )

                t0 = time.perf_counter()
                raw_resp = _call_llm_complete(
                    active_llm,
                    prompt,
                    system_prompt=mcqa_system,
                    temperature=0.0,
                    json_mode=concise_json,
                    max_tokens=96 if concise_json else 768,
                )
                dt = round(time.perf_counter() - t0, 3)

                pred_list = sorted(
                    parse_option_answers(raw_resp, raw_option_ids, expected_count=expected_count)
                )
                is_ok = pred_list == gold_list
                f1 = compute_set_f1(pred_list, gold_list)

                rel_totals[relevancy] += 1
                if is_ok:
                    rel_correct[relevancy] += 1

                outcome = EvalQuestionOutcome(
                    question_id=qid,
                    asset=resolved_asset,
                    asset_linked=bool(resolved_asset),
                    relevancy=relevancy,
                    question_type=str(row.get("question_type", f"k={expected_count}")),
                    gold_option=",".join(gold_list),
                    predicted_option=",".join(pred_list),
                    gold_options=gold_list,
                    predicted_options=pred_list,
                    is_correct=is_ok,
                    f1_score=f1,
                    latency_seconds=dt,
                    raw_response=raw_resp[:400],
                    seed_entities=seed_names[:8],
                    expanded_nodes=expanded_nodes,
                )
                details[mode].append(outcome)
                ckpt_fh.write(outcome.model_dump_json() + "\n")
                ckpt_fh.flush()

        items = details[mode]
        total_n = len(items)
        correct_n = sum(1 for item in items if item.is_correct)
        acc = round(correct_n / total_n, 4) if total_n else 0.0
        macro_f1 = round(sum(item.f1_score for item in items) / total_n, 4) if total_n else 0.0
        empty_n = sum(1 for item in items if not item.raw_response.strip())
        linked_n = sum(1 for item in items if item.asset_linked)
        avg_lat = round(sum(item.latency_seconds for item in items) / total_n, 3) if total_n else 0.0

        by_rel = {
            rel: {
                "correct": rel_correct[rel],
                "total": rel_totals[rel],
                "accuracy": round(rel_correct[rel] / rel_totals[rel], 4) if rel_totals[rel] else 0.0,
            }
            for rel in sorted(rel_totals.keys())
        }
        mode_metrics.append(
            ModeMetrics(
                mode=mode,
                total=total_n,
                correct=correct_n,
                accuracy=acc,
                macro_f1=macro_f1,
                empty_responses=empty_n,
                asset_linked_count=linked_n,
                avg_latency_seconds=avg_lat,
                by_relevancy=by_rel,
            )
        )

    summary = EvaluationSummary(
        suite_name=suite_name,
        total_questions=len(sampled),
        seed=seed,
        llm_provider=config.env.llm_provider,
        llm_model=config.env.llm_model,
        concise_json=concise_json,
        modes=mode_metrics,
        details=details,
    )

    (out_dir / "eval_results.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    report_path = out_dir / "eval_report.md"
    lines = [
        f"# FailureSensorIQ 量化评测报告 — `{suite_name}`",
        "",
        f"- **评测样本数**：{summary.total_questions}（分层随机抽样，`seed={summary.seed}`）",
        f"- **模型配置**：`{summary.llm_provider}` / `{summary.llm_model}` | `concise_json={concise_json}`",
        (
            "- **评测说明**：对比参数记忆推理（`llm_only`）、纯向量片段召回（`naive_rag`）"
            "与 ISO 知识图谱增强推理（`graphrag_kg`）在工业设备失效模式-传感器关联问答上的准确率。"
        ),
        "",
        "## 1. 总体指标对比",
        "",
        "| 检索/推理模式 | Exact Match | 准确率 | Macro-F1 | 空回复数 | 资产链接率 | 平均耗时 (s/题) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in summary.modes:
        link_pct = (m.asset_linked_count / m.total * 100) if m.total else 0.0
        lines.append(
            f"| `{m.mode}` | {m.correct} / {m.total} | **{m.accuracy * 100:.1f}%** | "
            f"{m.macro_f1 * 100:.1f}% | {m.empty_responses} | {link_pct:.0f}% | {m.avg_latency_seconds:.2f}s |"
        )

    lines.extend(["", "## 2. 分题型准确率拆解", ""])
    for m in summary.modes:
        lines.append(f"### 模式：`{m.mode}`")
        lines.append("| 题型 (Relevancy) | 正确 / 总数 | 准确率 |")
        lines.append("|---|---|---|")
        for rel, stat in m.by_relevancy.items():
            lines.append(
                f"| `{rel}` | {stat['correct']} / {stat['total']} | {float(stat['accuracy']) * 100:.1f}% |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary, report_path


def run_failuresensoriq_stress_suite(
    data_dir: Path,
    config: AppConfig,
    *,
    storage_dir: Path | None = None,
    output_dir: Path | None = None,
    limit_per_suite: int = 100,
    seed: int = 42,
    modes: list[str] | None = None,
    concise_json: bool = True,
    resume: bool = True,
    suites: list[tuple[str, str]] | None = None,
    llm: BaseLLMClient | None = None,
    embedder: BaseEmbeddingClient | None = None,
) -> tuple[StressSuiteReport, Path]:
    """Run all 4 FailureSensorIQ difficulty suites with checkpoint/resume and unified reporting."""
    active_suites = suites or DEFAULT_STRESS_SUITES
    base_out = (output_dir or config.output_path) / "stress_test"
    base_out.mkdir(parents=True, exist_ok=True)

    summaries: list[EvaluationSummary] = []
    for suite_name, filename in active_suites:
        q_path = data_dir / filename
        if not q_path.is_file():
            raise FileNotFoundError(f"Missing FailureSensorIQ suite file: {q_path}")
        suite_summary, _ = evaluate_failuresensoriq(
            questions_path=q_path,
            config=config,
            suite_name=suite_name,
            storage_dir=storage_dir,
            output_dir=base_out / suite_name,
            limit=limit_per_suite,
            seed=seed,
            modes=modes,
            concise_json=concise_json,
            resume=resume,
            llm=llm,
            embedder=embedder,
        )
        summaries.append(suite_summary)

    report = StressSuiteReport(
        limit_per_suite=limit_per_suite,
        seed=seed,
        llm_provider=config.env.llm_provider,
        llm_model=config.env.llm_model,
        concise_json=concise_json,
        suites=summaries,
    )

    (base_out / "stress_test_summary.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    report_md = base_out / "stress_test_report.md"

    total_q_all = sum(s.total_questions for s in summaries)
    lines = [
        "# FailureSensorIQ 四级难度极限压测汇总报告",
        "",
        f"- **每集抽样题数**：`{limit_per_suite}` 题（共 `{total_q_all}` 题，`seed={seed}`）",
        f"- **模型配置**：`{report.llm_provider}` / `{report.llm_model}` | `concise_json={concise_json}`",
        "",
        "## 1. 四级难度 Exact Match / Macro-F1 对比总表",
        "",
        "| 难度分级 (Suite) | 题数 | `llm_only` | `naive_rag` | `graphrag_kg` | KG 相对提升 (vs LLM) | 平均延迟 (KG) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        by_m = {m.mode: m for m in s.modes}

        def _fmt(m: ModeMetrics | None, suite: str = s.suite_name) -> str:
            if m is None:
                return "-"
            if suite == "L3_multi_answers":
                return f"**{m.accuracy * 100:.1f}%** (F1 {m.macro_f1 * 100:.1f}%)"
            return f"**{m.accuracy * 100:.1f}%** ({m.correct}/{m.total})"

        llm_m = by_m.get("llm_only")
        rag_m = by_m.get("naive_rag")
        kg_m = by_m.get("graphrag_kg")
        delta = (
            f"+{(kg_m.accuracy - llm_m.accuracy) * 100:.1f}%"
            if (kg_m and llm_m)
            else "-"
        )
        kg_lat = f"{kg_m.avg_latency_seconds:.2f}s" if kg_m else "-"
        lines.append(
            f"| `{s.suite_name}` | {s.total_questions} | {_fmt(llm_m)} | {_fmt(rag_m)} | "
            f"{_fmt(kg_m)} | {delta} | {kg_lat} |"
        )

    lines.extend(["", "## 2. 稳定性与实体链接诊断", ""])
    lines.append("| 难度分级 | 模式 | 空回复数 | 资产自动链接率 | 平均耗时 (s/题) |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        for m in s.modes:
            link_pct = (m.asset_linked_count / m.total * 100) if m.total else 0.0
            lines.append(
                f"| `{s.suite_name}` | `{m.mode}` | {m.empty_responses} | "
                f"{link_pct:.1f}% | {m.avg_latency_seconds:.2f}s |"
            )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, report_md

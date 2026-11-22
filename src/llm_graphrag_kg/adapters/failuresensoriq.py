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
from llm_graphrag_kg.harness.context_compactor import estimate_tokens
from llm_graphrag_kg.harness.runtime import LLMHarness
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


class MCQAAnswerSchema(BaseModel):
    """Structured JSON output contract enforced by `LLMHarness.run_structured`."""

    answers: list[str] = Field(min_length=1)


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
    prompt_tokens_est: int = 0
    repair_attempts: int = 0
    raw_response: str
    seed_entities: list[str] = Field(default_factory=list)
    expanded_nodes: int = 0


class ModeMetrics(BaseModel):
    """Aggregated accuracy, F1, empty-rate, token budget, and latency metrics for one mode."""

    mode: str
    total: int
    correct: int
    accuracy: float
    macro_f1: float = 0.0
    empty_responses: int = 0
    asset_linked_count: int = 0
    avg_latency_seconds: float = 0.0
    avg_prompt_tokens: float = 0.0
    total_repairs: int = 0
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
    *,
    compact_anchor_only: bool = True,
) -> tuple[str, list[str], int]:
    """Retrieve structured asset-exact KG subgraph for an MCQA question.

    When `compact_anchor_only=True` (Stage 2.1 L0 compaction), seeds are strictly extracted
    from the question text (stripping the asset name to prevent substring collisions such as
    `power` inside `power transformer`) rather than letting distractor options bloat the subgraph.
    """
    question = str(row.get("question", ""))
    options = [str(o) for o in row.get("options", [])]
    all_entities = graph_store.get_all_entities()
    all_relations = graph_store.get_all_relations()

    asset_eid = make_entity_id(f"equipment::{resolved_asset}") if resolved_asset else ""

    if compact_anchor_only and resolved_asset:
        q_low = re.sub(r"\s+", " ", question.strip()).lower()
        q_no_asset = re.sub(re.escape(resolved_asset), " ", q_low)

        # 1. Match FailureModes explicitly mentioned in the question for this asset (longest first)
        asset_fms = [
            e
            for e in all_entities
            if e.entity_type == "FailureMode" and e.metadata.get("asset") == resolved_asset
        ]
        matched_fms: list[Entity] = []
        for ent in sorted(
            asset_fms,
            key=lambda e: len(str(e.metadata.get("failure_category", ""))),
            reverse=True,
        ):
            fm_raw = str(ent.metadata.get("failure_category", "")).lower()
            if fm_raw and fm_raw in q_no_asset:
                matched_fms.append(ent)

        # 2. Match SignalChannels explicitly mentioned in the question (longest first)
        all_sensors = [e for e in all_entities if e.entity_type == "SignalChannel"]
        matched_sensors: list[Entity] = []
        occupied_spans: list[str] = []
        for ent in sorted(
            all_sensors,
            key=lambda e: max([len(e.name), *(len(a) for a in e.aliases)]),
            reverse=True,
        ):
            cand_names = sorted(
                [ent.name.lower(), *(a.lower() for a in ent.aliases)],
                key=len,
                reverse=True,
            )
            for cname in cand_names:
                if not cname or len(cname) < 3:
                    continue
                if re.search(rf"(?<![a-z0-9]){re.escape(cname)}(?![a-z0-9])", q_no_asset):
                    if not any(cname in prev for prev in occupied_spans):
                        matched_sensors.append(ent)
                        occupied_spans.append(cname)
                    break

        ent_by_id = {e.entity_id: e for e in all_entities}
        asset_fm_id_set = {e.entity_id for e in asset_fms}

        # Direction A: Question specifies FailureMode(s) -> return exact monitored sensors
        if matched_fms:
            target_fm_ids = {e.entity_id for e in matched_fms}
            fm_to_sensors: dict[str, list[str]] = defaultdict(list)
            chunk_ids: list[str] = []
            for rel in all_relations:
                if rel.relation_type == "MONITORED_BY" and rel.source_id in target_fm_ids:
                    fm_ent = ent_by_id.get(rel.source_id)
                    s_ent = ent_by_id.get(rel.target_id)
                    if fm_ent and s_ent:
                        fm_label = str(fm_ent.metadata.get("failure_category", fm_ent.name))
                        alias_note = f" (aliases: {', '.join(s_ent.aliases)})" if s_ent.aliases else ""
                        fm_to_sensors[fm_label].append(f"{s_ent.name}{alias_note}")
                        chunk_ids.extend(rel.source_chunk_ids)

            fact_lines = [
                f"- Asset `{resolved_asset}` | FailureMode `{fm_label}` "
                f"--[MONITORED_BY]--> [{', '.join(sorted(set(sensors)))}]"
                for fm_label, sensors in sorted(fm_to_sensors.items())
            ]
            seed_names = [resolved_asset, *(e.name for e in matched_fms)]
            expanded_count = 1 + len(matched_fms) + sum(len(set(v)) for v in fm_to_sensors.values())
            cid_note = f" (chunks: {', '.join(dict.fromkeys(chunk_ids[:2]))})" if chunk_ids else ""
            ctx = (
                f"Structured Knowledge Graph Facts for Asset `{resolved_asset}`{cid_note} "
                "(L0 Anchor-Compacted ISO Complete List):\n"
                + ("\n".join(fact_lines) if fact_lines else "- (no sensors linked)")
            )
            return ctx, seed_names, expanded_count

        # Direction B: Question specifies SignalChannel(s) -> return exact asset FailureModes monitored by it
        if matched_sensors:
            target_s_ids = {e.entity_id for e in matched_sensors}
            sensor_to_fms: dict[str, list[str]] = defaultdict(list)
            chunk_ids = []
            for rel in all_relations:
                if (
                    rel.relation_type == "MONITORED_BY"
                    and rel.source_id in asset_fm_id_set
                    and rel.target_id in target_s_ids
                ):
                    fm_ent = ent_by_id.get(rel.source_id)
                    s_ent = ent_by_id.get(rel.target_id)
                    if fm_ent and s_ent:
                        fm_label = str(fm_ent.metadata.get("failure_category", fm_ent.name))
                        sensor_to_fms[s_ent.name].append(fm_label)
                        chunk_ids.extend(rel.source_chunk_ids)

            fact_lines = []
            for s_ent in matched_sensors:
                fms_for_s = sorted(set(sensor_to_fms.get(s_ent.name, [])))
                alias_note = f" (aliases: {', '.join(s_ent.aliases)})" if s_ent.aliases else ""
                fact_lines.append(
                    f"- Asset `{resolved_asset}` | SignalChannel `{s_ent.name}{alias_note}` "
                    f"monitors ONLY these FailureModes: [{', '.join(fms_for_s) if fms_for_s else 'NONE'}]"
                )
            seed_names = [resolved_asset, *(e.name for e in matched_sensors)]
            expanded_count = 1 + len(matched_sensors) + sum(len(set(v)) for v in sensor_to_fms.values())
            cid_note = f" (chunks: {', '.join(dict.fromkeys(chunk_ids[:3]))})" if chunk_ids else ""
            ctx = (
                f"Structured Knowledge Graph Facts for Asset `{resolved_asset}`{cid_note} "
                "(L0 Anchor-Compacted ISO Complete List):\n"
                + "\n".join(fact_lines)
            )
            return ctx, seed_names, expanded_count

    # Legacy / fallback path: expand from question + options
    seed_ids: list[str] = []
    if asset_eid and graph_store.get_entity(asset_eid) is not None:
        seed_ids.append(asset_eid)

    q_and_opts = f"{question} {' '.join(options)}".lower()
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

    fact_lines = []
    fm_to_sensors_fb: dict[str, list[str]] = defaultdict(list)
    for rel in sub_relations:
        if rel.relation_type == "MONITORED_BY" and rel.source_id in asset_fm_ids:
            fm_ent = ent_map.get(rel.source_id)
            s_ent = ent_map.get(rel.target_id)
            if fm_ent and s_ent:
                fm_label = str(fm_ent.metadata.get("failure_category", fm_ent.name))
                alias_note = f" (aliases: {', '.join(s_ent.aliases)})" if s_ent.aliases else ""
                fm_to_sensors_fb[fm_label].append(f"{s_ent.name}{alias_note}")

    for fm_label in sorted(fm_to_sensors_fb.keys()):
        sensors_joined = ", ".join(sorted(set(fm_to_sensors_fb[fm_label])))
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
    if limit <= 0 or limit >= len(rows):
        return list(rows)

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


def compute_pairwise_gaps(
    details: dict[str, list[EvalQuestionOutcome]],
    *,
    primary_mode: str = "graphrag_kg",
) -> list[dict[str, Any]]:
    """Compute exact per-question Win/Loss gap matrix comparing primary_mode vs every baseline."""
    if primary_mode not in details:
        return []
    primary_by_id = {item.question_id: item.is_correct for item in details[primary_mode]}
    gaps: list[dict[str, Any]] = []
    for other_mode, items in details.items():
        if other_mode == primary_mode:
            continue
        both_ok = both_bad = primary_only = other_only = 0
        for item in items:
            if item.question_id not in primary_by_id:
                continue
            p_ok = primary_by_id[item.question_id]
            o_ok = item.is_correct
            if p_ok and o_ok:
                both_ok += 1
            elif p_ok and not o_ok:
                primary_only += 1
            elif not p_ok and o_ok:
                other_only += 1
            else:
                both_bad += 1
        total = both_ok + both_bad + primary_only + other_only
        net = primary_only - other_only
        gaps.append(
            {
                "primary": primary_mode,
                "baseline": other_mode,
                "total": total,
                "both_correct": both_ok,
                "primary_wins": primary_only,
                "baseline_wins": other_only,
                "both_wrong": both_bad,
                "net_gain_questions": net,
                "net_gain_pct": round((net / total) * 100.0, 2) if total else 0.0,
            }
        )
    return gaps


MODE_ESTIMATED_SECONDS: dict[str, float] = {
    "llm_only": 0.65,
    "naive_rag": 1.00,
    "graphrag_kg_raw": 1.50,
    "graphrag_kg": 0.96,
}


def plan_failuresensoriq_stress_suite(
    data_dir: Path,
    config: AppConfig,
    *,
    output_dir: Path | None = None,
    output_tag: str = "stress_test",
    limit_per_suite: int = 0,
    modes: list[str] | None = None,
    resume: bool = True,
    suites: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Estimate exact question counts, cached checkpoint progress, and wall time before launching."""
    active_suites = suites or DEFAULT_STRESS_SUITES
    selected_modes = modes or ["llm_only", "naive_rag", "graphrag_kg_raw", "graphrag_kg"]
    base_out = (output_dir or config.output_path) / output_tag

    plan_rows: list[dict[str, Any]] = []
    for suite_name, filename in active_suites:
        q_path = data_dir / filename
        if not q_path.is_file():
            raise FileNotFoundError(f"Missing FailureSensorIQ suite file: {q_path}")
        with q_path.open("r", encoding="utf-8") as fh:
            total_in_file = sum(1 for line in fh if line.strip())
        target_q = total_in_file if limit_per_suite <= 0 else min(limit_per_suite, total_in_file)

        for mode in selected_modes:
            ckpt_path = base_out / suite_name / f"{suite_name}_{mode}.checkpoint.jsonl"
            done_q = 0
            if resume and ckpt_path.is_file():
                seen_ids: set[int] = set()
                with ckpt_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            seen_ids.add(int(json.loads(line).get("question_id", -1)))
                done_q = min(len(seen_ids), target_q)
            remaining_q = max(0, target_q - done_q)
            est_sec = remaining_q * MODE_ESTIMATED_SECONDS.get(mode, 1.1)
            plan_rows.append(
                {
                    "suite": suite_name,
                    "mode": mode,
                    "total_in_file": total_in_file,
                    "target_questions": target_q,
                    "cached_done": done_q,
                    "remaining_calls": remaining_q,
                    "est_minutes": round(est_sec / 60.0, 1),
                }
            )
    return plan_rows


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
    compact_anchor_only: bool = True,
    resume: bool = True,
    progress_every: int = 50,
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
        mode_start = time.perf_counter()
        newly_evaluated = 0
        total_target = len(sampled)

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
                    use_compact_mode = False if mode == "graphrag_kg_raw" else compact_anchor_only
                    kg_ctx, seed_names, expanded_nodes = _build_graphrag_mcqa_context(
                        row,
                        resolved_asset,
                        graph_store,
                        vector_store,
                        active_embedder,
                        compact_anchor_only=use_compact_mode,
                    )
                    prompt = (
                        f"{kg_ctx}\n\n"
                        f"Question: {question}\n\nOptions:\n{options_block}\n\n"
                        "Note: The KG list above enumerates the ISO-relevant links for the target "
                        "entity on this asset. If the question asks for IRRELEVANT / NOT-relevant "
                        f"option(s), select the option(s) NOT listed in the KG.\n{instruction}"
                    )

                prompt_toks = estimate_tokens(prompt) + estimate_tokens(mcqa_system)
                repairs_used = 0
                raw_resp = ""
                pred_list: list[str] = []
                t0 = time.perf_counter()

                for transient_try in range(2):
                    try:
                        if concise_json:
                            valid_norm_set = set(norm_option_ids)

                            def _validate_mcqa(
                                model_out: MCQAAnswerSchema,
                                expected_n: int = expected_count,
                                valid_set: set[str] = valid_norm_set,
                            ) -> str | None:
                                cleaned_ids = list(
                                    dict.fromkeys(normalize_option_id(x) for x in model_out.answers)
                                )
                                bad = [x for x in cleaned_ids if x not in valid_set]
                                if bad:
                                    return f"Invalid option IDs {bad}; allowed={sorted(valid_set)}"
                                if len(cleaned_ids) != expected_n:
                                    return (
                                        f"Expected exactly {expected_n} distinct option ID(s), "
                                        f"got {len(cleaned_ids)}"
                                    )
                                return None

                            harness = LLMHarness(
                                active_llm,
                                json_repair_retries=config.harness.json_repair_retries,
                                max_calls=config.harness.json_repair_retries + 2,
                                max_wall_seconds=config.harness.max_wall_seconds,
                            )
                            try:
                                parsed_obj, raw_resp = harness.run_structured(
                                    prompt,
                                    MCQAAnswerSchema,
                                    step_name=f"fsiq_{mode}_q{qid}",
                                    system_prompt=mcqa_system,
                                    temperature=0.0,
                                    max_tokens=96,
                                    extra_validator=_validate_mcqa,
                                )
                                pred_list = sorted(
                                    list(
                                        dict.fromkeys(
                                            normalize_option_id(x) for x in parsed_obj.answers
                                        )
                                    )
                                )
                                repairs_used = harness.ledger.total_repairs
                            except ValueError:
                                repairs_used = harness.ledger.total_repairs
                                raw_resp = _call_llm_complete(
                                    active_llm,
                                    prompt,
                                    system_prompt=mcqa_system,
                                    temperature=0.0,
                                    json_mode=True,
                                    max_tokens=96,
                                )
                                pred_list = sorted(
                                    parse_option_answers(
                                        raw_resp, raw_option_ids, expected_count=expected_count
                                    )
                                )
                        else:
                            raw_resp = _call_llm_complete(
                                active_llm,
                                prompt,
                                system_prompt=mcqa_system,
                                temperature=0.0,
                                json_mode=False,
                                max_tokens=768,
                            )
                            pred_list = sorted(
                                parse_option_answers(
                                    raw_resp, raw_option_ids, expected_count=expected_count
                                )
                            )
                        break
                    except (TimeoutError, OSError, ConnectionError) as exc:
                        if transient_try == 0:
                            time.sleep(2.0)
                            continue
                        raw_resp = f"[TRANSIENT_ERROR] {type(exc).__name__}: {exc}"
                        pred_list = []

                dt = round(time.perf_counter() - t0, 3)

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
                    prompt_tokens_est=prompt_toks,
                    repair_attempts=repairs_used,
                    raw_response=raw_resp[:400],
                    seed_entities=seed_names[:8],
                    expanded_nodes=expanded_nodes,
                )
                details[mode].append(outcome)
                ckpt_fh.write(outcome.model_dump_json() + "\n")
                ckpt_fh.flush()

                newly_evaluated += 1
                done_now = len(details[mode])
                if progress_every > 0 and (done_now % progress_every == 0 or done_now == total_target):
                    elapsed = max(time.perf_counter() - mode_start, 1e-6)
                    rate = newly_evaluated / elapsed
                    rem_min = ((total_target - done_now) / rate) / 60.0 if rate > 0 else 0.0
                    cur_acc = sum(1 for x in details[mode] if x.is_correct) / done_now * 100.0
                    print(
                        f"[{suite_name} | {mode}] {done_now}/{total_target} "
                        f"({done_now / total_target * 100:.1f}%) | acc={cur_acc:.1f}% | "
                        f"avg={elapsed / newly_evaluated:.2f}s | ETA={rem_min:.1f}m",
                        flush=True,
                    )

        items = details[mode]
        total_n = len(items)
        correct_n = sum(1 for item in items if item.is_correct)
        acc = round(correct_n / total_n, 4) if total_n else 0.0
        macro_f1 = round(sum(item.f1_score for item in items) / total_n, 4) if total_n else 0.0
        empty_n = sum(1 for item in items if not item.raw_response.strip())
        linked_n = sum(1 for item in items if item.asset_linked)
        avg_lat = round(sum(item.latency_seconds for item in items) / total_n, 3) if total_n else 0.0
        avg_ptok = round(sum(item.prompt_tokens_est for item in items) / total_n, 1) if total_n else 0.0
        repairs_n = sum(item.repair_attempts for item in items)

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
                avg_prompt_tokens=avg_ptok,
                total_repairs=repairs_n,
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
        "| 检索/推理模式 | Exact Match | 准确率 | Macro-F1 | 平均 Prompt Tok | 修复重试 | 空回复 | 平均耗时 (s/题) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in summary.modes:
        lines.append(
            f"| `{m.mode}` | {m.correct} / {m.total} | **{m.accuracy * 100:.1f}%** | "
            f"{m.macro_f1 * 100:.1f}% | {m.avg_prompt_tokens:.0f} | {m.total_repairs} | "
            f"{m.empty_responses} | {m.avg_latency_seconds:.2f}s |"
        )

    pairwise = compute_pairwise_gaps(details, primary_mode="graphrag_kg")
    if pairwise:
        lines.extend(
            [
                "",
                "## 2. 逐题胜负差距矩阵 (Pairwise Win/Loss vs `graphrag_kg`)",
                "",
                "| 对比基线 | 共同答对 | 仅 KG 答对 (Win) | 仅基线答对 (Loss) | 共同答错 | 净胜题数 (Net Gain) |",
                "|---|---|---|---|---|---|",
            ]
        )
        for g in pairwise:
            sign = "+" if g["net_gain_questions"] >= 0 else ""
            lines.append(
                f"| `graphrag_kg` vs `{g['baseline']}` | {g['both_correct']} | "
                f"**{g['primary_wins']}** | {g['baseline_wins']} | {g['both_wrong']} | "
                f"**{sign}{g['net_gain_questions']} ({sign}{g['net_gain_pct']:.2f}%)** |"
            )

    lines.extend(["", "## 3. 分题型准确率拆解", ""])
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
    output_tag: str = "stress_test",
    limit_per_suite: int = 100,
    seed: int = 42,
    modes: list[str] | None = None,
    concise_json: bool = True,
    compact_anchor_only: bool = True,
    resume: bool = True,
    progress_every: int = 50,
    suites: list[tuple[str, str]] | None = None,
    llm: BaseLLMClient | None = None,
    embedder: BaseEmbeddingClient | None = None,
) -> tuple[StressSuiteReport, Path]:
    """Run FailureSensorIQ difficulty suites with checkpoint/resume and unified gap reporting."""
    active_suites = suites or DEFAULT_STRESS_SUITES
    base_out = (output_dir or config.output_path) / output_tag
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
            compact_anchor_only=compact_anchor_only,
            resume=resume,
            progress_every=progress_every,
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
    scope_desc = "全量 (ALL)" if limit_per_suite <= 0 else f"每集 `{limit_per_suite}` 题"
    lines = [
        "# FailureSensorIQ 全量 / 多级难度基准对比报告",
        "",
        f"- **评测规模**：{scope_desc}（累计 `{total_q_all}` 题，`seed={seed}`）",
        f"- **模型配置**：`{report.llm_provider}` / `{report.llm_model}` | `concise_json={concise_json}`",
        "",
        "## 1. 多基线 Exact Match / Macro-F1 对比总表",
        "",
        (
            "| 难度分级 (Suite) | 题数 | `llm_only` | `naive_rag` | `graphrag_kg_raw` | "
            "`graphrag_kg` | KG vs LLM | Compact vs Raw |"
        ),
        "|---|---|---|---|---|---|---|---|",
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
        raw_m = by_m.get("graphrag_kg_raw")
        kg_m = by_m.get("graphrag_kg")
        delta_llm = (
            f"{(kg_m.accuracy - llm_m.accuracy) * 100:+.1f}%"
            if (kg_m and llm_m)
            else "-"
        )
        delta_raw = (
            f"{(kg_m.accuracy - raw_m.accuracy) * 100:+.1f}%"
            if (kg_m and raw_m)
            else "-"
        )
        lines.append(
            f"| `{s.suite_name}` | {s.total_questions} | {_fmt(llm_m)} | {_fmt(rag_m)} | "
            f"{_fmt(raw_m)} | {_fmt(kg_m)} | {delta_llm} | {delta_raw} |"
        )

    lines.extend(["", "## 2. 逐题胜负差距矩阵 (Pairwise Win/Loss vs `graphrag_kg`)", ""])
    lines.append(
        "| 难度分级 | 对比基线 | 共同答对 | 仅 KG 答对 | 仅基线答对 | 共同答错 | 净胜差距 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        for g in compute_pairwise_gaps(s.details, primary_mode="graphrag_kg"):
            sign = "+" if g["net_gain_questions"] >= 0 else ""
            lines.append(
                f"| `{s.suite_name}` | `{g['baseline']}` | {g['both_correct']} | "
                f"**{g['primary_wins']}** | {g['baseline_wins']} | {g['both_wrong']} | "
                f"**{sign}{g['net_gain_questions']} ({sign}{g['net_gain_pct']:.2f}%)** |"
            )

    lines.extend(["", "## 3. 成本、延迟与稳定性遥测", ""])
    lines.append("| 难度分级 | 模式 | 平均 Prompt Tok | 修复重试 | 空回复数 | 资产链接率 | 平均耗时 |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        for m in s.modes:
            link_pct = (m.asset_linked_count / m.total * 100) if m.total else 0.0
            lines.append(
                f"| `{s.suite_name}` | `{m.mode}` | {m.avg_prompt_tokens:.0f} | "
                f"{m.total_repairs} | {m.empty_responses} | {link_pct:.1f}% | {m.avg_latency_seconds:.2f}s |"
            )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, report_md

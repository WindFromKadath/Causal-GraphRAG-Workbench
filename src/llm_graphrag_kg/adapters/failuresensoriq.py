"""Adapter and evaluation harness for IBM FailureSensorIQ (Apache-2.0).

Supports:
1. Ontology-First structured KG construction from `failuresensoriq.mapped.jsonl`
   (or directly from `failuresensoriq_standard__all.jsonl`):
       Equipment --[HAS_FAILURE_MODE]--> FailureMode (asset-scoped)
       FailureMode --[INSTANCE_OF]--> FailureCategory (cross-asset analogy)
       FailureMode --[MONITORED_BY]--> SignalChannel (normalized via Entity Resolution)
2. Traceable per-asset Markdown document generation with chunk-level provenance.
3. Quantitative MCQA benchmark comparing `llm_only`, `naive_rag`, and `graphrag_kg`.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient, BaseLLMClient
from llm_graphrag_kg.core.schema import Entity, GraphAuditStats, Relation, TextChunk
from llm_graphrag_kg.pipelines.community import CommunityDetector
from llm_graphrag_kg.pipelines.extraction import make_entity_id, make_relation_id
from llm_graphrag_kg.pipelines.retriever import GraphRAGRetriever
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


def normalize_sensor_name(raw_name: str) -> tuple[str, bool]:
    """Normalize a FailureSensorIQ sensor label; returns (canonical_name, was_merged)."""
    cleaned = re.sub(r"\s+", " ", raw_name.strip()).lower()
    canonical = SENSOR_CANONICAL_MAP.get(cleaned, cleaned)
    return canonical, canonical != cleaned


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
    relevancy: str
    question_type: str
    gold_option: str
    predicted_option: str
    is_correct: bool
    raw_response: str
    seed_entities: list[str] = Field(default_factory=list)
    expanded_nodes: int = 0


class ModeMetrics(BaseModel):
    """Aggregated accuracy metrics for one evaluation mode."""

    mode: str
    total: int
    correct: int
    accuracy: float
    by_relevancy: dict[str, dict[str, float | int]] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    """Complete side-by-side benchmark report."""

    total_questions: int
    seed: int
    llm_provider: str
    llm_model: str
    modes: list[ModeMetrics]
    details: dict[str, list[EvalQuestionOutcome]] = Field(default_factory=dict)


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

    # Peek first line to detect mapped vs standard QA JSONL format automatically
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

        # Build structured per-asset Markdown document while tracking exact char spans per FM chunk
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

            # Upsert asset-scoped FailureMode entity (prevents cross-asset sensor contamination)
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

            # Upsert canonical FailureCategory for cross-asset multi-hop analogy
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

            # Edge: Equipment --[HAS_FAILURE_MODE]--> FailureMode
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

            # Edge: FailureMode --[INSTANCE_OF]--> FailureCategory
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

            # Upsert normalized SignalChannel nodes & MONITORED_BY edges
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

        # Upsert Equipment node after collecting its chunk provenance
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

    # Embed chunks and run Louvain community detection
    if chunks:
        vectors = active_embedder.embed([c.text for c in chunks])
        vector_store.add_chunks(chunks, vectors)

    detector = CommunityDetector(
        resolution=config.community.resolution,
        seed=config.community.seed,
        summarize_with_llm=summarize_communities,
    )
    detector.detect_and_summarize(graph_store, llm=active_llm)

    # Persist storage artifacts
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
    return "\n".join(f"{oid}. {opt}" for oid, opt in zip(option_ids, options, strict=False))


def _parse_option_letter(raw_response: str, valid_ids: list[str]) -> str:
    """Extract the selected choice letter (e.g., A-E) from an LLM response."""
    text = raw_response.strip()
    valid_set = {v.upper() for v in valid_ids}

    # 1. Explicit "Answer: X" or JSON {"answer": "X"}
    for pattern in (
        r'"answer"\s*:\s*"([A-J])"',
        r"(?:final\s+)?answer\s*[:：]\s*\*{0,2}([A-J])\b",
        r"^\s*\*{0,2}([A-J])\*{0,2}\s*(?:[\.\):：]|$)",
    ):
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).upper() in valid_set:
            return m.group(1).upper()

    # 2. Fallback: first standalone valid option letter
    for token in re.findall(r"\b([A-J])\b", text):
        if token.upper() in valid_set:
            return token.upper()
    return ""


def _build_graphrag_mcqa_context(
    row: dict[str, Any],
    graph_store: Any,
    vector_store: Any,
    embedder: BaseEmbeddingClient,
) -> tuple[str, list[str], int]:
    """Retrieve structured asset-exact KG subgraph + provenance chunks for an MCQA question."""
    asset = re.sub(r"\s+", " ", str(row.get("asset_name", "")).strip()).lower()
    question = str(row.get("question", ""))
    options = [str(o) for o in row.get("options", [])]

    asset_eid = make_entity_id(f"equipment::{asset}")
    seed_ids: list[str] = []
    if graph_store.get_entity(asset_eid) is not None:
        seed_ids.append(asset_eid)

    # Match failure modes & sensors mentioned in question or options
    q_and_opts = f"{question} {' '.join(options)}".lower()
    all_entities = graph_store.get_all_entities()
    for ent in all_entities:
        if ent.entity_type == "FailureMode" and ent.metadata.get("asset") == asset:
            fm_raw = str(ent.metadata.get("failure_category", "")).lower()
            if fm_raw and fm_raw in q_and_opts:
                seed_ids.append(ent.entity_id)
        elif ent.entity_type == "SignalChannel":
            names = [ent.name.lower(), *(a.lower() for a in ent.aliases)]
            if any(n and n in q_and_opts for n in names):
                seed_ids.append(ent.entity_id)

    seed_ids = list(dict.fromkeys(seed_ids))
    # Expand 1-hop around asset-scoped FailureModes (and asset equipment) for exact ISO facts
    expanded_entities, sub_relations = graph_store.get_k_hop_subgraph(seed_ids, k_hops=1)

    # Filter facts strictly to the target asset to prevent cross-asset noise
    ent_map = {e.entity_id: e for e in expanded_entities}
    asset_fm_ids = {
        e.entity_id
        for e in expanded_entities
        if e.entity_type == "FailureMode" and e.metadata.get("asset") == asset
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
            f"- Asset `{asset}` | FailureMode `{fm_label}` --[MONITORED_BY]--> [{sensors_joined}]"
        )

    # Also add vector-matched asset chunks as grounding provenance
    q_vec = embedder.embed([f"{asset} {question}"])[0]
    chunk_hits = [
        chunk
        for chunk, _ in vector_store.search(q_vec, top_k=6)
        if chunk.metadata.get("asset") == asset
    ][:3]

    sections = [
        f"Structured Knowledge Graph Facts for Asset `{asset}` (ISO Complete Monitored-By List):\n"
        + ("\n".join(fact_lines) if fact_lines else "- (no asset triples matched)")
    ]
    if chunk_hits:
        sections.append(
            "Source Provenance Chunks:\n"
            + "\n".join(f"- [{c.chunk_id}] {c.text}" for c in chunk_hits)
        )

    seed_names = [ent_map[eid].name for eid in seed_ids if eid in ent_map]
    return "\n\n".join(sections), seed_names, len(expanded_entities)


def evaluate_failuresensoriq(
    questions_path: Path,
    config: AppConfig,
    *,
    storage_dir: Path | None = None,
    output_dir: Path | None = None,
    limit: int = 20,
    seed: int = 42,
    modes: list[str] | None = None,
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
    if any(m in {"naive_rag", "graphrag_kg"} for m in selected_modes):
        graph_store.load(src_storage)
        vector_store.load(src_storage)

    active_llm = llm or create_llm_client(config)
    active_embedder = embedder or create_embedding_client(config)

    # Load all questions and perform stratified balanced sampling across relevancy types & assets
    rows: list[dict[str, Any]] = []
    with questions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))

    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = f"{row.get('relevancy', 'unknown')}|{row.get('asset_name', 'unknown')}"
        by_bucket[bucket].append(row)

    sampled: list[dict[str, Any]] = []
    buckets = sorted(by_bucket.keys())
    rng.shuffle(buckets)
    for b in buckets:
        rng.shuffle(by_bucket[b])

    round_idx = 0
    while len(sampled) < min(limit, len(rows)):
        progressed = False
        for b in buckets:
            if round_idx < len(by_bucket[b]):
                sampled.append(by_bucket[b][round_idx])
                progressed = True
                if len(sampled) >= min(limit, len(rows)):
                    break
        if not progressed:
            break
        round_idx += 1

    mcqa_system = (
        "You are an industrial reliability & fault diagnosis expert answering a single-choice question. "
        "Reason concisely and end your response with `Answer: <LETTER>` where <LETTER> is one of the option IDs."
    )

    details: dict[str, list[EvalQuestionOutcome]] = {m: [] for m in selected_modes}
    mode_metrics: list[ModeMetrics] = []

    for mode in selected_modes:
        rel_totals: dict[str, int] = defaultdict(int)
        rel_correct: dict[str, int] = defaultdict(int)

        for row in sampled:
            options = [str(o) for o in row["options"]]
            option_ids = [str(oid) for oid in row["option_ids"]]
            correct_flags = [bool(c) for c in row["correct"]]
            gold_letter = next(
                (oid for oid, ok in zip(option_ids, correct_flags, strict=False) if ok), ""
            )
            options_block = _format_mcqa_options(options, option_ids)
            question = str(row["question"])
            asset = str(row.get("asset_name", ""))
            relevancy = str(row.get("relevancy", ""))

            seed_names: list[str] = []
            expanded_nodes = 0

            if mode == "llm_only":
                prompt = (
                    f"Question: {question}\n\nOptions:\n{options_block}\n\n"
                    "Select the single best option. End with `Answer: <LETTER>`."
                )
            elif mode == "naive_rag":
                q_vec = active_embedder.embed([f"{asset} {question}"])[0]
                hits = vector_store.search(q_vec, top_k=4)
                ctx = "\n".join(f"- [{c.chunk_id}] {c.text}" for c, _ in hits)
                prompt = (
                    f"Retrieved Text Chunks:\n{ctx}\n\n"
                    f"Question: {question}\n\nOptions:\n{options_block}\n\n"
                    "Use the retrieved chunks to pick the single best option. End with `Answer: <LETTER>`."
                )
            elif mode == "graphrag_kg":
                kg_ctx, seed_names, expanded_nodes = _build_graphrag_mcqa_context(
                    row, graph_store, vector_store, active_embedder
                )
                prompt = (
                    f"{kg_ctx}\n\n"
                    f"Question: {question}\n\nOptions:\n{options_block}\n\n"
                    "Note: The KG list above enumerates the ISO-relevant sensors for each listed failure mode "
                    "on this asset. If the question asks for an IRRELEVANT / NOT-relevant option, choose the "
                    "option NOT linked in the KG. End strictly with `Answer: <LETTER>`."
                )
            else:
                # Generic fallback via GraphRAGRetriever
                retriever = GraphRAGRetriever(
                    llm=active_llm,
                    embedder=active_embedder,
                    graph_store=graph_store,
                    vector_store=vector_store,
                )
                qres = retriever.query(f"{question}\n{options_block}", mode="hybrid")
                raw_resp = qres.answer
                pred = _parse_option_letter(raw_resp, option_ids)
                is_ok = pred == gold_letter
                rel_totals[relevancy] += 1
                if is_ok:
                    rel_correct[relevancy] += 1
                details[mode].append(
                    EvalQuestionOutcome(
                        question_id=int(row["id"]),
                        asset=asset,
                        relevancy=relevancy,
                        question_type=str(row.get("question_type", "")),
                        gold_option=gold_letter,
                        predicted_option=pred,
                        is_correct=is_ok,
                        raw_response=raw_resp[:400],
                    )
                )
                continue

            raw_resp = active_llm.complete(
                prompt,
                system_prompt=mcqa_system,
                temperature=0.0,
                json_mode=False,
            )
            pred = _parse_option_letter(raw_resp, option_ids)
            is_ok = pred == gold_letter
            rel_totals[relevancy] += 1
            if is_ok:
                rel_correct[relevancy] += 1

            details[mode].append(
                EvalQuestionOutcome(
                    question_id=int(row["id"]),
                    asset=asset,
                    relevancy=relevancy,
                    question_type=str(row.get("question_type", "")),
                    gold_option=gold_letter,
                    predicted_option=pred,
                    is_correct=is_ok,
                    raw_response=raw_resp[:400],
                    seed_entities=seed_names[:8],
                    expanded_nodes=expanded_nodes,
                )
            )

        total_n = len(details[mode])
        correct_n = sum(1 for item in details[mode] if item.is_correct)
        acc = round(correct_n / total_n, 4) if total_n else 0.0
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
                by_relevancy=by_rel,
            )
        )

    summary = EvaluationSummary(
        total_questions=len(sampled),
        seed=seed,
        llm_provider=config.env.llm_provider,
        llm_model=config.env.llm_model,
        modes=mode_metrics,
        details=details,
    )

    # Export JSON and Markdown benchmark reports
    (out_dir / "eval_results.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    report_path = out_dir / "eval_report.md"
    lines = [
        "# FailureSensorIQ 量化评测报告 (LLM-only vs Naive RAG vs GraphRAG+KG)",
        "",
        f"- **评测样本数**：{summary.total_questions}（分层随机抽样，`seed={summary.seed}`）",
        f"- **模型配置**：`{summary.llm_provider}` / `{summary.llm_model}`",
        (
            "- **评测说明**：对比参数记忆推理（`llm_only`）、纯向量片段召回（`naive_rag`）"
            "与 ISO 知识图谱增强推理（`graphrag_kg`）在工业设备失效模式-传感器关联问答上的准确率。"
        ),
        "",
        "## 1. 总体准确率对比",
        "",
        "| 检索/推理模式 | 正确数 / 总数 | 准确率 (Accuracy) |",
        "|---|---|---|",
    ]
    for m in summary.modes:
        lines.append(f"| `{m.mode}` | {m.correct} / {m.total} | **{m.accuracy * 100:.1f}%** |")

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

"""Industrial OREDA FMEA RDF / Turtle (.ttl) Knowledge Graph importer (Stage 3.3).

Zero-external-dependency Turtle parser tailored for industrial FMEA / OREDA knowledge graphs
such as IndustryAssetEQA's `FMEA_KG.ttl`. Features:
- Full @prefix, `;` predicate list, `,` object list, quoted literal & comment parsing
- Canonical Entity Resolution across duplicate RDF labels (preserving all source `rdf_uris`)
- Automatic semantic typing (`FailureModeCode`, `AssetClass`, `Component`, `FailureConcept`)
- Self-loop & dangling-endpoint hygiene filtering so imported graphs pass SHACL-lite cleanly
- Deterministic provenance `TextChunk` card generation (with `.ttl` line ranges) so every
  entity and relation supports full text retrieval (BM25 + Dense RRF), PPR, and citation audit
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient
from llm_graphrag_kg.core.schema import Entity, GraphAuditStats, Relation, TextChunk
from llm_graphrag_kg.pipelines.community import CommunityDetector
from llm_graphrag_kg.pipelines.extraction import make_entity_id, make_relation_id
from llm_graphrag_kg.pipelines.hipporag import SemanticGraphIndex, build_synonym_edges
from llm_graphrag_kg.providers.factory import (
    create_embedding_client,
    create_graph_store,
    create_vector_store,
)
from llm_graphrag_kg.viz.audit_report import compute_audit_stats

_OREDA_CODE_RE = re.compile(r"^[A-Z]{2,4}$")
_SNAKE_CLASS_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")

_TOKEN_RE = re.compile(
    r"""
    (?P<comment>\#[^\n]*)
    |(?P<string>"(?:\\.|[^"\\])*"(?:@[a-zA-Z\-]+|\^\^[^\s;,\.]+)?)
    |(?P<iri><[^>]*>)
    |(?P<punct>[;,\.])
    |(?P<term>[^\s;,\."<\#]+)
    """,
    flags=re.VERBOSE,
)


class RDFTripleRecord(BaseModel):
    """Single parsed RDF triple with source line number for provenance."""

    subject: str
    predicate: str
    obj: str
    is_literal: bool = False
    line_no: int = 1


class RDFImportReport(BaseModel):
    """Structured telemetry returned after importing an RDF / Turtle knowledge graph."""

    source_file: str
    raw_triples: int
    raw_rdf_nodes: int
    merged_duplicate_labels: int
    skipped_self_loops: int
    skipped_dangling: int
    skipped_isolated: int
    predicate_counts: dict[str, int] = Field(default_factory=dict)
    audit_stats: GraphAuditStats


def _strip_literal_quotes(token: str) -> str:
    # Strip optional @lang or ^^datatype suffix after closing quote
    end_q = token.rfind('"')
    inner = token[1:end_q] if end_q > 0 else token.strip('"')
    return (
        inner.replace('\\"', '"')
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\")
        .strip()
    )


def _expand_qname(token: str, prefixes: dict[str, str]) -> str:
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if token == "a":
        return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    if ":" in token:
        pfx, local = token.split(":", 1)
        if pfx in prefixes:
            return f"{prefixes[pfx]}{local}"
    return token


def _local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    if "/" in uri:
        return uri.rstrip("/").rsplit("/", 1)[-1]
    if ":" in uri:
        return uri.rsplit(":", 1)[-1]
    return uri


def parse_turtle_triples(text: str) -> tuple[dict[str, str], list[RDFTripleRecord]]:
    """Parse standard Turtle (.ttl) triples without external RDF libraries."""
    prefixes: dict[str, str] = {}
    triples: list[RDFTripleRecord] = []

    tokens: list[tuple[str, str, int]] = []
    line_no = 1
    pos = 0
    for m in _TOKEN_RE.finditer(text):
        line_no += text.count("\n", pos, m.start())
        pos = m.end()
        kind = m.lastgroup or "term"
        val = m.group(0)
        if kind == "comment":
            continue
        tokens.append((kind, val, line_no))

    i = 0
    n = len(tokens)
    while i < n:
        kind, val, ln = tokens[i]
        # Handle `@prefix pfx: <iri> .` or `PREFIX pfx: <iri>`
        if kind == "term" and val.lower() in {"@prefix", "prefix"}:
            if i + 2 < n:
                pfx_raw = tokens[i + 1][1]
                iri_raw = tokens[i + 2][1]
                pfx = pfx_raw[:-1] if pfx_raw.endswith(":") else pfx_raw
                iri = iri_raw[1:-1] if iri_raw.startswith("<") and iri_raw.endswith(">") else iri_raw
                prefixes[pfx] = iri
                i += 3
                if i < n and tokens[i][1] == ".":
                    i += 1
                continue
            i += 1
            continue

        if kind == "punct":
            i += 1
            continue

        subj = _expand_qname(val, prefixes)
        i += 1
        while i < n:
            if tokens[i][1] == ".":
                i += 1
                break
            if tokens[i][1] == ";":
                i += 1
                continue
            pred_raw = tokens[i][1]
            pred_ln = tokens[i][2]
            pred = _expand_qname(pred_raw, prefixes)
            i += 1
            while i < n:
                okind, oval, oln = tokens[i]
                if okind == "punct":
                    if oval == ",":
                        i += 1
                        continue
                    break
                if okind == "string":
                    triples.append(
                        RDFTripleRecord(
                            subject=subj,
                            predicate=pred,
                            obj=_strip_literal_quotes(oval),
                            is_literal=True,
                            line_no=oln or pred_ln,
                        )
                    )
                else:
                    triples.append(
                        RDFTripleRecord(
                            subject=subj,
                            predicate=pred,
                            obj=_expand_qname(oval, prefixes),
                            is_literal=False,
                            line_no=oln or pred_ln,
                        )
                    )
                i += 1
                if i < n and tokens[i][1] == ",":
                    i += 1
                    continue
                break
            if i < n and tokens[i][1] == ";":
                i += 1
                continue
            if i < n and tokens[i][1] == ".":
                i += 1
                break

    return prefixes, triples


def _classify_fmea_entity(label: str, raw_rdf_types: set[str], out_preds: set[str], in_preds: set[str]) -> str:
    """Map raw RDF entity types & OREDA labels into clean industrial FMEA ontology types."""
    clean = label.strip()
    if _OREDA_CODE_RE.match(clean):
        return "FailureModeCode"
    if _SNAKE_CLASS_RE.match(clean):
        return "AssetClass"
    if any(p in in_preds for p in ("ASSET", "ASSET_NAME", "ASSET_AFFECTED", "ASSET_EXAMPLE", "EXAMPLE_ASSET")):
        return "AssetClass"
    if any(
        p in in_preds
        for p in ("COMPONENT", "COMPONENT_AFFECTED", "EXAMPLE_COMPONENT")
    ) or "Organization" in raw_rdf_types:
        return "Component"
    if "Entity" in raw_rdf_types and len(clean) <= 6:
        return "FailureModeCode"
    return "FailureConcept"


def _normalize_rdf_predicate(pred_uri: str) -> str:
    local = _local_name(pred_uri).strip()
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", local).strip("_").upper()
    return norm or "RELATED_TO"


def ingest_fmea_rdf_ttl(
    ttl_path: Path,
    config: AppConfig,
    *,
    storage_dir: Path | None = None,
    embedder: BaseEmbeddingClient | None = None,
    drop_isolated: bool = True,
) -> RDFImportReport:
    """Import an industrial OREDA FMEA Turtle (.ttl) file into GraphStore + VectorStore + HippoRAG 2."""
    if not ttl_path.is_file():
        raise FileNotFoundError(f"RDF/Turtle file not found: {ttl_path}")

    out_storage = storage_dir or config.storage_path
    out_storage.mkdir(parents=True, exist_ok=True)

    raw_text = ttl_path.read_text(encoding="utf-8")
    _, triples = parse_turtle_triples(raw_text)

    rdf_type_uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    rdfs_label_uri = "http://www.w3.org/2000/01/rdf-schema#label"

    node_labels: dict[str, str] = {}
    node_types: dict[str, set[str]] = defaultdict(set)
    node_lines: dict[str, list[int]] = defaultdict(list)
    edge_records: list[tuple[str, str, str, int]] = []

    for t in triples:
        node_lines[t.subject].append(t.line_no)
        if t.predicate == rdf_type_uri:
            node_types[t.subject].add(_local_name(t.obj))
        elif t.predicate == rdfs_label_uri and t.is_literal:
            node_labels[t.subject] = t.obj
        elif not t.is_literal:
            pred_norm = _normalize_rdf_predicate(t.predicate)
            edge_records.append((t.subject, pred_norm, t.obj, t.line_no))

    # Collect all referenced URIs
    all_uris = set(node_labels.keys()) | set(node_types.keys())
    for s, _, o, _ in edge_records:
        all_uris.add(s)
        all_uris.add(o)

    # Canonical Entity Resolution by normalized label so duplicate URIs merge cleanly
    uri_to_eid: dict[str, str] = {}
    eid_to_name: dict[str, str] = {}
    eid_to_uris: dict[str, list[str]] = defaultdict(list)
    eid_to_raw_types: dict[str, set[str]] = defaultdict(set)
    eid_to_lines: dict[str, list[int]] = defaultdict(list)

    for uri in sorted(all_uris):
        label = node_labels.get(uri) or _local_name(uri)
        eid = make_entity_id(label)
        uri_to_eid[uri] = eid
        eid_to_name.setdefault(eid, label)
        eid_to_uris[eid].append(uri)
        eid_to_raw_types[eid].update(node_types.get(uri, set()))
        eid_to_lines[eid].extend(node_lines.get(uri, []))

    merged_dups = sum(len(u) - 1 for u in eid_to_uris.values() if len(u) > 1)

    # Track incoming/outgoing predicates per canonical entity for refined OREDA typing
    out_preds: dict[str, set[str]] = defaultdict(set)
    in_preds: dict[str, set[str]] = defaultdict(set)
    canonical_edges: dict[str, dict[str, Any]] = {}
    skipped_self_loops = 0
    skipped_dangling = 0
    pred_counts: dict[str, int] = defaultdict(int)

    for s_uri, pred_norm, o_uri, ln in edge_records:
        s_id = uri_to_eid.get(s_uri)
        t_id = uri_to_eid.get(o_uri)
        if not s_id or not t_id:
            skipped_dangling += 1
            continue
        if s_id == t_id:
            skipped_self_loops += 1
            continue
        out_preds[s_id].add(pred_norm)
        in_preds[t_id].add(pred_norm)
        rid = make_relation_id(s_id, pred_norm, t_id)
        if rid not in canonical_edges:
            canonical_edges[rid] = {
                "relation_id": rid,
                "source_id": s_id,
                "target_id": t_id,
                "relation_type": pred_norm,
                "weight": 1.0,
                "lines": [ln],
            }
            pred_counts[pred_norm] += 1
        else:
            canonical_edges[rid]["weight"] += 1.0
            canonical_edges[rid]["lines"].append(ln)

    # Identify connected entities (degree > 0)
    connected_eids: set[str] = set()
    for ed in canonical_edges.values():
        connected_eids.add(ed["source_id"])
        connected_eids.add(ed["target_id"])

    skipped_isolated = 0
    kept_eids: list[str] = []
    for eid in sorted(eid_to_name.keys()):
        if drop_isolated and eid not in connected_eids:
            skipped_isolated += 1
            continue
        kept_eids.append(eid)

    # Build outgoing relation summary per entity for rich textual provenance chunks
    out_by_eid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    in_by_eid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ed in canonical_edges.values():
        out_by_eid[ed["source_id"]].append(ed)
        in_by_eid[ed["target_id"]].append(ed)

    doc_id = f"doc_rdf_{ttl_path.stem.lower()}"
    chunks: list[TextChunk] = []
    eid_to_chunk_id: dict[str, str] = {}
    entities: list[Entity] = []

    for idx, eid in enumerate(kept_eids):
        name = eid_to_name[eid]
        etype = _classify_fmea_entity(
            name, eid_to_raw_types[eid], out_preds[eid], in_preds[eid]
        )
        cid = f"{doc_id}::chunk_{idx:04d}"
        eid_to_chunk_id[eid] = cid

        lns = sorted(set(eid_to_lines[eid]))
        ln_span = f"L{lns[0]}-L{lns[-1]}" if lns else "L1"
        out_desc_parts: list[str] = []
        grouped_out: dict[str, list[str]] = defaultdict(list)
        for ed in out_by_eid.get(eid, []):
            grouped_out[ed["relation_type"]].append(eid_to_name[ed["target_id"]])
        for p, targets in sorted(grouped_out.items()):
            uniq_t = list(dict.fromkeys(targets))
            out_desc_parts.append(f"{p}: {', '.join(uniq_t[:12])}")

        in_desc_parts: list[str] = []
        grouped_in: dict[str, list[str]] = defaultdict(list)
        for ed in in_by_eid.get(eid, []):
            grouped_in[ed["relation_type"]].append(eid_to_name[ed["source_id"]])
        for p, srcs in sorted(grouped_in.items()):
            uniq_s = list(dict.fromkeys(srcs))
            in_desc_parts.append(f"<-{p}- {', '.join(uniq_s[:10])}")

        chunk_lines = [
            f"[OREDA FMEA RDF Record] {etype}: {name} (RDF source: {ttl_path.name}:{ln_span})",
            f"RDF URIs: {', '.join(_local_name(u) for u in eid_to_uris[eid])}",
        ]
        if out_desc_parts:
            chunk_lines.append("Outgoing FMEA Relations: " + " | ".join(out_desc_parts))
        if in_desc_parts:
            chunk_lines.append("Incoming FMEA Relations: " + " | ".join(in_desc_parts))
        chunk_text = "\n".join(chunk_lines)

        chunks.append(
            TextChunk(
                chunk_id=cid,
                doc_id=doc_id,
                source=ttl_path.name,
                text=chunk_text,
                index=idx,
                start_char=0,
                end_char=len(chunk_text),
            )
        )
        desc = (
            f"OREDA FMEA {etype} `{name}`"
            + (f" ({out_desc_parts[0][:140]})" if out_desc_parts else ".")
        )
        entities.append(
            Entity(
                entity_id=eid,
                name=name,
                entity_type=etype,
                description=desc,
                source_chunk_ids=[cid],
                metadata={
                    "rdf_uris": eid_to_uris[eid],
                    "raw_rdf_types": sorted(eid_to_raw_types[eid]),
                    "line_span": ln_span,
                },
            )
        )

    relations: list[Relation] = []
    kept_set = set(kept_eids)
    for ed in canonical_edges.values():
        s_id = ed["source_id"]
        t_id = ed["target_id"]
        if s_id not in kept_set or t_id not in kept_set:
            continue
        rel_cids = list(dict.fromkeys([eid_to_chunk_id[s_id], eid_to_chunk_id[t_id]]))
        relations.append(
            Relation(
                relation_id=ed["relation_id"],
                source_id=s_id,
                target_id=t_id,
                relation_type=ed["relation_type"],
                description=(
                    f"{eid_to_name[s_id]} --[{ed['relation_type']}]--> {eid_to_name[t_id]} "
                    f"(from {ttl_path.name}:L{min(ed['lines'])})"
                ),
                weight=float(ed["weight"]),
                source_chunk_ids=rel_cids,
            )
        )

    gs = create_graph_store()
    vs = create_vector_store()
    active_embedder = embedder or create_embedding_client(config)

    if chunks:
        vecs = active_embedder.embed([c.text for c in chunks])
        vs.add_chunks(chunks, vecs)
    for ent in entities:
        gs.upsert_entity(ent)
    for rel in relations:
        gs.upsert_relation(rel)

    # HippoRAG 2 SYNONYM_OF + SemanticGraphIndex
    synonym_edges_added = 0
    precomputed_entities = None
    if config.retrieval.enable_synonym_edges and len(entities) >= 2:
        synonym_edges_added, ent_ids, ent_vecs = build_synonym_edges(
            gs,
            active_embedder,
            cosine_threshold=config.retrieval.synonym_cosine_threshold,
            max_synonyms_per_entity=config.retrieval.max_synonyms_per_entity,
        )
        precomputed_entities = (ent_ids, ent_vecs)

    sem_idx = SemanticGraphIndex()
    sem_stats = sem_idx.build(
        gs,
        active_embedder,
        precomputed_entities=precomputed_entities,
        max_indexed_triples=config.retrieval.max_indexed_triples,
    )

    detector = CommunityDetector(
        resolution=config.community.resolution,
        seed=config.community.seed,
        summarize_with_llm=False,
    )
    detector.detect_and_summarize(gs, llm=None)

    gs.save(out_storage)
    vs.save(out_storage)
    sem_idx.save(out_storage)
    manifest = {
        "doc_count": 1,
        "chunk_count": len(chunks),
        "embedding_provider": config.env.embedding_provider,
        "embedding_model": config.env.embedding_model,
        "source_format": "rdf_turtle",
        "source_file": ttl_path.name,
        "synonym_edges_added": synonym_edges_added,
        "semantic_index": sem_stats,
        "documents": [
            {"doc_id": doc_id, "source": ttl_path.name, "title": f"OREDA FMEA KG ({ttl_path.name})"}
        ],
    }
    (out_storage / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = compute_audit_stats(gs, vs, doc_count=1)
    return RDFImportReport(
        source_file=ttl_path.name,
        raw_triples=len(triples),
        raw_rdf_nodes=len(all_uris),
        merged_duplicate_labels=merged_dups,
        skipped_self_loops=skipped_self_loops,
        skipped_dangling=skipped_dangling,
        skipped_isolated=skipped_isolated,
        predicate_counts=dict(sorted(pred_counts.items(), key=lambda x: -x[1])),
        audit_stats=stats,
    )

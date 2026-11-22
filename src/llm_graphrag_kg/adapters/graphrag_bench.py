"""Adapter & multi-paradigm evaluator for GraphRAG-Bench (ICLR, medical & novel multi-hop QA)."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient, BaseLLMClient
from llm_graphrag_kg.core.schema import Entity, GraphAuditStats, Relation
from llm_graphrag_kg.pipelines.community import CommunityDetector
from llm_graphrag_kg.pipelines.extraction import make_entity_id, make_relation_id
from llm_graphrag_kg.pipelines.ingestion import DocumentIngestor
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.factory import (
    create_embedding_client,
    create_graph_store,
    create_llm_client,
    create_vector_store,
)
from llm_graphrag_kg.viz.audit_report import compute_audit_stats

STOPWORDS = {
    "is", "in", "of", "to", "on", "at", "by", "as", "an", "or", "be", "it",
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "have", "has",
    "had", "can", "may", "will", "should", "could", "would", "not", "but", "about", "into",
    "your", "their", "they", "them", "what", "when", "where", "which", "who", "why", "how",
    "also", "most", "more", "some", "such", "than", "then", "there", "these", "those", "been",
    "being", "other", "only", "both", "each", "very", "used", "often", "usually", "many",
}


def _norm_tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2 and t not in STOPWORDS
    ]


def compute_token_f1(prediction: str, reference: str) -> float:
    pred_toks = _norm_tokens(prediction)
    ref_toks = _norm_tokens(reference)
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_toks)
    rec = num_same / len(ref_toks)
    return round(2 * prec * rec / (prec + rec), 4)


def compute_rouge_l_f1(prediction: str, reference: str) -> float:
    p = _norm_tokens(prediction)[:200]
    r = _norm_tokens(reference)[:200]
    if not p or not r:
        return 0.0
    m, n = len(p), len(r)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        pi = p[i - 1]
        for j in range(1, n + 1):
            if pi == r[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] >= prev[j] else prev[j]
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return round(2 * prec * rec / (prec + rec), 4)


def compute_evidence_coverage(prediction: str, evidence_text: str) -> float:
    ref_terms = set(_norm_tokens(evidence_text))
    if not ref_terms:
        return 0.0
    pred_terms = set(_norm_tokens(prediction))
    return round(len(ref_terms & pred_terms) / len(ref_terms), 4)


class GRBQuestionOutcome(BaseModel):
    question_id: str
    question_type: str
    mode: str
    token_f1: float
    rouge_l: float
    evidence_coverage: float
    grounding_score: float | None = None
    tot_branches: int = 0
    tot_pruned: int = 0
    llm_calls: int = 1
    latency_seconds: float
    prediction: str


class GRBModeSummary(BaseModel):
    mode: str
    total: int
    avg_token_f1: float
    avg_rouge_l: float
    avg_evidence_coverage: float
    avg_grounding_score: float | None = None
    avg_llm_calls: float
    avg_latency_seconds: float


class GRBEvalReport(BaseModel):
    question_type_filter: str
    total_questions: int
    llm_model: str
    modes: list[GRBModeSummary] = Field(default_factory=list)


def _extract_lexical_medical_kg(
    chunks: list[Any],
) -> tuple[list[Entity], list[Relation]]:
    """Zero-LLM deterministic domain phrase & co-occurrence KG builder for rapid large-corpus indexing."""
    # Extract recurring multi-word clinical noun phrases / acronyms directly from corpus text
    phrase_pattern = re.compile(
        r"\b(?:[A-Z]{2,6}|[a-z]+(?:\s+[a-z]+){1,2})\b"
    )
    domain_anchors = (
        "cancer", "carcinoma", "lymphoma", "leukemia", "tumor", "biopsy", "surgery", "radiation",
        "therapy", "treatment", "risk", "factor", "skin", "cell", "basal", "immune", "transplant",
        "sun", "uv", "exam", "imaging", "scan", "recurrence", "lymph", "node", "chemotherapy",
        "diagnosis", "symptom", "bump", "lesion", "margin", "mohs", "follow-up", "screening",
        "mutation", "gene", "blood", "bone", "kidney", "prostate", "metastasis", "staging",
        "bcc", "aml", "all", "rcc", "pet", "ct", "mri", "uvb", "uva",
    )
    term_counts: Counter[str] = Counter()
    chunk_terms: dict[str, list[str]] = {}
    for ch in chunks:
        found_in_ch: list[str] = []
        for m in phrase_pattern.finditer(ch.text):
            raw = m.group(0).strip()
            rl = raw.lower()
            if len(rl) < 3 or rl in STOPWORDS:
                continue
            if any(anc in rl for anc in domain_anchors):
                words = [w for w in rl.split() if w not in STOPWORDS]
                if not words:
                    continue
                norm = " ".join(words)
                if len(norm) >= 3:
                    term_counts[norm] += 1
                    found_in_ch.append(norm)
        # Keep top unique domain phrases per chunk
        chunk_terms[ch.chunk_id] = list(dict.fromkeys(found_in_ch))[:12]

    # Filter to recurring terms (freq >= 2) to avoid noise
    kept_terms = {t for t, c in term_counts.most_common(220) if c >= 2}
    entities_by_id: dict[str, Entity] = {}
    relations_by_id: dict[str, Relation] = {}

    for ch in chunks:
        valid = [t for t in chunk_terms[ch.chunk_id] if t in kept_terms][:8]
        eids: list[str] = []
        for term in valid:
            eid = make_entity_id(term)
            eids.append(eid)
            if eid not in entities_by_id:
                etype = (
                    "Treatment"
                    if any(k in term for k in ("surgery", "therapy", "treatment", "mohs", "chemo"))
                    else (
                        "DiagnosticTest"
                        if any(k in term for k in ("biopsy", "exam", "imaging", "scan", "ct", "mri"))
                        else (
                            "RiskFactor"
                            if any(k in term for k in ("risk", "uv", "sun", "transplant", "immune"))
                            else "DiseaseConcept"
                        )
                    )
                )
                entities_by_id[eid] = Entity(
                    entity_id=eid,
                    name=term,
                    entity_type=etype,
                    description=f"Corpus medical term `{term}` extracted from GraphRAG-Bench.",
                    source_chunk_ids=[ch.chunk_id],
                )
            elif ch.chunk_id not in entities_by_id[eid].source_chunk_ids:
                entities_by_id[eid].source_chunk_ids.append(ch.chunk_id)

        # Connect co-occurring concepts within the same chunk (star + adjacent pairs)
        for i in range(len(eids)):
            for j in range(i + 1, min(len(eids), i + 4)):
                s_id, t_id = sorted((eids[i], eids[j]))
                if s_id == t_id:
                    continue
                rid = make_relation_id(s_id, "CO_OCCURS_WITH", t_id)
                if rid not in relations_by_id:
                    relations_by_id[rid] = Relation(
                        relation_id=rid,
                        source_id=s_id,
                        target_id=t_id,
                        relation_type="CO_OCCURS_WITH",
                        description=f"Co-occurs in medical guideline chunk {ch.chunk_id}.",
                        weight=1.0,
                        source_chunk_ids=[ch.chunk_id],
                    )
                else:
                    relations_by_id[rid].weight += 1.0
                    if ch.chunk_id not in relations_by_id[rid].source_chunk_ids:
                        relations_by_id[rid].source_chunk_ids.append(ch.chunk_id)

    return list(entities_by_id.values()), list(relations_by_id.values())


def ingest_graphrag_bench_corpus(
    corpus_path: Path,
    config: AppConfig,
    *,
    docs_dir: Path | None = None,
    storage_dir: Path | None = None,
    max_chars: int = 120_000,
    fast_lexical: bool = True,
    llm: BaseLLMClient | None = None,
    embedder: BaseEmbeddingClient | None = None,
) -> GraphAuditStats:
    """Ingest GraphRAG-Bench Corpus JSON strictly from corpus text (zero question leakage)."""
    if not corpus_path.is_file():
        raise FileNotFoundError(f"GraphRAG-Bench corpus not found: {corpus_path}")

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    out_docs = docs_dir or config.input_path
    out_storage = storage_dir or config.storage_path
    out_docs.mkdir(parents=True, exist_ok=True)
    out_storage.mkdir(parents=True, exist_ok=True)

    # Clear previous generated md files in out_docs
    for old_md in out_docs.glob("grb_*.md"):
        old_md.unlink()

    if isinstance(raw, dict) and "context" in raw:
        full_text = str(raw["context"])
        if max_chars > 0:
            full_text = full_text[:max_chars]
        # Partition into clean ~20K char guideline sections
        step = 20_000
        for idx, start in enumerate(range(0, len(full_text), step)):
            seg = full_text[start : start + step]
            (out_docs / f"grb_medical_{idx:03d}.md").write_text(
                f"# GraphRAG-Bench Medical Section {idx}\n\n{seg}\n", encoding="utf-8"
            )
    elif isinstance(raw, list):
        budget = max_chars if max_chars > 0 else 10**9
        used = 0
        for idx, item in enumerate(raw):
            txt = str(item.get("context", item.get("text", "")))
            if not txt:
                continue
            take = txt[: max(0, budget - used)]
            if not take:
                break
            (out_docs / f"grb_doc_{idx:03d}.md").write_text(
                f"# GraphRAG-Bench Document {idx}\n\n{take}\n", encoding="utf-8"
            )
            used += len(take)
            if used >= budget:
                break

    if not fast_lexical:
        pipeline = GraphRAGPipeline(config, llm=llm, embedder=embedder)
        return pipeline.build(out_docs, storage_dir=out_storage)

    # Fast deterministic lexical KG build + real vector store + Louvain communities
    ingestor = DocumentIngestor(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
    documents = ingestor.load_directory(out_docs, base_dir=config.project_root)
    chunks = ingestor.chunk_documents(documents)

    gs = create_graph_store()
    vs = create_vector_store()
    active_embedder = embedder or create_embedding_client(config)
    if chunks:
        vecs = active_embedder.embed([c.text for c in chunks])
        vs.add_chunks(chunks, vecs)

    entities, relations = _extract_lexical_medical_kg(chunks)
    for e in entities:
        gs.upsert_entity(e)
    for r in relations:
        gs.upsert_relation(r)

    detector = CommunityDetector(
        resolution=config.community.resolution,
        seed=config.community.seed,
        summarize_with_llm=False,
    )
    detector.detect_and_summarize(gs, llm=None)

    gs.save(out_storage)
    vs.save(out_storage)
    manifest = {
        "doc_count": len(documents),
        "chunk_count": len(chunks),
        "documents": [
            {"doc_id": d.doc_id, "source": d.source, "title": d.title} for d in documents
        ],
    }
    (out_storage / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return compute_audit_stats(gs, vs, doc_count=len(documents))


def evaluate_graphrag_bench(
    questions_path: Path,
    config: AppConfig,
    *,
    storage_dir: Path | None = None,
    output_dir: Path | None = None,
    question_type: str = "Complex Reasoning",
    limit: int = 10,
    modes: list[str] | None = None,
    llm: BaseLLMClient | None = None,
    embedder: BaseEmbeddingClient | None = None,
) -> tuple[GRBEvalReport, Path]:
    """Compare Naive RAG vs Hybrid Direct vs PPR+CoT vs PPR+ToT on GraphRAG-Bench."""
    if not questions_path.is_file():
        raise FileNotFoundError(f"GraphRAG-Bench questions file not found: {questions_path}")

    selected_modes = modes or ["naive_rag", "hybrid_direct", "ppr_cot", "ppr_tot"]
    src_storage = storage_dir or config.storage_path
    out_dir = output_dir or config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    active_llm = llm or create_llm_client(config)
    active_embedder = embedder or create_embedding_client(config)
    pipeline = GraphRAGPipeline(config, llm=active_llm, embedder=active_embedder)
    pipeline.load(src_storage)

    all_qs: list[dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))
    if question_type.strip().upper() != "ALL":
        filtered = [q for q in all_qs if q.get("question_type") == question_type]
    else:
        filtered = all_qs
    sampled = filtered[:limit] if limit > 0 else filtered

    summaries: list[GRBModeSummary] = []
    all_outcomes: dict[str, list[GRBQuestionOutcome]] = {}

    for mode in selected_modes:
        outcomes: list[GRBQuestionOutcome] = []
        for q in sampled:
            qid = str(q.get("id", ""))
            qtext = str(q["question"])
            gold_ans = str(q.get("answer", ""))
            gold_ev = str(q.get("evidence", ""))
            t0 = time.perf_counter()

            g_score: float | None = None
            tot_branches = 0
            tot_pruned = 0
            llm_calls = 1

            if mode == "naive_rag":
                q_vec = active_embedder.embed([qtext])[0]
                hits = pipeline.vector_store.search(q_vec, top_k=config.retrieval.top_k_chunks)
                ctx = "\n".join(f"- [{c.chunk_id}] {c.text}" for c, _ in hits)
                prompt = (
                    f"Retrieved Text Chunks:\n{ctx}\n\n"
                    f"Question: {qtext}\n\n"
                    "Answer concisely and accurately based ONLY on the retrieved chunks."
                )
                pred = active_llm.complete(prompt, temperature=0.1, json_mode=False)
            elif mode == "hybrid_direct":
                res = pipeline.query(
                    qtext, mode="hybrid", reasoning="direct", auto_load=False, enable_compact=True
                )
                pred = res.answer
                ledger = res.metadata.get("harness_ledger", {})
                if isinstance(ledger, dict):
                    llm_calls = int(ledger.get("total_calls", 1))
            elif mode == "ppr_cot":
                res = pipeline.query(
                    qtext, mode="ppr", reasoning="cot", auto_load=False, enable_compact=True
                )
                pred = res.answer
                rm = res.metadata.get("reasoning", {})
                if isinstance(rm, dict):
                    ga = rm.get("grounding_audit", {})
                    if isinstance(ga, dict):
                        g_score = float(ga.get("grounding_score", 0.0))
                ledger = res.metadata.get("harness_ledger", {})
                if isinstance(ledger, dict):
                    llm_calls = int(ledger.get("total_calls", 1))
            elif mode == "ppr_tot":
                res = pipeline.query(
                    qtext, mode="ppr", reasoning="tot", auto_load=False, enable_compact=True
                )
                pred = res.answer
                rm = res.metadata.get("reasoning", {})
                if isinstance(rm, dict):
                    ga = rm.get("grounding_audit", {})
                    if isinstance(ga, dict):
                        g_score = float(ga.get("grounding_score", 0.0))
                    branches = rm.get("branches", [])
                    if isinstance(branches, list):
                        tot_branches = len(branches)
                        tot_pruned = sum(1 for b in branches if b.get("status") == "pruned")
                ledger = res.metadata.get("harness_ledger", {})
                if isinstance(ledger, dict):
                    llm_calls = int(ledger.get("total_calls", 1))
            else:
                raise ValueError(f"Unsupported GraphRAG-Bench eval mode: {mode!r}")

            dt = round(time.perf_counter() - t0, 3)
            outcomes.append(
                GRBQuestionOutcome(
                    question_id=qid,
                    question_type=str(q.get("question_type", "")),
                    mode=mode,
                    token_f1=compute_token_f1(pred, gold_ans),
                    rouge_l=compute_rouge_l_f1(pred, gold_ans),
                    evidence_coverage=compute_evidence_coverage(pred, gold_ev or gold_ans),
                    grounding_score=g_score,
                    tot_branches=tot_branches,
                    tot_pruned=tot_pruned,
                    llm_calls=llm_calls,
                    latency_seconds=dt,
                    prediction=pred[:500],
                )
            )

        all_outcomes[mode] = outcomes
        n = max(len(outcomes), 1)
        g_vals = [o.grounding_score for o in outcomes if o.grounding_score is not None]
        summaries.append(
            GRBModeSummary(
                mode=mode,
                total=len(outcomes),
                avg_token_f1=round(sum(o.token_f1 for o in outcomes) / n, 4),
                avg_rouge_l=round(sum(o.rouge_l for o in outcomes) / n, 4),
                avg_evidence_coverage=round(sum(o.evidence_coverage for o in outcomes) / n, 4),
                avg_grounding_score=round(sum(g_vals) / len(g_vals), 4) if g_vals else None,
                avg_llm_calls=round(sum(o.llm_calls for o in outcomes) / n, 2),
                avg_latency_seconds=round(sum(o.latency_seconds for o in outcomes) / n, 2),
            )
        )

    report = GRBEvalReport(
        question_type_filter=question_type,
        total_questions=len(sampled),
        llm_model=config.env.llm_model,
        modes=summaries,
    )
    (out_dir / "grb_eval_results.json").write_text(
        json.dumps(
            {
                "report": report.model_dump(),
                "outcomes": {k: [o.model_dump() for o in v] for k, v in all_outcomes.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md_lines = [
        f"# GraphRAG-Bench 多范式评测报告（`{question_type}`）",
        "",
        f"- **题数**：`{len(sampled)}` | **模型**：`{config.env.llm_model}`",
        "",
        "| 范式模式 | Token-F1 | ROUGE-L | 证据覆盖率 | Grounding Score | 平均调用 | 平均耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        g_str = f"{s.avg_grounding_score * 100:.1f}%" if s.avg_grounding_score is not None else "-"
        md_lines.append(
            f"| `{s.mode}` | **{s.avg_token_f1 * 100:.1f}%** | {s.avg_rouge_l * 100:.1f}% | "
            f"{s.avg_evidence_coverage * 100:.1f}% | {g_str} | {s.avg_llm_calls:.1f} | "
            f"{s.avg_latency_seconds:.2f}s |"
        )
    report_md = out_dir / "grb_eval_report.md"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report, report_md

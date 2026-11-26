"""Adapter & multi-paradigm evaluator for GraphRAG-Bench (ICLR, medical & novel multi-hop QA)."""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient, BaseLLMClient
from llm_graphrag_kg.core.schema import Entity, GraphAuditStats, Relation
from llm_graphrag_kg.pipelines.community import CommunityDetector
from llm_graphrag_kg.pipelines.extraction import (
    SemanticOpenIEExtractor,
    make_entity_id,
    make_relation_id,
)
from llm_graphrag_kg.pipelines.hipporag import SemanticGraphIndex, build_synonym_edges
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
    source: str = ""
    question_type: str
    mode: str
    token_f1: float
    rouge_l: float
    evidence_coverage: float
    retrieval_evidence_recall: float = 0.0
    grounding_score: float | None = None
    tot_branches: int = 0
    tot_pruned: int = 0
    route_taken: str = ""
    early_stopped: bool = False
    planner_model: str = ""
    planner_seconds: float = 0.0
    load_seconds: float = 0.0
    max_depth_reached: int = 0
    backtrack_count: int = 0
    llm_calls: int = 1
    latency_seconds: float
    prediction: str
    retrieved_context: str = ""


class GRBModeSummary(BaseModel):
    mode: str
    total: int
    avg_token_f1: float
    avg_rouge_l: float
    avg_evidence_coverage: float
    avg_retrieval_evidence_recall: float = 0.0
    avg_grounding_score: float | None = None
    avg_llm_calls: float
    avg_latency_seconds: float


class GRBEvalReport(BaseModel):
    question_type_filter: str
    source_filter: str = "ALL"
    total_questions: int
    llm_model: str
    embedding_model: str = ""
    modes: list[GRBModeSummary] = Field(default_factory=list)


def _extract_lexical_kg(
    chunks: list[Any],
    *,
    domain: str = "medical",
) -> tuple[list[Entity], list[Relation]]:
    """Zero-LLM deterministic entity & co-occurrence KG builder (supports medical & novel)."""
    medical_anchors = (
        "cancer", "carcinoma", "lymphoma", "leukemia", "tumor", "biopsy", "surgery", "radiation",
        "therapy", "treatment", "risk", "factor", "skin", "cell", "basal", "immune", "transplant",
        "sun", "uv", "exam", "imaging", "scan", "recurrence", "lymph", "node", "chemotherapy",
        "diagnosis", "symptom", "bump", "lesion", "margin", "mohs", "follow-up", "screening",
        "mutation", "gene", "blood", "bone", "kidney", "prostate", "metastasis", "staging",
        "bcc", "aml", "all", "rcc", "pet", "ct", "mri", "uvb", "uva", "hodgkin", "marrow",
    )
    med_phrase_pat = re.compile(r"\b(?:[A-Z]{2,6}|[a-z]+(?:\s+[a-z]+){1,2})\b")
    proper_noun_pat = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+(?:of|the|de|da)?\s*[A-Z][a-z]{2,}){0,2}\b")

    term_counts: Counter[str] = Counter()
    chunk_terms: dict[str, list[str]] = {}

    for ch in chunks:
        found_in_ch: list[str] = []
        if domain == "medical":
            for m in med_phrase_pat.finditer(ch.text):
                rl = m.group(0).strip().lower()
                if len(rl) < 3 or rl in STOPWORDS:
                    continue
                if any(anc in rl for anc in medical_anchors):
                    words = [w for w in rl.split() if w not in STOPWORDS]
                    if not words:
                        continue
                    norm = " ".join(words)
                    if len(norm) >= 3:
                        term_counts[norm] += 1
                        found_in_ch.append(norm)
        else:
            # Novel / general prose: extract recurring capitalized entities (characters, places, events)
            for m in proper_noun_pat.finditer(ch.text):
                raw = re.sub(r"\s+", " ", m.group(0).strip())
                rl = raw.lower()
                if len(rl) < 3 or rl in STOPWORDS or rl in {
                    "the", "and", "but", "for", "with", "from", "that", "this", "what", "when",
                    "where", "which", "chapter", "book", "part", "graphrag", "section",
                }:
                    continue
                term_counts[rl] += 1
                found_in_ch.append(rl)

        chunk_terms[ch.chunk_id] = list(dict.fromkeys(found_in_ch))[:12]

    max_vocab = 450 if len(chunks) > 300 else 240
    kept_terms = {t for t, c in term_counts.most_common(max_vocab) if c >= 2}
    entities_by_id: dict[str, Entity] = {}
    relations_by_id: dict[str, Relation] = {}

    for ch in chunks:
        valid = [t for t in chunk_terms[ch.chunk_id] if t in kept_terms][:8]
        eids: list[str] = []
        for term in valid:
            eid = make_entity_id(term)
            eids.append(eid)
            if eid not in entities_by_id:
                if domain == "medical":
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
                else:
                    etype = "NarrativeEntity"
                entities_by_id[eid] = Entity(
                    entity_id=eid,
                    name=term,
                    entity_type=etype,
                    description=f"Corpus entity `{term}` extracted from GraphRAG-Bench ({domain}).",
                    source_chunk_ids=[ch.chunk_id],
                )
            elif ch.chunk_id not in entities_by_id[eid].source_chunk_ids:
                entities_by_id[eid].source_chunk_ids.append(ch.chunk_id)

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
                        description=f"Co-occurs in chunk {ch.chunk_id}.",
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
    max_chars: int = 0,
    source_filter: str | None = None,
    fast_lexical: bool = True,
    use_openie: bool = False,
    openie_max_chunks: int = 0,
    openie_resume: bool = True,
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

    for old_md in out_docs.glob("grb_*.md"):
        old_md.unlink()

    domain = "medical" if "medical" in corpus_path.name.lower() else "novel"

    if isinstance(raw, dict) and "context" in raw:
        full_text = str(raw["context"])
        if max_chars > 0:
            full_text = full_text[:max_chars]
        step = 20_000
        for idx, start in enumerate(range(0, len(full_text), step)):
            seg = full_text[start : start + step]
            (out_docs / f"grb_medical_{idx:03d}.md").write_text(
                f"# GraphRAG-Bench Medical Section {idx}\n\n{seg}\n", encoding="utf-8"
            )
    elif isinstance(raw, list):
        wanted_sources = (
            {s.strip() for s in source_filter.split(",") if s.strip()} if source_filter else None
        )
        budget = max_chars if max_chars > 0 else 10**9
        used = 0
        for idx, item in enumerate(raw):
            cname = str(item.get("corpus_name", f"doc_{idx}"))
            if wanted_sources is not None and cname not in wanted_sources:
                continue
            txt = str(item.get("context", item.get("text", "")))
            if not txt:
                continue
            take = txt[: max(0, budget - used)]
            if not take:
                break
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", cname)
            (out_docs / f"grb_{slug}_{idx:03d}.md").write_text(
                f"# GraphRAG-Bench {cname}\n\n{take}\n", encoding="utf-8"
            )
            used += len(take)
            if used >= budget:
                break

    if not fast_lexical and not use_openie:
        pipeline = GraphRAGPipeline(config, llm=llm, embedder=embedder)
        return pipeline.build(out_docs, storage_dir=out_storage)

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

    openie_stats_dict: dict[str, Any] | None = None
    if use_openie:
        active_llm = llm or create_llm_client(config)
        extractor = SemanticOpenIEExtractor(
            active_llm,
            entity_types=config.extraction.entity_types,
            domain=domain,
            temperature=config.extraction.temperature,
            max_entities_per_chunk=config.extraction.max_entities_per_chunk,
        )
        ckpt_file = out_storage / "openie_chunks.checkpoint.jsonl"
        openie_ents, openie_rels, openie_stats = extractor.extract_chunks_checkpointed(
            chunks,
            checkpoint_path=ckpt_file,
            resume=openie_resume,
            max_chunks=openie_max_chunks,
            progress_every=10,
        )
        openie_stats_dict = openie_stats.model_dump()
        for e in openie_ents:
            gs.upsert_entity(e)
        for r in openie_rels:
            gs.upsert_relation(r)
        # If openie_max_chunks capped LLM extraction to a prefix, keep lexical coverage on remainder
        if openie_max_chunks > 0 and openie_max_chunks < len(chunks):
            lex_ents, lex_rels = _extract_lexical_kg(chunks[openie_max_chunks:], domain=domain)
            for e in lex_ents:
                gs.upsert_entity(e)
            for r in lex_rels:
                gs.upsert_relation(r)
    else:
        entities, relations = _extract_lexical_kg(chunks, domain=domain)
        for e in entities:
            gs.upsert_entity(e)
        for r in relations:
            gs.upsert_relation(r)

    # Stage 3.1 HippoRAG 2: SYNONYM_OF auto-linking + Semantic Entity/Triple Index
    synonym_edges_added = 0
    precomputed_entities = None
    if config.retrieval.enable_synonym_edges:
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
        "doc_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_provider": config.env.embedding_provider,
        "embedding_model": config.env.embedding_model,
        "synonym_edges_added": synonym_edges_added,
        "semantic_index": sem_stats,
        "openie_stats": openie_stats_dict,
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
    output_tag: str = "default",
    question_type: str = "Complex Reasoning",
    source_filter: str | None = None,
    limit: int = 10,
    seed: int = 42,
    modes: list[str] | None = None,
    resume: bool = True,
    progress_every: int = 5,
    llm: BaseLLMClient | None = None,
    embedder: BaseEmbeddingClient | None = None,
) -> tuple[GRBEvalReport, Path]:
    """Compare Naive RAG vs Hybrid Direct vs PPR+CoT vs PPR+ToT on GraphRAG-Bench with checkpoints."""
    if not questions_path.is_file():
        raise FileNotFoundError(f"GraphRAG-Bench questions file not found: {questions_path}")

    selected_modes = modes or ["naive_rag", "hybrid_direct", "ppr_cot", "ppr_tot"]
    src_storage = storage_dir or config.storage_path
    out_dir = (output_dir or config.output_path) / output_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    active_llm = llm or create_llm_client(config)
    active_embedder = embedder or create_embedding_client(config)
    pipeline = GraphRAGPipeline(config, llm=active_llm, embedder=active_embedder)
    pipeline.load(src_storage)

    all_qs: list[dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))
    filtered = all_qs
    if question_type.strip().upper() != "ALL":
        wanted_types = {t.strip() for t in question_type.split(",") if t.strip()}
        filtered = [q for q in filtered if q.get("question_type") in wanted_types]
    if source_filter:
        wanted_src = {s.strip() for s in source_filter.split(",") if s.strip()}
        filtered = [q for q in filtered if q.get("source") in wanted_src]

    # Deterministic stratified sampling across (question_type, source) so multi-type evals stay balanced
    if limit > 0 and limit < len(filtered):
        rng = random.Random(seed)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for q in filtered:
            key = f"{q.get('question_type', '')}|{q.get('source', '')}"
            buckets[key].append(q)
        bkeys = sorted(buckets.keys())
        for bk in bkeys:
            rng.shuffle(buckets[bk])
        sampled: list[dict[str, Any]] = []
        r_idx = 0
        while len(sampled) < limit:
            progressed = False
            for bk in bkeys:
                if r_idx < len(buckets[bk]):
                    sampled.append(buckets[bk][r_idx])
                    progressed = True
                    if len(sampled) >= limit:
                        break
            if not progressed:
                break
            r_idx += 1
    else:
        sampled = filtered

    summaries: list[GRBModeSummary] = []
    all_outcomes: dict[str, list[GRBQuestionOutcome]] = {}

    for mode in selected_modes:
        ckpt_path = out_dir / f"grb_{mode}.checkpoint.jsonl"
        cached: dict[str, GRBQuestionOutcome] = {}
        if resume and ckpt_path.is_file():
            with ckpt_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        item = GRBQuestionOutcome(**json.loads(line))
                        cached[item.question_id] = item

        outcomes: list[GRBQuestionOutcome] = []
        mode_t0 = time.perf_counter()
        newly_run = 0
        total_n = len(sampled)

        with ckpt_path.open("a", encoding="utf-8") as ckpt_fh:
            for idx_q, q in enumerate(sampled, start=1):
                qid = str(q.get("id", f"q_{idx_q}"))
                if qid in cached:
                    outcomes.append(cached[qid])
                    continue

                qtext = str(q["question"])
                gold_ans = str(q.get("answer", ""))
                gold_ev = str(q.get("evidence", ""))
                t0 = time.perf_counter()

                g_score: float | None = None
                tot_branches = 0
                tot_pruned = 0
                route_taken = ""
                early_stopped = False
                planner_model = ""
                planner_seconds = 0.0
                load_seconds = 0.0
                max_depth_reached = 0
                backtrack_count = 0
                llm_calls = 1
                retrieved_text = ""
                pred = ""

                try:
                    if mode == "naive_rag":
                        q_vec = active_embedder.embed([qtext])[0]
                        hits = pipeline.vector_store.search(
                            q_vec, top_k=config.retrieval.top_k_chunks
                        )
                        ctx = "\n".join(f"- [{c.chunk_id}] {c.text}" for c, _ in hits)
                        retrieved_text = " ".join(c.text for c, _ in hits)
                        prompt = (
                            f"Retrieved Text Chunks:\n{ctx}\n\n"
                            f"Question: {qtext}\n\n"
                            "Answer concisely and accurately in the same language as the question "
                            "based ONLY on the retrieved chunks."
                        )
                        pred = active_llm.complete(prompt, temperature=0.1, json_mode=False)
                    elif mode == "hybrid_direct":
                        res = pipeline.query(
                            qtext,
                            mode="hybrid",
                            reasoning="direct",
                            auto_load=False,
                            enable_compact=True,
                        )
                        pred = res.answer
                        retrieved_text = " ".join(ev.text for ev in res.evidences)
                        ledger = res.metadata.get("harness_ledger", {})
                        if isinstance(ledger, dict):
                            llm_calls = int(ledger.get("total_calls", 1))
                    elif mode == "ppr_cot":
                        res = pipeline.query(
                            qtext,
                            mode="ppr",
                            reasoning="cot",
                            auto_load=False,
                            enable_compact=True,
                        )
                        pred = res.answer
                        retrieved_text = " ".join(ev.text for ev in res.evidences)
                        rm = res.metadata.get("reasoning", {})
                        if isinstance(rm, dict):
                            ga = rm.get("grounding_audit", {})
                            if isinstance(ga, dict):
                                g_score = float(ga.get("grounding_score", 0.0))
                        ledger = res.metadata.get("harness_ledger", {})
                        if isinstance(ledger, dict):
                            llm_calls = int(ledger.get("total_calls", 1))
                    elif mode in {
                        "ppr_tot",
                        "ppr_multi_branch",
                        "ppr_auto",
                        "ppr_cot_iter",
                        "ppr_tot_bfs",
                        "ppr_tot_dfs",
                    }:
                        r_map = {
                            "ppr_tot": "tot",
                            "ppr_multi_branch": "multi_branch",
                            "ppr_auto": "auto",
                            "ppr_cot_iter": "cot_iter",
                            "ppr_tot_bfs": "tot_bfs",
                            "ppr_tot_dfs": "tot_dfs",
                        }
                        res = pipeline.query(
                            qtext,
                            mode="ppr",
                            reasoning=r_map[mode],
                            auto_load=False,
                            enable_compact=True,
                        )
                        pred = res.answer
                        retrieved_text = " ".join(ev.text for ev in res.evidences)
                        rd = res.metadata.get("route_decision")
                        if isinstance(rd, dict):
                            route_taken = str(rd.get("route", ""))
                            early_stopped = bool(rd.get("early_stop", False))
                        rm = res.metadata.get("reasoning", {})
                        if isinstance(rm, dict):
                            ga = rm.get("grounding_audit", {})
                            if isinstance(ga, dict):
                                g_score = float(ga.get("grounding_score", 0.0))
                            branches = rm.get("branches", [])
                            if isinstance(branches, list):
                                tot_branches = len(branches)
                                tot_pruned = sum(
                                    1 for b in branches if b.get("status") == "pruned"
                                )
                        ledger = res.metadata.get("harness_ledger", {})
                        if isinstance(ledger, dict):
                            llm_calls = int(ledger.get("total_calls", 1))
                            planner_model = str(ledger.get("planner_model", ""))
                            planner_seconds = float(ledger.get("planner_seconds", 0.0))
                            load_seconds = round(
                                float(ledger.get("planner_load_seconds", 0.0))
                                + float(ledger.get("synth_load_seconds", 0.0)),
                                3,
                            )
                            max_depth_reached = int(ledger.get("max_depth_reached", 0))
                            backtrack_count = int(ledger.get("backtrack_count", 0))
                    else:
                        raise ValueError(f"Unsupported GraphRAG-Bench eval mode: {mode!r}")
                except (ValueError, TimeoutError, OSError, ConnectionError) as exc:
                    pred = f"[EVAL_FALLBACK_ERROR] {type(exc).__name__}: {exc}"

                dt = round(time.perf_counter() - t0, 3)
                outcome = GRBQuestionOutcome(
                    question_id=qid,
                    source=str(q.get("source", "")),
                    question_type=str(q.get("question_type", "")),
                    mode=mode,
                    token_f1=compute_token_f1(pred, gold_ans),
                    rouge_l=compute_rouge_l_f1(pred, gold_ans),
                    evidence_coverage=compute_evidence_coverage(pred, gold_ev or gold_ans),
                    retrieval_evidence_recall=compute_evidence_coverage(
                        retrieved_text, gold_ev or gold_ans
                    ),
                    grounding_score=g_score,
                    tot_branches=tot_branches,
                    tot_pruned=tot_pruned,
                    route_taken=route_taken,
                    early_stopped=early_stopped,
                    planner_model=planner_model,
                    planner_seconds=planner_seconds,
                    load_seconds=load_seconds,
                    max_depth_reached=max_depth_reached,
                    backtrack_count=backtrack_count,
                    llm_calls=llm_calls,
                    latency_seconds=dt,
                    prediction=pred[:500],
                    retrieved_context=retrieved_text[:2400],
                )
                outcomes.append(outcome)
                ckpt_fh.write(outcome.model_dump_json() + "\n")
                ckpt_fh.flush()
                newly_run += 1

                if progress_every > 0 and (len(outcomes) % progress_every == 0 or len(outcomes) == total_n):
                    elapsed = max(time.perf_counter() - mode_t0, 1e-6)
                    avg_s = elapsed / max(newly_run, 1)
                    eta_m = ((total_n - len(outcomes)) * avg_s) / 60.0
                    cur_f1 = sum(o.token_f1 for o in outcomes) / len(outcomes) * 100.0
                    cur_ev = sum(o.evidence_coverage for o in outcomes) / len(outcomes) * 100.0
                    print(
                        f"[GRB {output_tag} | {mode}] {len(outcomes)}/{total_n} | "
                        f"F1={cur_f1:.1f}% | EvCov={cur_ev:.1f}% | avg={avg_s:.1f}s | ETA={eta_m:.1f}m",
                        flush=True,
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
                avg_retrieval_evidence_recall=round(
                    sum(o.retrieval_evidence_recall for o in outcomes) / n, 4
                ),
                avg_grounding_score=round(sum(g_vals) / len(g_vals), 4) if g_vals else None,
                avg_llm_calls=round(sum(o.llm_calls for o in outcomes) / n, 2),
                avg_latency_seconds=round(sum(o.latency_seconds for o in outcomes) / n, 2),
            )
        )

    embed_label = (
        f"{config.env.embedding_provider}/{config.env.embedding_model}"
        if config.env.embedding_provider != "local_hash"
        else "local_hash"
    )
    report = GRBEvalReport(
        question_type_filter=question_type,
        source_filter=source_filter or "ALL",
        total_questions=len(sampled),
        llm_model=config.env.llm_model,
        embedding_model=embed_label,
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
        f"# GraphRAG-Bench 多范式评测报告（`{question_type}` | tag=`{output_tag}`）",
        "",
        (
            f"- **题数**：`{len(sampled)}` | **语料过滤**：`{source_filter or 'ALL'}` | "
            f"**模型**：`{config.env.llm_model}` | **嵌入**：`{embed_label}`"
        ),
        "",
        "| 范式模式 | Token-F1 | ROUGE-L | 回答证据覆盖 | 检索证据召回 | Grounding | 平均调用 | 平均耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        g_str = f"{s.avg_grounding_score * 100:.1f}%" if s.avg_grounding_score is not None else "-"
        md_lines.append(
            f"| `{s.mode}` | **{s.avg_token_f1 * 100:.1f}%** | {s.avg_rouge_l * 100:.1f}% | "
            f"{s.avg_evidence_coverage * 100:.1f}% | {s.avg_retrieval_evidence_recall * 100:.1f}% | "
            f"{g_str} | {s.avg_llm_calls:.1f} | {s.avg_latency_seconds:.2f}s |"
        )
    report_md = out_dir / "grb_eval_report.md"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report, report_md

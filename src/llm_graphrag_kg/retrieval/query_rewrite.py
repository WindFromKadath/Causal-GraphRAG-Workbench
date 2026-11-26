"""Stage 4.3 Query Rewrite, Multi-Query Expansion & Self-Query Metadata Filter Extraction.

Follows JavaGuide enterprise RAG guardrails:
- Always preserves the original user question in parallel fusion (`queries[0] == original_query`)
  so LLM rewrites can NEVER drift off-topic or drop exact fault codes / entity names.
- Extracts lightweight Self-Query metadata filters (doc/section hints, failure codes, element types).
- Provides deterministic zero-LLM fast path + optional LLM structured multi-query expansion.
- Supports Corrective RAG query broadening (`expand_query_for_retry`) when initial evidence is weak.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.interfaces import BaseLLMClient
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.providers.storage.local_vector import tokenize_bm25

_CODE_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}(?:[-_][A-Z0-9]{1,6})?\b")
_CONJUNCTION_SPLIT_RE = re.compile(r"(?:同时|并且|以及|和|与|\band\b|\bplus\b|,|，|；|;)", flags=re.IGNORECASE)
_TABLE_HINT_RE = re.compile(r"(表格|阈值表|参数表|对照表|标准值|报警值|table|threshold|spec)", flags=re.IGNORECASE)
_CODE_HINT_RE = re.compile(r"(代码|脚本|函数|类|实现|python|code|function|class)", flags=re.IGNORECASE)


class SelfQueryFilters(BaseModel):
    """Structured metadata hints extracted from natural language queries."""

    element_types: list[str] = Field(default_factory=list)  # table | code | text
    exact_codes: list[str] = Field(default_factory=list)  # e.g., SHH, ISO, FMEA
    section_hints: list[str] = Field(default_factory=list)


class QueryRewriteResult(BaseModel):
    """Multi-query expansion output guaranteeing the original query is always preserved."""

    original_query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    self_query_filters: SelfQueryFilters = Field(default_factory=SelfQueryFilters)
    used_llm: bool = False
    strategy: str = "deterministic"


class _LLMRewriteSchema(BaseModel):
    rewritten_queries: list[str] = Field(default_factory=list, max_length=3)
    section_hints: list[str] = Field(default_factory=list, max_length=3)


def extract_self_query_filters(question: str) -> SelfQueryFilters:
    """Deterministically extract structured metadata hints (codes, table/code intent)."""
    elem_types: list[str] = []
    if _TABLE_HINT_RE.search(question):
        elem_types.append("table")
    if _CODE_HINT_RE.search(question):
        elem_types.append("code")

    codes = list(dict.fromkeys(m.group(0) for m in _CODE_TOKEN_RE.finditer(question)))[:5]
    return SelfQueryFilters(element_types=elem_types, exact_codes=codes)


def _deterministic_multi_queries(question: str, *, max_variants: int = 2) -> list[str]:
    """Generate clean keyword/sub-clause query variants without any LLM latency."""
    q = question.strip()
    variants: list[str] = []

    # 1. Clause decomposition for multi-symptom / multi-hop questions
    clauses = [c.strip(" ?？。！!") for c in _CONJUNCTION_SPLIT_RE.split(q) if len(c.strip()) >= 6]
    if len(clauses) >= 2:
        for cl in clauses[:max_variants]:
            if cl.lower() != q.lower() and cl not in variants:
                variants.append(cl)

    # 2. High-salience keyword distillation (strips conversational filler)
    toks = tokenize_bm25(q)
    if len(toks) >= 4:
        salient = " ".join(dict.fromkeys(sorted(toks, key=lambda t: -len(t))[:8]))
        if salient and salient.lower() != q.lower() and salient not in variants:
            variants.append(salient)

    return variants[:max_variants]


def rewrite_query(
    question: str,
    *,
    llm: BaseLLMClient | None = None,
    use_llm: bool = False,
    max_variants: int = 2,
) -> QueryRewriteResult:
    """Produce parallel multi-queries while strictly preserving `original_query` at index 0."""
    clean_q = question.strip()
    filters = extract_self_query_filters(clean_q)
    variants: list[str] = []
    used_llm = False
    strategy = "deterministic"

    if use_llm and llm is not None:
        try:
            harness = LLMHarness(llm, json_repair_retries=0, max_calls=2, max_wall_seconds=15.0)
            prompt = (
                "Generate up to 2 concise search rewrites for hybrid vector+BM25 retrieval.\n"
                "Preserve exact domain terms, sensor names, and fault codes.\n"
                f"Question: {clean_q}\n"
                'Return JSON: {"rewritten_queries": ["..."], "section_hints": ["..."]}'
            )
            parsed = harness.call_structured(
                _LLMRewriteSchema,
                [
                    {"role": "system", "content": "You rewrite queries for enterprise RAG retrieval. JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            variants = [
                v.strip()
                for v in parsed.rewritten_queries
                if v.strip() and v.strip().lower() != clean_q.lower()
            ][:max_variants]
            if parsed.section_hints:
                filters.section_hints = [s.strip() for s in parsed.section_hints if s.strip()][:3]
            used_llm = True
            strategy = "llm_multi_query"
        except Exception:  # noqa: BLE001
            variants = _deterministic_multi_queries(clean_q, max_variants=max_variants)
            strategy = "deterministic_fallback"
    else:
        variants = _deterministic_multi_queries(clean_q, max_variants=max_variants)

    # Iron rule: ALWAYS include original query first so parallel fusion never loses recall
    deduped = [clean_q]
    for v in variants:
        if v.lower() not in {x.lower() for x in deduped}:
            deduped.append(v)

    return QueryRewriteResult(
        original_query=clean_q,
        rewritten_queries=deduped,
        self_query_filters=filters,
        used_llm=used_llm,
        strategy=strategy,
    )


def expand_query_for_retry(
    original_query: str,
    previous_queries: list[str],
    *,
    retry_index: int = 1,
) -> list[str]:
    """Corrective RAG fallback: broaden retrieval terms when first-pass evidence is insufficient."""
    toks = [t for t in tokenize_bm25(original_query) if len(t) >= 2]
    expanded = list(previous_queries)
    # Add focused bigram/keyword windows and relaxed fault-diagnosis synonyms
    if toks:
        core = " ".join(dict.fromkeys(toks[:6]))
        if core and core not in expanded:
            expanded.append(core)
        if retry_index >= 1 and len(toks) >= 2:
            tail = " ".join(dict.fromkeys(toks[-4:]))
            if tail and tail not in expanded:
                expanded.append(tail)
    extra_hints: list[str] = []
    low = original_query.lower()
    if any(k in low for k in ("振动", "vibration", "2倍频", "2x")):
        extra_hints.append("轴系不对中 联轴器 轴承振动 ISO 20816")
    if any(k in low for k in ("温升", "温度", "temperature", "bearing", "轴承")):
        extra_hints.append("滚动轴承 润滑脂劣化 轴承过热 排查措施")
    for h in extra_hints:
        if h not in expanded:
            expanded.append(h)
    return expanded[:5]


def fuse_multi_query_hits(
    per_query_hits: list[list[tuple[Any, float]]],
    *,
    rrf_k: int = 60,
    top_k: int = 40,
) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion across original + rewritten queries (giving original 1.25x anchor weight)."""
    fused: dict[str, float] = {}
    obj_map: dict[str, Any] = {}
    for q_idx, hits in enumerate(per_query_hits):
        weight = 1.25 if q_idx == 0 else 1.0
        for rank, (chunk, _raw_score) in enumerate(hits, start=1):
            cid = str(getattr(chunk, "chunk_id", id(chunk)))
            obj_map[cid] = chunk
            fused[cid] = fused.get(cid, 0.0) + weight / (rrf_k + rank)
    ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [(obj_map[cid], round(score, 6)) for cid, score in ranked]

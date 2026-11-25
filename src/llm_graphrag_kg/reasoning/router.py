"""Adaptive Query Router & Dynamic Early Stopping for Stage 3.2 GraphRAG.

Inspired by GraphRAG-Bench ("When to use Graphs in RAG"), OpenSPG KAG, and Search-R1:
routes each query to the optimal reasoning paradigm based on linguistic intent signals
and deterministic first-hop retrieval coverage:
  - `exclusive_diagnosis` -> `tot` (competitive pruning + L2 counter-evidence folding)
  - `multi_facet_synthesis` -> `multi_branch` (non-pruning additive union synthesis)
  - `fact_direct` (or high-confidence early stop) -> `cot` (single-pass grounded CoT)
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.schema import Entity, EvidenceItem, Relation

_EXCLUSIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "zh_exclusive_diag",
        re.compile(
            r"(?:优先排查|最可能|根因|排除|反证|同时出现|并发|故障定位|哪一种|哪一个|哪种故障|不属于|无关|SOP)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "en_exclusive_diag",
        re.compile(
            r"\b(?:root cause|most likely|primary cause|rule out|prioritize|"
            r"differential diagnosis|which (?:fault|failure|sensor|cause)|"
            r"irrelevant|not relevant|exclude|instead of)\b",
            flags=re.IGNORECASE,
        ),
    ),
)

_MULTI_FACET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "zh_multi_facet",
        re.compile(
            r"(?:总结|综述|概述|全面|对比|比较|关系|如何影响|哪些因素|综合|演进|分别|多方面|机制|联系)"
        ),
    ),
    (
        "en_multi_facet",
        re.compile(
            r"\b(?:summarize|summary|overview|comprehensive|compare|contrast|"
            r"relationship|relationships|how do|how does|explain how|explain why|"
            r"discuss|factors|roles|implications|across|impact of|influence|"
            r"connect|connections|themes|interplay|aspects|what are the|describe how|"
            r"contextual|evolution|contribute to|different ways)\b",
            flags=re.IGNORECASE,
        ),
    ),
)

_SIMPLE_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "zh_simple_fact",
        re.compile(r"(?:是什么|是谁|何时|哪里|多少|定义是什么)"),
    ),
    (
        "en_simple_fact",
        re.compile(
            r"^(?:what is|who is|who was|when did|when was|where is|where was|how many|define)\b",
            flags=re.IGNORECASE,
        ),
    ),
)

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "into", "what", "when", "where", "which",
    "who", "why", "how", "can", "may", "will", "should", "could", "would", "does",
    "about", "their", "they", "them", "these", "those", "between", "through",
}


class RouteDecision(BaseModel):
    """Explainable routing decision produced before invoking a reasoning engine."""

    route: Literal["fact_direct", "exclusive_diagnosis", "multi_facet_synthesis"]
    target_paradigm: Literal["cot", "tot", "multi_branch"]
    early_stop: bool = False
    coverage_score: float = 0.0
    matched_signals: list[str] = Field(default_factory=list)
    reason: str = ""


def _extract_content_terms(text: str) -> set[str]:
    lowered = text.lower()
    en = {
        tok
        for tok in re.findall(r"[a-z0-9_]{3,}", lowered)
        if tok not in _STOPWORDS
    }
    cjk = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
    return en | bigrams


def compute_first_hop_coverage(
    question: str,
    *,
    seed_entities: list[Entity],
    relations: list[Relation],
    evidences: list[EvidenceItem],
) -> float:
    """Estimate deterministic first-hop coverage of question terms by retrieved facts."""
    q_terms = _extract_content_terms(question)
    if not q_terms:
        return 0.0
    haystack = " ".join(
        [
            *(f"{e.name} {e.description}" for e in seed_entities[:10]),
            *(r.description for r in relations[:16]),
            *(ev.text for ev in evidences[:5]),
        ]
    ).lower()
    if not haystack.strip():
        return 0.0
    hits = sum(1 for term in q_terms if term in haystack)
    return round(min(1.0, hits / len(q_terms)), 4)


def route_query(
    question: str,
    *,
    seed_entities: list[Entity] | None = None,
    relations: list[Relation] | None = None,
    evidences: list[EvidenceItem] | None = None,
    enable_early_stop: bool = True,
    early_stop_coverage_threshold: float = 0.80,
) -> RouteDecision:
    """Deterministically route a query to `cot`, `tot`, or `multi_branch`."""
    q = question.strip()
    seeds = seed_entities or []
    rels = relations or []
    evs = evidences or []

    coverage = compute_first_hop_coverage(
        q,
        seed_entities=seeds,
        relations=rels,
        evidences=evs,
    )

    exclusive_hits = [name for name, pat in _EXCLUSIVE_PATTERNS if pat.search(q)]
    multi_hits = [name for name, pat in _MULTI_FACET_PATTERNS if pat.search(q)]
    simple_hits = [name for name, pat in _SIMPLE_FACT_PATTERNS if pat.search(q)]

    # 1. Exclusive diagnosis / competing hypothesis elimination -> Graph-ToT (competitive pruning)
    if exclusive_hits:
        return RouteDecision(
            route="exclusive_diagnosis",
            target_paradigm="tot",
            early_stop=False,
            coverage_score=coverage,
            matched_signals=exclusive_hits,
            reason=(
                "检测到互斥根因排查/反证排除意图，路由至 Graph-ToT 竞争分支剪枝与 L2 反证折叠。"
            ),
        )

    # 2. Explicit multi-facet synthesis / summarization -> Non-Pruning Multi-Branch Aggregation
    if multi_hits:
        return RouteDecision(
            route="multi_facet_synthesis",
            target_paradigm="multi_branch",
            early_stop=False,
            coverage_score=coverage,
            matched_signals=multi_hits,
            reason=(
                "检测到多因子综合/跨上下文摘要意图，路由至 Multi-Branch 非剪枝多分支并集聚合。"
            ),
        )

    # 3. Simple single-hop factoid lookup -> Fast Path 1-step Grounded CoT
    word_count = len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", q))
    if simple_hits and word_count <= 16:
        return RouteDecision(
            route="fact_direct",
            target_paradigm="cot",
            early_stop=True,
            coverage_score=coverage,
            matched_signals=[*simple_hits, "short_fact_query"],
            reason="检测到直接事实单跳问句，启用 Fast Path 单轮显性 Graph-CoT。",
        )

    # 4. Dynamic Early Stopping (Search-R1 inspired): high first-hop coverage skips multi-branch
    if enable_early_stop and coverage >= early_stop_coverage_threshold and len(evs) >= 3:
        return RouteDecision(
            route="fact_direct",
            target_paradigm="cot",
            early_stop=True,
            coverage_score=coverage,
            matched_signals=[f"early_stop_cov={coverage:.2f}>={early_stop_coverage_threshold:.2f}"],
            reason=(
                f"首跳图谱与原文覆盖率达 {coverage * 100:.1f}%（≥ {early_stop_coverage_threshold * 100:.0f}%），"
                "触发动态早停直走单轮 Grounded CoT 降低延迟。"
            ),
        )

    # 5. Default for complex multi-clause reasoning without exclusive pruning cues -> Multi-Branch
    return RouteDecision(
        route="multi_facet_synthesis",
        target_paradigm="multi_branch",
        early_stop=False,
        coverage_score=coverage,
        matched_signals=["complex_default_additive"],
        reason=(
            f"复杂推理问题且首跳覆盖率为 {coverage * 100:.1f}%，路由至 Multi-Branch 多分支无剪枝聚合。"
        ),
    )

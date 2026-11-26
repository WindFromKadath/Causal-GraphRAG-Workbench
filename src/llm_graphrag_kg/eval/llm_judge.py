"""Official-aligned 4-Dimensional LLM-as-a-Judge evaluator (Stage 3.3).

Implements GraphRAG-Bench's four semantic evaluation dimensions without LangChain:
1. Answer Accuracy: 0.75 * F1(TP, FP, FN) + 0.25 * scaled_cosine_similarity(pred, ref)
2. Faithfulness: fraction of atomic answer claims directly supported by retrieved context
3. Semantic Coverage: fraction of reference factual points semantically covered by prediction
4. Refusal Rate: explicit abstention / non-answer detection so short refusals cannot cheat metrics
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient, BaseLLMClient
from llm_graphrag_kg.harness import LLMHarness
from llm_graphrag_kg.providers.factory import create_embedding_client, create_llm_client

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial GraphRAG evaluation judge. "
    "Evaluate factual alignment, faithfulness to context, and reference fact coverage objectively. "
    "Respond ONLY with valid JSON matching the requested schema."
)

_REFUSAL_PATTERNS = [
    r"\b(?:does|do)\s+not\s+(?:provide|contain|mention|specify|include|state)\b",
    r"\bnot\s+(?:explicitly\s+)?(?:mentioned|specified|provided|stated|found|available|detailed)\b",
    r"\b(?:cannot|can't|unable\s+to)\s+(?:be\s+determined|determine|answer|verify|confirm)\b",
    r"\binsufficient\s+(?:information|evidence|context|data)\b",
    r"\bno\s+(?:relevant\s+)?(?:information|evidence|data|details)\s+(?:is\s+)?(?:provided|available|found)\b",
    r"未(?:明确)?提及|信息不足|无法(?:确定|回答|判断)|缺乏(?:足够)?证据",
]
_REFUSAL_REGEX = re.compile("|".join(_REFUSAL_PATTERNS), flags=re.IGNORECASE)


def detect_refusal(prediction: str) -> bool:
    """Detect explicit abstention/refusal answers that lack substantive factual content."""
    text = prediction.strip()
    if not text:
        return True
    matches = _REFUSAL_REGEX.findall(text)
    if not matches:
        return False
    # If the response is very short or predominantly hedging/abstaining, classify as refusal.
    # Long answers that give substantive facts plus a minor caveat ("specific route not detailed")
    # are only flagged if the refusal occurs in the opening sentence or dominates the text.
    first_sent = re.split(r"[.!?。\n]", text, maxsplit=1)[0]
    if len(text) <= 220 or _REFUSAL_REGEX.search(first_sent):
        return True
    return len(matches) >= 2 and len(text) <= 420


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    """GraphRAG-Bench official F-beta score over TP/FP/FN statement classifications."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    if precision + recall == 0.0:
        return 0.0
    b2 = beta * beta
    return float((1.0 + b2) * (precision * recall) / (b2 * precision + recall + 1e-10))


def _scaled_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (na * nb)))
    return float((cos + 1.0) / 2.0)


class AccuracyCoverageOutput(BaseModel):
    """Structured TP/FP/FN + Reference Coverage verdict matching GraphRAG-Bench semantics."""

    tp_claims: list[str] = Field(
        default_factory=list,
        description="Prediction claims directly supported by the reference answer.",
    )
    fp_claims: list[str] = Field(
        default_factory=list,
        description="Prediction claims contradicted by or unsupported by the reference answer.",
    )
    fn_claims: list[str] = Field(
        default_factory=list,
        description="Key reference facts omitted from the prediction.",
    )
    reference_facts_total: int = Field(default=0, ge=0)
    reference_facts_covered: int = Field(default=0, ge=0)
    is_refusal: bool = Field(
        default=False,
        description="True if the prediction declines to answer due to missing context.",
    )


class ClaimVerdict(BaseModel):
    statement: str
    supported: int = Field(ge=0, le=1)
    reason: str = ""


class FaithfulnessOutput(BaseModel):
    """Structured statement-level faithfulness check against retrieved context."""

    verdicts: list[ClaimVerdict] = Field(default_factory=list)


class JudgeQuestionScore(BaseModel):
    """Per-question 4D official-aligned LLM-as-a-Judge result."""

    question_id: str
    question_type: str
    mode: str
    answer_accuracy: float = Field(ge=0.0, le=1.0)
    factuality_f1: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    semantic_coverage: float = Field(ge=0.0, le=1.0)
    retrieval_evidence_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    is_refusal: bool = False
    tp_count: int = 0
    fp_count: int = 0
    fn_count: int = 0
    faithful_claims: int = 0
    total_claims: int = 0
    token_f1_legacy: float = 0.0
    judge_latency_seconds: float = 0.0


class JudgeModeSummary(BaseModel):
    """Aggregated 4D Judge metrics for a single retrieval/reasoning mode."""

    mode: str
    total: int
    avg_answer_accuracy: float
    avg_factuality_f1: float
    avg_semantic_similarity: float
    avg_faithfulness: float
    avg_semantic_coverage: float
    avg_retrieval_evidence_recall: float
    refusal_rate: float
    avg_token_f1_legacy: float
    avg_judge_latency_seconds: float
    by_question_type: dict[str, dict[str, float]] = Field(default_factory=dict)


class JudgeReport(BaseModel):
    """Cross-mode official-aligned 4D LLM-as-a-Judge report."""

    tag: str
    judge_model: str
    embedding_model: str
    total_questions: int
    modes: list[JudgeModeSummary] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# GraphRAG-Bench 官方口径四维 LLM-as-a-Judge 评测报告（tag=`{self.tag}`）",
            "",
            f"- **题数**：`{self.total_questions}` | **裁判模型**：`{self.judge_model}` "
            f"| **语义相似度嵌入**：`{self.embedding_model}`",
            "- **口径说明**：`Answer Accuracy = 0.75 * Factuality_F1 + 0.25 * CosineSim`，"
            "`Faithfulness` 逐断言校验检索上下文忠实度，`Sem. Coverage` 语义核验参考事实覆盖率，"
            "`Refusal Rate` 独立揭示短拒答伪装。",
            "",
            "| 范式模式 | Answer Accuracy | Faithfulness | Sem. Coverage | "
            "Ret EvRec | Refusal Rate | Legacy Token-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in self.modes:
            lines.append(
                f"| `{s.mode}` | **{s.avg_answer_accuracy * 100:.1f}%** | "
                f"{s.avg_faithfulness * 100:.1f}% | {s.avg_semantic_coverage * 100:.1f}% | "
                f"{s.avg_retrieval_evidence_recall * 100:.1f}% | "
                f"{s.refusal_rate * 100:.1f}% | {s.avg_token_f1_legacy * 100:.1f}% |"
            )

        # Per question-type breakdown table if multiple types exist
        all_types = sorted({qt for s in self.modes for qt in s.by_question_type})
        if all_types:
            lines.extend(["", "## 分题型细项（Answer Accuracy / Sem. Coverage / Refusal）", ""])
            for qt in all_types:
                lines.append(f"### {qt}")
                lines.append("")
                lines.append("| 范式模式 | Accuracy | Faithfulness | Sem. Coverage | Refusal |")
                lines.append("|---|---:|---:|---:|---:|")
                for s in self.modes:
                    bt = s.by_question_type.get(qt)
                    if not bt:
                        continue
                    lines.append(
                        f"| `{s.mode}` | **{bt['answer_accuracy'] * 100:.1f}%** | "
                        f"{bt['faithfulness'] * 100:.1f}% | {bt['semantic_coverage'] * 100:.1f}% | "
                        f"{bt['refusal_rate'] * 100:.1f}% |"
                    )
                lines.append("")

        return "\n".join(lines) + "\n"


class LLMJudgeEvaluator:
    """Protocol-decoupled 4D LLM-as-a-Judge aligned with GraphRAG-Bench Evaluation."""

    def __init__(
        self,
        llm: BaseLLMClient,
        embedder: BaseEmbeddingClient,
        *,
        factuality_weight: float = 0.75,
        similarity_weight: float = 0.25,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.harness = LLMHarness(
            llm, json_repair_retries=1, max_calls=2000, max_wall_seconds=3600.0
        )
        self.factuality_weight = factuality_weight
        self.similarity_weight = similarity_weight

    def score_one(
        self,
        *,
        question_id: str,
        question_type: str,
        mode: str,
        question: str,
        prediction: str,
        reference: str,
        contexts: list[str],
        retrieval_evidence_recall: float = 0.0,
        token_f1_legacy: float = 0.0,
    ) -> JudgeQuestionScore:
        """Compute official 4D semantic metrics for a single QA prediction."""
        t0 = time.perf_counter()
        pred_clean = prediction.strip()
        ref_clean = reference.strip()

        rule_refusal = detect_refusal(pred_clean)

        # 1. Semantic embedding similarity (GraphRAG-Bench answer_accuracy.py)
        if pred_clean and ref_clean:
            vecs = self.embedder.embed([pred_clean[:2500], ref_clean[:2500]])
            sem_sim = _scaled_cosine_similarity(vecs[0], vecs[1])
        else:
            sem_sim = 0.0

        # 2. Factuality TP/FP/FN + Reference Fact Coverage (LLM Judge Call 1)
        acc_prompt = (
            "LLM_JUDGE_ACCURACY_COVERAGE_TASK\n"
            "Evaluate the Candidate Prediction against the Reference Answer for the Question.\n"
            "1. Break the Reference Answer into atomic factual points (reference_facts_total).\n"
            "2. Classify atomic factual claims from Prediction & Reference into:\n"
            "   - tp_claims: claims in Prediction directly supported by Reference\n"
            "   - fp_claims: claims in Prediction contradicted or unsupported by Reference\n"
            "   - fn_claims: key facts in Reference missing from Prediction\n"
            "3. Count how many reference facts are semantically covered (reference_facts_covered).\n"
            "4. Set is_refusal=true ONLY if Prediction refuses/abstains from answering.\n"
            "Return strict JSON:\n"
            '{"tp_claims": ["..."], "fp_claims": ["..."], "fn_claims": ["..."], '
            '"reference_facts_total": 3, "reference_facts_covered": 2, "is_refusal": false}\n\n'
            f"Question: {question[:600]}\n\n"
            f"Reference Answer:\n{ref_clean[:2200]}\n\n"
            f"Candidate Prediction:\n{pred_clean[:2200]}"
        )
        try:
            acc_out, _ = self.harness.run_structured(
                acc_prompt,
                AccuracyCoverageOutput,
                step_name="judge_accuracy_coverage",
                system_prompt=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1024,
            )
            tp = len(acc_out.tp_claims)
            fp = len(acc_out.fp_claims)
            fn = len(acc_out.fn_claims)
            fact_f1 = fbeta_score(tp, fp, fn, beta=1.0)
            if acc_out.reference_facts_total > 0:
                sem_cov = min(1.0, acc_out.reference_facts_covered / acc_out.reference_facts_total)
            elif tp + fn > 0:
                sem_cov = tp / (tp + fn)
            else:
                sem_cov = 0.0
            is_refusal = bool(rule_refusal or acc_out.is_refusal)
        except ValueError:
            tp = fp = fn = 0
            fact_f1 = 0.0
            sem_cov = 0.0
            is_refusal = rule_refusal

        # Guardrail: an explicit refusal with zero TP cannot score high accuracy via cosine alone
        if is_refusal and tp == 0:
            fact_f1 = 0.0
            sem_cov = 0.0

        ans_acc = (
            self.factuality_weight * fact_f1 + self.similarity_weight * sem_sim
            if (self.factuality_weight + self.similarity_weight) > 0
            else 0.0
        )

        # 3. Faithfulness against Context (LLM Judge Call 2)
        context_str = "\n".join(c.strip() for c in contexts if c.strip())[:8000]
        faithful_claims = 0
        total_claims = 0
        if not pred_clean:
            faithfulness = 0.0
        elif not context_str:
            faithfulness = 0.0
        else:
            faith_prompt = (
                "LLM_JUDGE_FAITHFULNESS_TASK\n"
                "Decompose the Candidate Prediction into up to 8 concise atomic factual claims "
                "and verify whether each claim can be directly inferred from the Context.\n"
                "Return strict JSON:\n"
                '{"verdicts": [{"statement": "<claim>", "supported": 1, "reason": "<brief>"}]}\n'
                "Use supported=1 ONLY when directly supported by Context, else 0.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Candidate Prediction:\n{pred_clean[:2200]}"
            )
            try:
                faith_out, _ = self.harness.run_structured(
                    faith_prompt,
                    FaithfulnessOutput,
                    step_name="judge_faithfulness",
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=1024,
                )
                total_claims = len(faith_out.verdicts)
                faithful_claims = sum(1 for v in faith_out.verdicts if int(v.supported) == 1)
                faithfulness = (faithful_claims / total_claims) if total_claims > 0 else 0.0
            except ValueError:
                faithfulness = 0.0

        elapsed = time.perf_counter() - t0
        return JudgeQuestionScore(
            question_id=question_id,
            question_type=question_type,
            mode=mode,
            answer_accuracy=round(max(0.0, min(1.0, ans_acc)), 4),
            factuality_f1=round(max(0.0, min(1.0, fact_f1)), 4),
            semantic_similarity=round(max(0.0, min(1.0, sem_sim)), 4),
            faithfulness=round(max(0.0, min(1.0, faithfulness)), 4),
            semantic_coverage=round(max(0.0, min(1.0, sem_cov)), 4),
            retrieval_evidence_recall=round(max(0.0, min(1.0, retrieval_evidence_recall)), 4),
            is_refusal=is_refusal,
            tp_count=tp,
            fp_count=fp,
            fn_count=fn,
            faithful_claims=faithful_claims,
            total_claims=total_claims,
            token_f1_legacy=round(token_f1_legacy, 4),
            judge_latency_seconds=round(elapsed, 3),
        )


def _summarize_mode_scores(mode: str, scores: list[JudgeQuestionScore]) -> JudgeModeSummary:
    n = len(scores) or 1
    by_type_groups: dict[str, list[JudgeQuestionScore]] = {}
    for s in scores:
        by_type_groups.setdefault(s.question_type, []).append(s)

    by_type: dict[str, dict[str, float]] = {}
    for qt, grp in sorted(by_type_groups.items()):
        m = len(grp) or 1
        by_type[qt] = {
            "count": float(len(grp)),
            "answer_accuracy": round(sum(x.answer_accuracy for x in grp) / m, 4),
            "faithfulness": round(sum(x.faithfulness for x in grp) / m, 4),
            "semantic_coverage": round(sum(x.semantic_coverage for x in grp) / m, 4),
            "refusal_rate": round(sum(1 for x in grp if x.is_refusal) / m, 4),
        }

    return JudgeModeSummary(
        mode=mode,
        total=len(scores),
        avg_answer_accuracy=round(sum(x.answer_accuracy for x in scores) / n, 4),
        avg_factuality_f1=round(sum(x.factuality_f1 for x in scores) / n, 4),
        avg_semantic_similarity=round(sum(x.semantic_similarity for x in scores) / n, 4),
        avg_faithfulness=round(sum(x.faithfulness for x in scores) / n, 4),
        avg_semantic_coverage=round(sum(x.semantic_coverage for x in scores) / n, 4),
        avg_retrieval_evidence_recall=round(
            sum(x.retrieval_evidence_recall for x in scores) / n, 4
        ),
        refusal_rate=round(sum(1 for x in scores if x.is_refusal) / n, 4),
        avg_token_f1_legacy=round(sum(x.token_f1_legacy for x in scores) / n, 4),
        avg_judge_latency_seconds=round(sum(x.judge_latency_seconds for x in scores) / n, 3),
        by_question_type=by_type,
    )


def judge_grb_checkpoints(
    *,
    questions_path: Path,
    checkpoint_map: dict[str, Path],
    config: AppConfig,
    output_dir: Path,
    tag: str = "stage33_judge",
    limit: int = 0,
    resume: bool = True,
    progress_every: int = 2,
    llm: BaseLLMClient | None = None,
    embedder: BaseEmbeddingClient | None = None,
) -> tuple[JudgeReport, Path]:
    """Run resumable 4D LLM-as-a-Judge across one or more GraphRAG-Bench checkpoint files."""
    if not questions_path.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    raw_qs: list[dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))
    q_lookup: dict[str, dict[str, Any]] = {str(q["id"]): q for q in raw_qs if "id" in q}

    active_llm = llm or create_llm_client(config)
    active_embedder = embedder or create_embedding_client(config)
    evaluator = LLMJudgeEvaluator(active_llm, active_embedder)

    out_dir = output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[JudgeModeSummary] = []
    all_scores: dict[str, list[JudgeQuestionScore]] = {}

    for mode_label, ckpt_path in checkpoint_map.items():
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for mode `{mode_label}`: {ckpt_path}")

        rows = [
            json.loads(line)
            for line in ckpt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit > 0:
            rows = rows[:limit]

        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", mode_label)
        judge_ckpt = out_dir / f"judge_{safe_label}.checkpoint.jsonl"
        cached: dict[str, JudgeQuestionScore] = {}
        if resume and judge_ckpt.is_file():
            for line in judge_ckpt.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = JudgeQuestionScore.model_validate_json(line)
                    cached[rec.question_id] = rec

        scores: list[JudgeQuestionScore] = []
        with judge_ckpt.open("a", encoding="utf-8") as ckpt_fh:
            for idx, row in enumerate(rows, start=1):
                qid = str(row["question_id"])
                if qid in cached:
                    scores.append(cached[qid])
                    continue

                q_meta = q_lookup.get(qid, {})
                q_text = str(q_meta.get("question", qid))
                ref_ans = str(q_meta.get("answer", ""))
                ev_raw = q_meta.get("evidence", [])
                ev_list = [str(x) for x in ev_raw] if isinstance(ev_raw, list) else [str(ev_raw)]

                # Prefer retrieved_context if saved in checkpoint, else fall back to gold evidence
                ret_ctx = str(row.get("retrieved_context", "")).strip()
                contexts = [ret_ctx] if ret_ctx else ev_list

                scored = evaluator.score_one(
                    question_id=qid,
                    question_type=str(row.get("question_type", q_meta.get("question_type", ""))),
                    mode=mode_label,
                    question=q_text,
                    prediction=str(row.get("prediction", "")),
                    reference=ref_ans,
                    contexts=contexts,
                    retrieval_evidence_recall=float(row.get("retrieval_evidence_recall", 0.0)),
                    token_f1_legacy=float(row.get("token_f1", 0.0)),
                )
                scores.append(scored)
                ckpt_fh.write(scored.model_dump_json() + "\n")
                ckpt_fh.flush()

                if progress_every > 0 and (idx % progress_every == 0 or idx == len(rows)):
                    avg_acc = sum(x.answer_accuracy for x in scores) / len(scores)
                    avg_f = sum(x.faithfulness for x in scores) / len(scores)
                    avg_c = sum(x.semantic_coverage for x in scores) / len(scores)
                    ref_r = sum(1 for x in scores if x.is_refusal) / len(scores)
                    print(
                        f"[Judge {tag} | {mode_label}] {idx}/{len(rows)} | "
                        f"Acc={avg_acc * 100:.1f}% | Faith={avg_f * 100:.1f}% | "
                        f"Cov={avg_c * 100:.1f}% | Refusal={ref_r * 100:.1f}%",
                        flush=True,
                    )

        all_scores[mode_label] = scores
        summaries.append(_summarize_mode_scores(mode_label, scores))

    report = JudgeReport(
        tag=tag,
        judge_model=config.env.llm_model,
        embedding_model=f"{config.env.embedding_provider}/{config.env.embedding_model}",
        total_questions=max((s.total for s in summaries), default=0),
        modes=summaries,
    )
    report_md = out_dir / "judge_eval_report.md"
    report_json = out_dir / "judge_eval_results.json"
    report_md.write_text(report.to_markdown(), encoding="utf-8")
    report_json.write_text(
        json.dumps(
            {
                "report": report.model_dump(),
                "scores": {m: [x.model_dump() for x in lst] for m, lst in all_scores.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report, report_md

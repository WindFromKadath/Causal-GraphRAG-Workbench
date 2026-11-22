"""LLMHarness runtime: Prompt compilation, budget guardrails, JSON auto-repair, and TraceLedger."""

from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from llm_graphrag_kg.core.interfaces import BaseLLMClient
from llm_graphrag_kg.harness.context_compactor import ContextCompactor, estimate_tokens

T = TypeVar("T", bound=BaseModel)


class CallTraceRecord(BaseModel):
    """Telemetry record for a single LLM invocation executed inside `LLMHarness`."""

    step_name: str
    prompt_tokens_est: int
    response_tokens_est: int
    wall_seconds: float
    repair_attempts: int = 0
    succeeded: bool = True
    error: str = ""


class TraceLedger(BaseModel):
    """Aggregated telemetry ledger across one single-step or multi-step reasoning session."""

    records: list[CallTraceRecord] = Field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def total_repairs(self) -> int:
        return sum(r.repair_attempts for r in self.records)

    @property
    def total_wall_seconds(self) -> float:
        return round(sum(r.wall_seconds for r in self.records), 3)

    @property
    def total_prompt_tokens_est(self) -> int:
        return sum(r.prompt_tokens_est for r in self.records)

    @property
    def total_response_tokens_est(self) -> int:
        return sum(r.response_tokens_est for r in self.records)

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_repairs": self.total_repairs,
            "total_wall_seconds": self.total_wall_seconds,
            "total_prompt_tokens_est": self.total_prompt_tokens_est,
            "total_response_tokens_est": self.total_response_tokens_est,
        }


def _extract_json_block(raw_text: str) -> str:
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class LLMHarness:
    """Engineering harness wrapping any `BaseLLMClient` with guardrails, repair, and telemetry."""

    def __init__(
        self,
        llm: BaseLLMClient,
        *,
        compactor: ContextCompactor | None = None,
        json_repair_retries: int = 1,
        max_calls: int = 12,
        max_wall_seconds: float = 180.0,
    ) -> None:
        self.llm = llm
        self.compactor = compactor or ContextCompactor()
        self.json_repair_retries = max(0, json_repair_retries)
        self.max_calls = max(1, max_calls)
        self.max_wall_seconds = max_wall_seconds
        self.ledger = TraceLedger()
        self._started_at: float | None = None

    def reset_ledger(self) -> None:
        self.ledger = TraceLedger()
        self._started_at = None

    @staticmethod
    def compile_prompt(
        *,
        task_instruction: str,
        context_text: str = "",
        question: str = "",
        output_contract: str = "",
        memory_lines: list[str] | None = None,
    ) -> str:
        """Assemble decoupled prompt sections into a clean, repeatable runtime prompt."""
        parts: list[str] = []
        if memory_lines:
            parts.append("Compacted Prior Reasoning State:\n" + "\n".join(memory_lines))
        if context_text:
            parts.append(context_text)
        if question:
            parts.append(f"Question: {question}")
        if task_instruction:
            parts.append(task_instruction)
        if output_contract:
            parts.append(f"Output Contract:\n{output_contract}")
        return "\n\n".join(parts)

    def _check_guardrails(self, step_name: str) -> None:
        if self._started_at is None:
            self._started_at = time.perf_counter()
        if self.ledger.total_calls >= self.max_calls:
            raise RuntimeError(
                f"LLMHarness guardrail triggered at `{step_name}`: "
                f"max_calls ({self.max_calls}) reached."
            )
        elapsed = time.perf_counter() - self._started_at
        if elapsed > self.max_wall_seconds:
            raise TimeoutError(
                f"LLMHarness guardrail triggered at `{step_name}`: "
                f"elapsed {elapsed:.1f}s > max_wall_seconds ({self.max_wall_seconds}s)."
            )

    def _invoke_raw(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        temperature: float,
        json_mode: bool,
        max_tokens: int | None,
    ) -> str:
        sig = inspect.signature(self.llm.complete)
        kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "temperature": temperature,
            "json_mode": json_mode,
        }
        if max_tokens is not None and "max_tokens" in sig.parameters:
            kwargs["max_tokens"] = max_tokens
        return self.llm.complete(prompt, **kwargs)

    def run_text(
        self,
        prompt: str,
        *,
        step_name: str = "generate",
        system_prompt: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Execute a guarded LLM text completion and log telemetry to `TraceLedger`."""
        self._check_guardrails(step_name)
        t0 = time.perf_counter()
        prompt_toks = estimate_tokens(prompt) + estimate_tokens(system_prompt or "")
        try:
            output = self._invoke_raw(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
            dt = round(time.perf_counter() - t0, 3)
            self.ledger.records.append(
                CallTraceRecord(
                    step_name=step_name,
                    prompt_tokens_est=prompt_toks,
                    response_tokens_est=estimate_tokens(output),
                    wall_seconds=dt,
                    succeeded=bool(output.strip()),
                    error="" if output.strip() else "empty_response",
                )
            )
            return output
        except Exception as exc:
            dt = round(time.perf_counter() - t0, 3)
            self.ledger.records.append(
                CallTraceRecord(
                    step_name=step_name,
                    prompt_tokens_est=prompt_toks,
                    response_tokens_est=0,
                    wall_seconds=dt,
                    succeeded=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

    def run_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        step_name: str = "structured_generate",
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_validator: Callable[[T], str | None] | None = None,
    ) -> tuple[T, str]:
        """Execute a structured JSON LLM call with automatic schema validation & repair retries."""
        self._check_guardrails(step_name)
        schema_hint = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        current_prompt = prompt
        last_raw = ""
        last_err = ""
        repairs = 0
        t0 = time.perf_counter()
        total_prompt_toks = 0
        total_resp_toks = 0

        for attempt in range(self.json_repair_retries + 1):
            if attempt > 0:
                self._check_guardrails(f"{step_name}_repair_{attempt}")
                repairs += 1
                current_prompt = (
                    f"{prompt}\n\n"
                    f"PREVIOUS_INVALID_OUTPUT:\n{last_raw[:400]}\n\n"
                    f"VALIDATION_ERROR: {last_err}\n"
                    f"Repair your output and return ONLY valid JSON matching schema: {schema_hint}"
                )

            total_prompt_toks += estimate_tokens(current_prompt) + estimate_tokens(system_prompt or "")
            raw = self._invoke_raw(
                current_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                json_mode=True,
                max_tokens=max_tokens,
            )
            last_raw = raw
            total_resp_toks += estimate_tokens(raw)

            try:
                parsed_json = json.loads(_extract_json_block(raw))
                validated = response_model.model_validate(parsed_json)
                if extra_validator is not None:
                    custom_err = extra_validator(validated)
                    if custom_err:
                        raise ValueError(custom_err)
                dt = round(time.perf_counter() - t0, 3)
                self.ledger.records.append(
                    CallTraceRecord(
                        step_name=step_name,
                        prompt_tokens_est=total_prompt_toks,
                        response_tokens_est=total_resp_toks,
                        wall_seconds=dt,
                        repair_attempts=repairs,
                        succeeded=True,
                    )
                )
                return validated, raw
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_err = str(exc)[:300]

        dt = round(time.perf_counter() - t0, 3)
        self.ledger.records.append(
            CallTraceRecord(
                step_name=step_name,
                prompt_tokens_est=total_prompt_toks,
                response_tokens_est=total_resp_toks,
                wall_seconds=dt,
                repair_attempts=repairs,
                succeeded=False,
                error=last_err,
            )
        )
        raise ValueError(
            f"LLMHarness `{step_name}` failed after {repairs} repair attempt(s): {last_err}"
        )

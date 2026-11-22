"""Native Ollama `/api/chat` LLM adapter supporting explicit `think`, `num_ctx`, and telemetry."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any


class OllamaNativeLLMClient:
    """Direct Ollama client via `/api/chat` for exact control over `think` and `num_ctx`."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: int = 180,
        think: bool | None = None,
        num_ctx: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/v1"):
            cleaned = cleaned[:-3]
        self.base_url = cleaned
        self.model = model
        self.timeout = timeout
        self.think = think
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.last_telemetry: dict[str, Any] = {}

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        json_mode: bool = False,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {"temperature": temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        effective_max = max_tokens if max_tokens is not None else self.max_tokens
        if effective_max is not None:
            options["num_predict"] = effective_max

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": messages,
        }
        effective_think = self.think if think is None else think
        if effective_think is not None:
            payload["think"] = effective_think
        if json_mode:
            payload["format"] = "json"

        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        wall = time.perf_counter() - t0

        eval_count = int(res.get("eval_count", 0))
        eval_sec = max(res.get("eval_duration", 0) / 1e9, 1e-6)
        self.last_telemetry = {
            "wall_seconds": round(wall, 3),
            "prompt_tokens": int(res.get("prompt_eval_count", 0)),
            "eval_tokens": eval_count,
            "eval_tok_per_s": round(eval_count / eval_sec, 1) if eval_count else 0.0,
            "done_reason": res.get("done_reason", ""),
        }

        msg = res.get("message", {})
        content = str(msg.get("content", "")).strip()
        if not content and msg.get("thinking"):
            # Safe fallback when a thinking model exhausts tokens inside thinking channel.
            content = str(msg["thinking"]).strip()
        return content

    def health_check(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in data.get("models", [])]
            if self.model in names:
                return (
                    True,
                    f"Connected to Ollama native {self.base_url}; model `{self.model}` ready "
                    f"(think={self.think}, num_ctx={self.num_ctx}).",
                )
            return (
                True,
                f"Connected to Ollama native {self.base_url} (models: {', '.join(names[:8])}). "
                f"Note: `{self.model}` not listed.",
            )
        except Exception as exc:
            return False, f"Failed to reach Ollama {self.base_url}: {type(exc).__name__}: {exc}"

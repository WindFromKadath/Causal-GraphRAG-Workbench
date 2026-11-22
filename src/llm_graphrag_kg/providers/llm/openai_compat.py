"""LLM client adapters (OpenAI-compatible API / local Magpie / Ollama + offline Mock)."""

from __future__ import annotations

from openai import OpenAI


class OpenAICompatLLMClient:
    """Adapter for any OpenAI-compatible chat completion API (cloud or local Magpie/Ollama/vLLM)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = OpenAI(
            api_key=api_key or "local-no-key",
            base_url=self.base_url,
            timeout=timeout,
        )

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        except Exception:
            if not json_mode:
                raise
            # Some local OpenAI-compatible servers do not support response_format=json_object.
            kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]

        message = response.choices[0].message
        content = (message.content or "").strip()
        if not content:
            # Fallback for reasoning models that emit text in `reasoning` / `reasoning_content`.
            for attr in ("reasoning_content", "reasoning"):
                extra = getattr(message, attr, None)
                if extra:
                    content = str(extra).strip()
                    break
        return content

    def health_check(self) -> tuple[bool, str]:
        """Verify endpoint availability by listing models or issuing a minimal completion."""
        try:
            models = self._client.models.list()
            names = [m.id for m in models.data]
            if not names:
                return True, f"Connected to {self.base_url}, endpoint returned empty model list."
            if self.model in names:
                return True, f"Connected to {self.base_url}; target model `{self.model}` is available."
            preview = ", ".join(names[:8])
            return (
                True,
                f"Connected to {self.base_url} (available models: {preview}). "
                f"Note: configured model `{self.model}` was not in the listed IDs.",
            )
        except Exception as exc:
            return False, f"Failed to reach {self.base_url}: {type(exc).__name__}: {exc}"

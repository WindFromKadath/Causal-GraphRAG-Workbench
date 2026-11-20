"""OpenAI-compatible embedding client."""

from __future__ import annotations

from openai import OpenAI


class OpenAICompatEmbeddingClient:
    """Adapter for any OpenAI-compatible `/v1/embeddings` endpoint."""

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
        self._client = OpenAI(
            api_key=api_key or "local-no-key",
            base_url=self.base_url,
            timeout=timeout,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

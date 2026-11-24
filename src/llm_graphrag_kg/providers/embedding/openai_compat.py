"""OpenAI-compatible embedding client with chunked batching and optional disk caching."""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from llm_graphrag_kg.providers.embedding.ollama_native import EmbeddingDiskCache


class OpenAICompatEmbeddingClient:
    """Adapter for any OpenAI-compatible `/v1/embeddings` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 120,
        batch_size: int = 32,
        cache_dir: Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.cache = EmbeddingDiskCache(cache_dir / "embed_cache.sqlite3") if cache_dir else None
        self._client = OpenAI(
            api_key=api_key or "local-no-key",
            base_url=self.base_url,
            timeout=timeout,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        normalized = [t if t.strip() else " " for t in texts]
        cached_by_key: dict[str, list[float]] = {}
        if self.cache is not None:
            cached_by_key = self.cache.get_many(self.model, normalized)

        missing: list[str] = []
        seen_missing: set[str] = set()
        for txt in normalized:
            key = EmbeddingDiskCache.make_key(self.model, txt)
            if key not in cached_by_key and key not in seen_missing:
                seen_missing.add(key)
                missing.append(txt)

        newly_computed: list[tuple[str, list[float]]] = []
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            for txt, item in zip(batch, ordered, strict=True):
                vec = [float(x) for x in item.embedding]
                cached_by_key[EmbeddingDiskCache.make_key(self.model, txt)] = vec
                newly_computed.append((txt, vec))

        if self.cache is not None and newly_computed:
            self.cache.put_many(self.model, newly_computed)

        return [cached_by_key[EmbeddingDiskCache.make_key(self.model, txt)] for txt in normalized]

    def health_check(self) -> tuple[bool, str]:
        try:
            vecs = self.embed(["graphrag health check"])
            dim = len(vecs[0]) if vecs else 0
            return True, f"OpenAI-compat embedder ready: model={self.model} dim={dim} url={self.base_url}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Embedding health check failed ({self.model}): {exc}"

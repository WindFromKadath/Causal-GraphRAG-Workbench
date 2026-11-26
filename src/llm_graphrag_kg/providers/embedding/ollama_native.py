"""Native Ollama `/api/embed` client with chunked batching and SQLite content-hash caching."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_ollama_root(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    for suffix in ("/v1/embeddings", "/v1", "/api/embed", "/api/embeddings", "/api"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url or "http://127.0.0.1:11434"


class EmbeddingDiskCache:
    """Zero-dependency SQLite content-hash cache for dense embedding vectors."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.cache_path), timeout=30.0)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        cache_key TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        dim INTEGER NOT NULL,
                        vec BLOB NOT NULL
                    )
                    """
                )

    @staticmethod
    def make_key(model: str, text: str) -> str:
        payload = f"{model}\x00{text}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> dict[str, list[float]]:
        if not texts:
            return {}
        keys = [self.make_key(model, t) for t in texts]
        unique_keys = list(dict.fromkeys(keys))
        found: dict[str, list[float]] = {}
        with closing(self._connect()) as conn:
            # Query in slices to stay well below SQLite variable limits
            for start in range(0, len(unique_keys), 400):
                batch_keys = unique_keys[start : start + 400]
                placeholders = ",".join("?" for _ in batch_keys)
                rows = conn.execute(
                    f"SELECT cache_key, vec FROM embeddings WHERE cache_key IN ({placeholders})",
                    batch_keys,
                ).fetchall()
                for ck, blob in rows:
                    arr = np.frombuffer(blob, dtype=np.float32)
                    found[ck] = arr.tolist()
        return found

    def put_many(self, model: str, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        rows = []
        for text, vec in items:
            arr = np.asarray(vec, dtype=np.float32)
            rows.append((self.make_key(model, text), model, int(arr.size), arr.tobytes()))
        with closing(self._connect()) as conn:
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (cache_key, model, dim, vec) VALUES (?, ?, ?, ?)",
                    rows,
                )


class OllamaNativeEmbeddingClient:
    """Adapter calling Ollama native `/api/embed` with Fix A (`num_ctx`) & Fix B (`num_gpu`) controls."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3-embedding:0.6b",
        timeout: int = 120,
        batch_size: int = 32,
        truncate: bool = True,
        num_gpu: int | None = 0,
        num_ctx: int | None = 2048,
        dimensions: int | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.root_url = _normalize_ollama_root(base_url)
        self.endpoint = f"{self.root_url}/api/embed"
        self.model = model
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        self.truncate = truncate
        self.num_gpu = num_gpu
        self.num_ctx = num_ctx
        self.dimensions = dimensions if (dimensions is not None and dimensions > 0) else None
        self.cache = EmbeddingDiskCache(cache_dir / "embed_cache.sqlite3") if cache_dir else None
        self.last_telemetry: dict[str, Any] = {}

    @property
    def _cache_model_key(self) -> str:
        return f"{self.model}@d{self.dimensions}" if self.dimensions else self.model

    def _apply_mrl_truncate(self, vec: list[float]) -> list[float]:
        if self.dimensions is None or len(vec) <= self.dimensions:
            return vec
        arr = np.asarray(vec[: self.dimensions], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-12:
            arr = arr / norm
        return [float(x) for x in arr.tolist()]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        t0 = time.perf_counter()
        normalized_texts = [t if t.strip() else " " for t in texts]
        cm_key = self._cache_model_key
        cached_by_key: dict[str, list[float]] = {}
        if self.cache is not None:
            cached_by_key = self.cache.get_many(cm_key, normalized_texts)

        # Deduplicate uncached texts so identical strings are embedded once
        missing_texts: list[str] = []
        seen_missing: set[str] = set()
        for txt in normalized_texts:
            key = EmbeddingDiskCache.make_key(cm_key, txt)
            if key not in cached_by_key and key not in seen_missing:
                seen_missing.add(key)
                missing_texts.append(txt)

        api_calls = 0
        newly_computed: list[tuple[str, list[float]]] = []
        for start in range(0, len(missing_texts), self.batch_size):
            batch = missing_texts[start : start + self.batch_size]
            vecs = self._embed_batch(batch)
            api_calls += 1
            for txt, vec in zip(batch, vecs, strict=True):
                key = EmbeddingDiskCache.make_key(cm_key, txt)
                cached_by_key[key] = vec
                newly_computed.append((txt, vec))

        if self.cache is not None and newly_computed:
            self.cache.put_many(cm_key, newly_computed)

        results = [
            cached_by_key[EmbeddingDiskCache.make_key(cm_key, txt)] for txt in normalized_texts
        ]
        dim = len(results[0]) if results else 0
        self.last_telemetry = {
            "model": self.model,
            "num_gpu": self.num_gpu,
            "num_ctx": self.num_ctx,
            "total_texts": len(texts),
            "cache_hits": len(texts) - len(missing_texts),
            "api_calls": api_calls,
            "dim": dim,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }
        return results

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": batch,
            "truncate": self.truncate,
        }
        options: dict[str, Any] = {}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if options:
            payload["options"] = options

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            hint = (
                f" Hint: run `ollama pull {self.model}` first."
                if exc.code == 404 or "not found" in body.lower()
                else ""
            )
            raise RuntimeError(
                f"Ollama /api/embed HTTP {exc.code} for model `{self.model}`: {body[:300]}{hint}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama embedding endpoint at {self.endpoint}: {exc}"
            ) from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise RuntimeError(
                f"Unexpected Ollama /api/embed response: expected {len(batch)} vectors, "
                f"got {type(embeddings).__name__}."
            )
        return [self._apply_mrl_truncate([float(x) for x in vec]) for vec in embeddings]

    def health_check(self) -> tuple[bool, str]:
        try:
            vecs = self._embed_batch(["graphrag health check"])
            dim = len(vecs[0]) if vecs else 0
            return True, (
                f"Ollama native embedder ready: model={self.model} dim={dim} "
                f"num_gpu={self.num_gpu} num_ctx={self.num_ctx} endpoint={self.endpoint}"
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"Embedding health check failed ({self.model}): {exc}"

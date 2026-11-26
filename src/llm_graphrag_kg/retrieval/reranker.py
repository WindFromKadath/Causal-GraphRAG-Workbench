"""Stage 4.3 Local ONNX/CPU Cross-Encoder Reranker + Redis Rerank Cache (Zero 8 GB VRAM Contention).

Runs quantized INT8 ONNX cross-encoders (e.g. `BAAI/bge-reranker-v2-m3` ONNX / `Xenova/bge-reranker-base`)
strictly on `CPUExecutionProvider` via `onnxruntime` + Rust `tokenizers` (zero PyTorch, zero GPU VRAM!),
so `qwen3.5:9b @ 16k` never suffers VRAM eviction swaps on 8 GB GPUs.
Includes deterministic bilingual lexical+dense fusion fallback (`hybrid_lexical`) for offline CI/tests
and optional Redis result caching (`graphrag:cache:{tenant_id}:rerank:*`).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from llm_graphrag_kg.core.schema import TextChunk
from llm_graphrag_kg.providers.storage.local_vector import tokenize_bm25


class RerankCandidate(BaseModel):
    chunk: TextChunk
    coarse_score: float = 0.0
    rerank_score: float = 0.0
    coarse_rank: int = 0
    final_rank: int = 0


class RerankResult(BaseModel):
    backend_used: str  # onnx_cpu | hybrid_lexical | passthrough
    model_name: str
    input_count: int
    output_count: int
    cache_hit: bool = False
    latency_ms: float = 0.0
    candidates: list[RerankCandidate] = Field(default_factory=list)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _lexical_cross_score(query: str, text: str, coarse_score: float) -> float:
    """Fast bilingual query-document cross-relevance score (offline fallback, zero model download)."""
    q_toks = tokenize_bm25(query)
    if not q_toks:
        return round(max(0.0, min(1.0, coarse_score)), 6)
    d_toks = set(tokenize_bm25(text))
    q_unique = list(dict.fromkeys(q_toks))
    exact_hits = sum(1 for t in q_unique if t in d_toks)
    coverage = exact_hits / max(1, len(q_unique))

    # Bigram / phrase proximity bonus
    low_text = text.lower()
    phrase_bonus = 0.0
    long_toks = [t for t in q_unique if len(t) >= 2]
    if long_toks:
        substr_hits = sum(1 for t in long_toks if t in low_text)
        phrase_bonus = 0.25 * (substr_hits / len(long_toks))

    # Normalize coarse RRF/cosine score smoothly into [0, 1]
    norm_coarse = 1.0 - math.exp(-max(0.0, coarse_score) * 20.0) if coarse_score < 0.1 else min(1.0, coarse_score)
    raw = 0.55 * coverage + 0.25 * norm_coarse + phrase_bonus
    return round(max(0.0, min(1.0, raw)), 6)


class CrossEncoderReranker:
    """Enterprise ONNX/CPU Cross-Encoder Reranker with Redis cache and deterministic fallback."""

    def __init__(
        self,
        *,
        backend: str = "auto",  # auto | onnx | hybrid_lexical | off
        model_name: str = "Xenova/bge-reranker-base",
        onnx_file: str = "onnx/model_quantized.onnx",
        model_dir: Path | None = None,
        max_length: int = 512,
        redis_url: str | None = None,
        redis_client: Any = None,
        cache_ttl_seconds: int = 600,
        auto_download: bool = False,
    ) -> None:
        self.backend = backend.strip().lower()
        self.model_name = model_name
        self.onnx_file = onnx_file
        self.model_dir = model_dir
        self.max_length = max(64, int(max_length))
        self.redis_url = redis_url
        self._redis = redis_client
        self.cache_ttl_seconds = cache_ttl_seconds
        self.auto_download = auto_download

        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: set[str] = set()
        self._onnx_ready = False
        self._onnx_error: str | None = None

        if self.backend in {"onnx", "auto"}:
            self._try_load_onnx(allow_download=(self.backend == "onnx" or self.auto_download))

    def _resolve_onnx_files(self, *, allow_download: bool) -> tuple[Path, Path] | None:
        """Locate `(model_onnx_path, tokenizer_json_path)` locally or via HuggingFace Hub."""
        # 1. Explicit local directory
        if self.model_dir is not None and self.model_dir.is_dir():
            tok = self.model_dir / "tokenizer.json"
            for cand in (
                self.model_dir / self.onnx_file,
                self.model_dir / "model_quantized.onnx",
                self.model_dir / "model_int8.onnx",
                self.model_dir / "model.onnx",
            ):
                if cand.is_file() and tok.is_file():
                    return cand, tok

        # 2. HuggingFace Hub cache (local_files_only first unless allow_download=True)
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        except ImportError:
            return None

        local_only = not allow_download
        try:
            tok_path = Path(
                hf_hub_download(
                    repo_id=self.model_name,
                    filename="tokenizer.json",
                    local_files_only=local_only,
                )
            )
            for fname in dict.fromkeys(
                [self.onnx_file, "onnx/model_quantized.onnx", "onnx/model_int8.onnx", "onnx/model.onnx", "model.onnx"]
            ):
                try:
                    onnx_path = Path(
                        hf_hub_download(
                            repo_id=self.model_name,
                            filename=fname,
                            local_files_only=local_only,
                        )
                    )
                    return onnx_path, tok_path
                except Exception:  # noqa: BLE001
                    continue
        except Exception as exc:  # noqa: BLE001
            self._onnx_error = str(exc)
        return None

    def _try_load_onnx(self, *, allow_download: bool) -> None:
        files = self._resolve_onnx_files(allow_download=allow_download)
        if files is None:
            return
        onnx_path, tok_path = files
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
            from tokenizers import Tokenizer  # type: ignore[import-not-found]

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Pure CPU execution provider: ZERO 8 GB GPU VRAM contention!
            self._session = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = {inp.name for inp in self._session.get_inputs()}
            tok = Tokenizer.from_file(str(tok_path))
            tok.enable_truncation(max_length=self.max_length)
            tok.enable_padding(pad_id=1, pad_token="<pad>")
            self._tokenizer = tok
            self._onnx_ready = True
        except Exception as exc:  # noqa: BLE001
            self._onnx_error = str(exc)
            self._onnx_ready = False

    def _score_onnx_cpu(self, query: str, texts: list[str]) -> list[float]:
        assert self._session is not None and self._tokenizer is not None
        pairs = [(query, t[:2000]) for t in texts]
        encodings = self._tokenizer.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        ort_inputs: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            ort_inputs["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            ort_inputs["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            ort_inputs["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        outputs = self._session.run(None, ort_inputs)
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim > 1:
            logits = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
        probs = _sigmoid(logits)
        return [round(float(p), 6) for p in probs.tolist()]

    def _get_redis(self) -> Any:
        if self._redis is not None:
            return self._redis
        if not self.redis_url:
            return None
        try:
            import redis  # type: ignore[import-not-found]

            self._redis = redis.from_url(self.redis_url, decode_responses=True, socket_connect_timeout=1)
            return self._redis
        except Exception:  # noqa: BLE001
            return None

    def _cache_key(self, tenant_id: str, query: str, chunk_ids: list[str], backend_tag: str) -> str:
        raw = f"{backend_tag}|{self.model_name}|{query.strip()}|{'|'.join(chunk_ids)}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"graphrag:cache:{tenant_id}:rerank:{digest}"

    def rerank(
        self,
        query: str,
        candidates: list[tuple[TextChunk, float]],
        *,
        top_n: int = 8,
        tenant_id: str = "default",
    ) -> RerankResult:
        """Rerank coarse candidates (e.g. Top-40 -> Top-8) using ONNX CPU or lexical fallback."""
        t0 = time.perf_counter()
        if not candidates:
            return RerankResult(
                backend_used="passthrough",
                model_name=self.model_name,
                input_count=0,
                output_count=0,
            )

        if self.backend == "off":
            wrapped = [
                RerankCandidate(
                    chunk=ch,
                    coarse_score=round(float(sc), 6),
                    rerank_score=round(float(sc), 6),
                    coarse_rank=idx,
                    final_rank=idx,
                )
                for idx, (ch, sc) in enumerate(candidates[:top_n], start=1)
            ]
            return RerankResult(
                backend_used="passthrough",
                model_name="off",
                input_count=len(candidates),
                output_count=len(wrapped),
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                candidates=wrapped,
            )

        backend_tag = "onnx_cpu" if self._onnx_ready else "hybrid_lexical"
        chunk_ids = [ch.chunk_id for ch, _ in candidates]
        cache_key = self._cache_key(tenant_id, query, chunk_ids, backend_tag)

        # Check Redis rerank cache
        r = self._get_redis()
        cached_scores: list[float] | None = None
        if r is not None:
            try:
                raw_hit = r.get(cache_key)
                if raw_hit:
                    parsed = json.loads(raw_hit)
                    if isinstance(parsed, list) and len(parsed) == len(candidates):
                        cached_scores = [float(x) for x in parsed]
            except Exception:  # noqa: BLE001
                cached_scores = None

        cache_hit = cached_scores is not None
        if cached_scores is not None:
            scores = cached_scores
        elif self._onnx_ready:
            texts = [str(ch.metadata.get("breadcrumb_text") or ch.text) for ch, _ in candidates]
            scores = self._score_onnx_cpu(query, texts)
        else:
            scores = [
                _lexical_cross_score(query, str(ch.metadata.get("breadcrumb_text") or ch.text), sc)
                for ch, sc in candidates
            ]

        if not cache_hit and r is not None:
            try:
                r.setex(cache_key, self.cache_ttl_seconds, json.dumps(scores))
            except Exception:  # noqa: BLE001
                pass

        scored_items = [
            RerankCandidate(
                chunk=ch,
                coarse_score=round(float(c_sc), 6),
                rerank_score=round(float(r_sc), 6),
                coarse_rank=idx,
            )
            for idx, ((ch, c_sc), r_sc) in enumerate(zip(candidates, scores, strict=True), start=1)
        ]
        scored_items.sort(key=lambda item: (-item.rerank_score, item.coarse_rank))
        top_items = scored_items[:top_n]
        for rank, item in enumerate(top_items, start=1):
            item.final_rank = rank

        return RerankResult(
            backend_used=backend_tag,
            model_name=self.model_name if backend_tag == "onnx_cpu" else "bilingual_lexical_cross",
            input_count=len(candidates),
            output_count=len(top_items),
            cache_hit=cache_hit,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            candidates=top_items,
        )


def create_reranker(
    *,
    backend: str = "auto",
    model_name: str = "Xenova/bge-reranker-base",
    onnx_file: str = "onnx/model_quantized.onnx",
    model_dir: str | Path | None = None,
    max_length: int = 512,
    redis_url: str | None = None,
    auto_download: bool = False,
) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        backend=backend,
        model_name=model_name,
        onnx_file=onnx_file,
        model_dir=Path(model_dir) if model_dir else None,
        max_length=max_length,
        redis_url=redis_url,
        auto_download=auto_download,
    )

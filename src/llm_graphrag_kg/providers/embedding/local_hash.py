"""Deterministic offline hashing embedding supporting both Chinese and English text."""

from __future__ import annotations

import hashlib
import re

import numpy as np


class LocalHashEmbeddingClient:
    """Zero-dependency hashed n-gram embedding for offline testing and lightweight retrieval."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        normalized = text.lower().strip()
        if not normalized:
            return vec.tolist()

        features: list[str] = []
        # English / alphanumeric word tokens
        features.extend(re.findall(r"[a-z0-9_]+", normalized))
        # Chinese character unigrams & bigrams
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
        features.extend(cjk_chars)
        features.extend(
            cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)
        )

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

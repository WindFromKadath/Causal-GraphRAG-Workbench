"""Lightweight NumPy + pure-Python Okapi BM25 / RRF hybrid vector store."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from llm_graphrag_kg.core.schema import TextChunk

_BM25_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "into", "what", "when", "where", "which",
    "who", "why", "how", "can", "may", "will", "should", "could", "would",
}


def tokenize_bm25(text: str) -> list[str]:
    """Bilingual tokenizer producing lowercase alphanumeric terms plus CJK unigrams & bigrams."""
    lowered = text.lower().strip()
    if not lowered:
        return []
    tokens: list[str] = [
        tok
        for tok in re.findall(r"[a-z0-9_]+", lowered)
        if len(tok) >= 2 and tok not in _BM25_STOPWORDS
    ]
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    if cjk_chars:
        tokens.extend(cjk_chars)
        tokens.extend(cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1))
    return tokens


class LocalVectorStore:
    """Zero-dependency Dense + Okapi BM25 + RRF hybrid vector store implementing `BaseVectorStore`."""

    def __init__(self, *, bm25_k1: float = 1.5, bm25_b: float = 0.75) -> None:
        self._chunks: list[TextChunk] = []
        self._chunk_map: dict[str, TextChunk] = {}
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b

        # Lazy BM25 inverted index structures
        self._bm25_dirty: bool = True
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_len: float = 1.0
        self._idf: dict[str, float] = {}
        self._inverted_index: dict[str, list[int]] = {}

    @property
    def vector_dim(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.ndim == 2 and self._vectors.size > 0 else 0

    def add_chunks(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")
        if not chunks:
            return

        new_mat = np.asarray(vectors, dtype=np.float32)
        if new_mat.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {new_mat.shape}.")
        norms = np.linalg.norm(new_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_mat = new_mat / norms

        if self._vectors.size == 0:
            self._vectors = new_mat
        else:
            if self._vectors.shape[1] != new_mat.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: existing={self._vectors.shape[1]}, "
                    f"new={new_mat.shape[1]}."
                )
            self._vectors = np.vstack([self._vectors, new_mat])

        self._chunks.extend(chunks)
        for chunk in chunks:
            self._chunk_map[chunk.chunk_id] = chunk
        self._bm25_dirty = True

    def _ensure_bm25_index(self) -> None:
        if not self._bm25_dirty:
            return
        self._doc_term_freqs = []
        self._doc_lengths = []
        self._inverted_index = {}
        df: Counter[str] = Counter()

        for idx, chunk in enumerate(self._chunks):
            toks = tokenize_bm25(chunk.text)
            tf = Counter(toks)
            self._doc_term_freqs.append(tf)
            self._doc_lengths.append(len(toks))
            for term in tf:
                df[term] += 1
                self._inverted_index.setdefault(term, []).append(idx)

        n_docs = len(self._chunks)
        self._avg_doc_len = (sum(self._doc_lengths) / n_docs) if n_docs > 0 else 1.0
        if self._avg_doc_len <= 0.0:
            self._avg_doc_len = 1.0
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._bm25_dirty = False

    def search(self, query_vector: list[float], top_k: int = 4) -> list[tuple[TextChunk, float]]:
        """Pure dense cosine similarity search (backward-compatible BaseVectorStore contract)."""
        if not self._chunks or self._vectors.size == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"Query embedding dim ({q.shape[0]}) != persisted vector store dim "
                f"({self._vectors.shape[1]}). Please rebuild or re-ingest the index with "
                f"the active embedding provider/model."
            )
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm

        scores = self._vectors @ q
        k = min(max(1, top_k), len(self._chunks))
        top_indices = np.argsort(-scores)[:k]
        return [(self._chunks[int(idx)], float(scores[int(idx)])) for idx in top_indices]

    def search_bm25(self, query_text: str, top_k: int = 10) -> list[tuple[TextChunk, float]]:
        """Pure-Python Okapi BM25 lexical search over persisted chunks."""
        if not self._chunks:
            return []
        self._ensure_bm25_index()
        q_tokens = tokenize_bm25(query_text)
        if not q_tokens:
            return []

        scores: dict[int, float] = {}
        k1 = self._bm25_k1
        b = self._bm25_b
        avg_dl = self._avg_doc_len

        for term in dict.fromkeys(q_tokens):
            postings = self._inverted_index.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_idx in postings:
                tf = self._doc_term_freqs[doc_idx][term]
                dl = self._doc_lengths[doc_idx]
                denom = tf + k1 * (1.0 - b + b * (dl / avg_dl))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (tf * (k1 + 1.0)) / denom

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, top_k)]
        return [(self._chunks[idx], float(score)) for idx, score in ranked]

    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 4,
        *,
        bm25_weight: float = 0.35,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> list[tuple[TextChunk, float]]:
        """Weighted Reciprocal Rank Fusion (RRF) combining dense cosine and Okapi BM25."""
        if not self._chunks or self._vectors.size == 0:
            return []

        w_bm25 = min(max(float(bm25_weight), 0.0), 1.0)
        w_dense = 1.0 - w_bm25
        if w_bm25 == 0.0:
            return self.search(query_vector, top_k=top_k)

        pool_k = min(len(self._chunks), max(top_k * candidate_multiplier, top_k + 12))
        dense_hits = self.search(query_vector, top_k=pool_k)
        bm25_hits = self.search_bm25(query_text, top_k=pool_k)

        if not bm25_hits:
            return dense_hits[:top_k]

        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, TextChunk] = {}

        for rank, (chunk, _score) in enumerate(dense_hits, start=1):
            cid = chunk.chunk_id
            chunk_lookup[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_dense / (rrf_k + rank)

        for rank, (chunk, _score) in enumerate(bm25_hits, start=1):
            cid = chunk.chunk_id
            chunk_lookup[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_bm25 / (rrf_k + rank)

        ranked_ids = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, top_k)]
        # Scale RRF scores into a readable [0, 1] range relative to theoretical rank-1 max
        max_possible = 1.0 / (rrf_k + 1)
        return [
            (chunk_lookup[cid], round(min(1.0, score / max_possible), 4))
            for cid, score in ranked_ids
        ]

    def get_chunk(self, chunk_id: str) -> TextChunk | None:
        return self._chunk_map.get(chunk_id)

    def get_all_chunks(self) -> list[TextChunk]:
        return list(self._chunks)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self._chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(directory / "embeddings.npy", self._vectors)
        meta = {
            "chunk_count": len(self._chunks),
            "vector_dim": self.vector_dim,
        }
        (directory / "vector_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, directory: Path) -> None:
        chunks_file = directory / "chunks.json"
        vec_file = directory / "embeddings.npy"
        if not chunks_file.is_file() or not vec_file.is_file():
            raise FileNotFoundError(
                f"Vector store not found under {directory}. Please run `build` first."
            )
        raw_chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        self._chunks = [TextChunk(**item) for item in raw_chunks]
        self._chunk_map = {c.chunk_id: c for c in self._chunks}
        self._vectors = np.load(vec_file)
        self._bm25_dirty = True

"""HippoRAG 2 semantic graph enhancements: SYNONYM_OF auto-linking & Query-to-Triple indexing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from llm_graphrag_kg.core.interfaces import BaseEmbeddingClient, BaseGraphStore
from llm_graphrag_kg.core.schema import Entity, Relation
from llm_graphrag_kg.pipelines.extraction import make_relation_id

_BOILERPLATE_DESC_PAT = re.compile(
    r"^(?:Corpus entity `.*` extracted from GraphRAG-Bench|Co-occurs in chunk )",
    flags=re.IGNORECASE,
)


def entity_embedding_text(entity: Entity) -> str:
    """Build clean semantic text for an entity without boilerplate inflation."""
    parts = [entity.name.strip()]
    if entity.aliases:
        parts.append("aliases: " + ", ".join(entity.aliases[:4]))
    desc = entity.description.strip()
    if desc and not _BOILERPLATE_DESC_PAT.match(desc):
        parts.append(desc[:180])
    return " | ".join(p for p in parts if p)


def triple_embedding_text(rel: Relation, ent_map: dict[str, Entity]) -> str:
    """Build concise semantic verbalization for Query-to-Triple retrieval."""
    src = ent_map.get(rel.source_id)
    tgt = ent_map.get(rel.target_id)
    s_name = src.name if src else rel.source_id
    t_name = tgt.name if tgt else rel.target_id
    rel_label = rel.relation_type.replace("_", " ").lower()
    base = f"{s_name} {rel_label} {t_name}"
    desc = rel.description.strip()
    if desc and not _BOILERPLATE_DESC_PAT.match(desc):
        return f"{base}: {desc[:180]}"
    return base


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32)
    arr = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def build_synonym_edges(
    graph_store: BaseGraphStore,
    embedder: BaseEmbeddingClient,
    *,
    cosine_threshold: float = 0.85,
    max_synonyms_per_entity: int = 5,
) -> tuple[int, list[str], np.ndarray]:
    """Add HippoRAG 2 `SYNONYM_OF` edges between semantically close entities (cos >= threshold).

    Returns `(added_edge_count, entity_ids, normalized_entity_vectors)` so entity embeddings
    can be reused directly by `SemanticGraphIndex` without re-encoding.
    """
    entities = graph_store.get_all_entities()
    if not entities:
        return 0, [], np.empty((0, 0), dtype=np.float32)

    ent_ids = [e.entity_id for e in entities]
    texts = [entity_embedding_text(e) for e in entities]
    raw_vecs = embedder.embed(texts)
    ent_mat = _l2_normalize(np.asarray(raw_vecs, dtype=np.float32))

    if len(entities) < 2 or cosine_threshold > 1.0:
        return 0, ent_ids, ent_mat

    sim_matrix = ent_mat @ ent_mat.T
    # Zero out lower triangle and diagonal so each undirected pair is considered once
    np.fill_diagonal(sim_matrix, -1.0)
    upper_i, upper_j = np.where(np.triu(sim_matrix, k=1) >= cosine_threshold)

    if upper_i.size == 0:
        return 0, ent_ids, ent_mat

    candidates: list[tuple[float, int, int]] = [
        (float(sim_matrix[int(i), int(j)]), int(i), int(j))
        for i, j in zip(upper_i, upper_j, strict=True)
    ]
    candidates.sort(key=lambda item: -item[0])

    degree_counts: dict[str, int] = {}
    existing_rel_ids = {r.relation_id for r in graph_store.get_all_relations()}
    added = 0

    for sim, i, j in candidates:
        e1 = entities[i]
        e2 = entities[j]
        if e1.name.strip().lower() == e2.name.strip().lower():
            continue
        if (
            degree_counts.get(e1.entity_id, 0) >= max_synonyms_per_entity
            or degree_counts.get(e2.entity_id, 0) >= max_synonyms_per_entity
        ):
            continue

        s_id, t_id = sorted((e1.entity_id, e2.entity_id))
        rid = make_relation_id(s_id, "SYNONYM_OF", t_id)
        if rid in existing_rel_ids:
            continue

        # Attach valid source chunks from endpoints so SHACL-lite provenance audit stays 100% PASS
        src_chunks = list(dict.fromkeys([*e1.source_chunk_ids[:1], *e2.source_chunk_ids[:1]]))
        rel = Relation(
            relation_id=rid,
            source_id=s_id,
            target_id=t_id,
            relation_type="SYNONYM_OF",
            description=(
                f"Semantic synonym link (cosine={sim:.3f}) between `{e1.name}` and `{e2.name}`."
            ),
            weight=round(sim, 3),
            source_chunk_ids=src_chunks,
            metadata={"auto_synonym": True, "cosine_sim": round(sim, 4)},
        )
        graph_store.upsert_relation(rel)
        existing_rel_ids.add(rid)
        degree_counts[e1.entity_id] = degree_counts.get(e1.entity_id, 0) + 1
        degree_counts[e2.entity_id] = degree_counts.get(e2.entity_id, 0) + 1
        added += 1

    return added, ent_ids, ent_mat


class SemanticGraphIndex:
    """Persisted dense index over KG entities and triples for HippoRAG 2 Query-to-Triple."""

    def __init__(self) -> None:
        self.entity_ids: list[str] = []
        self.entity_vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.triple_ids: list[str] = []
        self.triple_endpoints: list[tuple[str, str]] = []
        self.triple_texts: list[str] = []
        self.triple_vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    @property
    def is_ready(self) -> bool:
        return bool(self.entity_ids and self.entity_vectors.size > 0)

    @property
    def has_triples(self) -> bool:
        return bool(self.triple_ids and self.triple_vectors.size > 0)

    def build(
        self,
        graph_store: BaseGraphStore,
        embedder: BaseEmbeddingClient,
        *,
        precomputed_entities: tuple[list[str], np.ndarray] | None = None,
        max_indexed_triples: int = 3000,
    ) -> dict[str, int]:
        """Encode entities and top relations into normalized NumPy arrays."""
        all_entities = graph_store.get_all_entities()
        ent_map = {e.entity_id: e for e in all_entities}

        if precomputed_entities is not None and len(precomputed_entities[0]) == len(all_entities):
            self.entity_ids = list(precomputed_entities[0])
            self.entity_vectors = _l2_normalize(precomputed_entities[1])
        elif all_entities:
            self.entity_ids = [e.entity_id for e in all_entities]
            e_texts = [entity_embedding_text(e) for e in all_entities]
            self.entity_vectors = _l2_normalize(np.asarray(embedder.embed(e_texts), dtype=np.float32))
        else:
            self.entity_ids = []
            self.entity_vectors = np.empty((0, 0), dtype=np.float32)

        all_relations = [
            r
            for r in graph_store.get_all_relations()
            if r.source_id in ent_map and r.target_id in ent_map and r.relation_type != "SYNONYM_OF"
        ]
        # Prioritize typed semantic edges over generic CO_OCCURS_WITH, then higher weight
        all_relations.sort(
            key=lambda r: (
                0 if r.relation_type != "CO_OCCURS_WITH" else 1,
                -float(r.weight),
                r.relation_id,
            )
        )
        kept_rels = all_relations[: max(0, max_indexed_triples)]

        if kept_rels:
            self.triple_ids = [r.relation_id for r in kept_rels]
            self.triple_endpoints = [(r.source_id, r.target_id) for r in kept_rels]
            self.triple_texts = [triple_embedding_text(r, ent_map) for r in kept_rels]
            self.triple_vectors = _l2_normalize(
                np.asarray(embedder.embed(self.triple_texts), dtype=np.float32)
            )
        else:
            self.triple_ids = []
            self.triple_endpoints = []
            self.triple_texts = []
            self.triple_vectors = np.empty((0, 0), dtype=np.float32)

        return {
            "indexed_entities": len(self.entity_ids),
            "indexed_triples": len(self.triple_ids),
        }

    def search_entities(
        self,
        query_vector: list[float],
        *,
        top_k: int = 6,
        min_score: float = 0.25,
    ) -> list[tuple[str, float]]:
        if not self.is_ready:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape[0] != self.entity_vectors.shape[1]:
            return []
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        scores = self.entity_vectors @ q
        k = min(max(1, top_k), len(self.entity_ids))
        top_idx = np.argsort(-scores)[:k]
        hits: list[tuple[str, float]] = []
        for idx in top_idx:
            sc = float(scores[int(idx)])
            if sc >= min_score:
                hits.append((self.entity_ids[int(idx)], round(sc, 4)))
        return hits

    def search_triples(
        self,
        query_vector: list[float],
        *,
        top_k: int = 8,
        min_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        if not self.has_triples:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape[0] != self.triple_vectors.shape[1]:
            return []
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        scores = self.triple_vectors @ q
        k = min(max(1, top_k), len(self.triple_ids))
        top_idx = np.argsort(-scores)[:k]
        hits: list[dict[str, Any]] = []
        for idx in top_idx:
            sc = float(scores[int(idx)])
            if sc >= min_score:
                src_id, tgt_id = self.triple_endpoints[int(idx)]
                hits.append(
                    {
                        "relation_id": self.triple_ids[int(idx)],
                        "source_id": src_id,
                        "target_id": tgt_id,
                        "text": self.triple_texts[int(idx)],
                        "score": round(sc, 4),
                    }
                )
        return hits

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "entity_ids": self.entity_ids,
            "triple_ids": self.triple_ids,
            "triple_endpoints": [list(pair) for pair in self.triple_endpoints],
            "triple_texts": self.triple_texts,
        }
        (directory / "semantic_index_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez(
            directory / "semantic_index.npz",
            entity_vectors=self.entity_vectors,
            triple_vectors=self.triple_vectors,
        )

    def load(self, directory: Path) -> bool:
        meta_file = directory / "semantic_index_meta.json"
        vec_file = directory / "semantic_index.npz"
        if not meta_file.is_file() or not vec_file.is_file():
            self.entity_ids = []
            self.entity_vectors = np.empty((0, 0), dtype=np.float32)
            self.triple_ids = []
            self.triple_endpoints = []
            self.triple_texts = []
            self.triple_vectors = np.empty((0, 0), dtype=np.float32)
            return False

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.entity_ids = list(meta.get("entity_ids", []))
        self.triple_ids = list(meta.get("triple_ids", []))
        self.triple_endpoints = [
            (str(pair[0]), str(pair[1])) for pair in meta.get("triple_endpoints", [])
        ]
        self.triple_texts = list(meta.get("triple_texts", []))
        with np.load(vec_file) as data:
            self.entity_vectors = np.asarray(data["entity_vectors"], dtype=np.float32)
            self.triple_vectors = np.asarray(data["triple_vectors"], dtype=np.float32)
        return True

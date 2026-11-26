"""NetworkX-backed Knowledge Graph store with JSON persistence and k-hop traversal."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import networkx as nx

from llm_graphrag_kg.core.schema import Community, Entity, HyperEdge, Relation


def normalize_key(text: str) -> str:
    """Normalize an entity name or relation label for case/whitespace-insensitive matching."""
    return re.sub(r"\s+", " ", text.strip()).lower()


class NetworkXGraphStore:
    """Zero-infrastructure local KG store implementing `BaseGraphStore`.

    Uses a directed graph (`nx.DiGraph`) and persists cleanly to portable JSON files.
    Future migration to Neo4j/Kuzu only requires implementing the same `BaseGraphStore` protocol.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        self._communities: dict[int, Community] = {}
        self._hyperedges: dict[str, HyperEdge] = {}

    def upsert_entity(self, entity: Entity) -> Entity:
        existing = self._entities.get(entity.entity_id)
        if existing is None:
            self._entities[entity.entity_id] = entity.model_copy(deep=True)
            return self._entities[entity.entity_id]

        merged_aliases = list(dict.fromkeys([*existing.aliases, *entity.aliases]))
        if entity.name != existing.name and entity.name not in merged_aliases:
            merged_aliases.append(entity.name)

        merged_chunks = list(dict.fromkeys([*existing.source_chunk_ids, *entity.source_chunk_ids]))
        description = existing.description
        if entity.description and entity.description not in description:
            description = (
                f"{description} | {entity.description}" if description else entity.description
            )

        updated = existing.model_copy(
            update={
                "entity_type": existing.entity_type
                if existing.entity_type != "Concept"
                else entity.entity_type,
                "description": description[:600],
                "aliases": merged_aliases,
                "source_chunk_ids": merged_chunks,
                "community_id": entity.community_id
                if entity.community_id is not None
                else existing.community_id,
                "metadata": {**existing.metadata, **entity.metadata},
            }
        )
        self._entities[entity.entity_id] = updated
        return updated

    def upsert_relation(self, relation: Relation) -> Relation:
        existing = self._relations.get(relation.relation_id)
        if existing is None:
            self._relations[relation.relation_id] = relation.model_copy(deep=True)
            return self._relations[relation.relation_id]

        merged_chunks = list(
            dict.fromkeys([*existing.source_chunk_ids, *relation.source_chunk_ids])
        )
        description = existing.description
        if relation.description and relation.description not in description:
            description = (
                f"{description} | {relation.description}" if description else relation.description
            )

        updated = existing.model_copy(
            update={
                "weight": existing.weight + relation.weight,
                "description": description[:600],
                "source_chunk_ids": merged_chunks,
                "metadata": {**existing.metadata, **relation.metadata},
            }
        )
        self._relations[relation.relation_id] = updated
        return updated

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def find_entities_by_query(self, query: str, top_k: int = 5) -> list[Entity]:
        q_norm = normalize_key(query)
        if not q_norm:
            return []

        q_tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", q_norm))
        scored: list[tuple[float, Entity]] = []

        for entity in self._entities.values():
            candidates = [entity.name, *entity.aliases]
            score = 0.0
            for cand in candidates:
                c_norm = normalize_key(cand)
                if not c_norm:
                    continue
                if c_norm in q_norm:
                    score = max(score, 3.0 + min(len(c_norm), 12) * 0.1)
                elif q_norm in c_norm:
                    score = max(score, 2.5)
                else:
                    c_tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", c_norm))
                    overlap = q_tokens & c_tokens
                    if overlap:
                        score = max(score, 1.0 + len(overlap) * 0.5)

            # Secondary signal from entity description
            if score == 0.0 and entity.description:
                desc_norm = normalize_key(entity.description)
                desc_hits = sum(1 for tok in q_tokens if tok in desc_norm)
                if desc_hits > 0:
                    score = 0.4 * desc_hits

            if score > 0.0:
                scored.append((score, entity))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [entity for _, entity in scored[:top_k]]

    def get_all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def get_all_relations(self) -> list[Relation]:
        return list(self._relations.values())

    def upsert_hyperedge(self, hyperedge: HyperEdge) -> HyperEdge:
        """Insert or merge a multi-condition N-ary HyperEdge (Stage 3.4)."""
        existing = self._hyperedges.get(hyperedge.hyperedge_id)
        if existing is None:
            self._hyperedges[hyperedge.hyperedge_id] = hyperedge.model_copy(deep=True)
            return self._hyperedges[hyperedge.hyperedge_id]

        merged_conds = list(
            dict.fromkeys([*existing.condition_entity_ids, *hyperedge.condition_entity_ids])
        )
        merged_chunks = list(
            dict.fromkeys([*existing.source_chunk_ids, *hyperedge.source_chunk_ids])
        )
        updated = existing.model_copy(
            update={
                "condition_entity_ids": merged_conds,
                "weight": existing.weight + hyperedge.weight,
                "confidence": max(existing.confidence, hyperedge.confidence),
                "source_chunk_ids": merged_chunks,
                "description": hyperedge.description or existing.description,
                "metadata": {**existing.metadata, **hyperedge.metadata},
            }
        )
        self._hyperedges[hyperedge.hyperedge_id] = updated
        return updated

    def get_all_hyperedges(self) -> list[HyperEdge]:
        return list(self._hyperedges.values())

    def get_k_hop_subgraph(
        self, seed_entity_ids: list[str], k_hops: int = 1
    ) -> tuple[list[Entity], list[Relation]]:
        valid_seeds = [eid for eid in seed_entity_ids if eid in self._entities]
        if not valid_seeds:
            return [], []

        adjacency: dict[str, set[str]] = {eid: set() for eid in self._entities}
        for rel in self._relations.values():
            if rel.source_id in adjacency and rel.target_id in adjacency:
                adjacency[rel.source_id].add(rel.target_id)
                adjacency[rel.target_id].add(rel.source_id)

        visited: set[str] = set(valid_seeds)
        queue: deque[tuple[str, int]] = deque((eid, 0) for eid in valid_seeds)

        while queue:
            current, depth = queue.popleft()
            if depth >= k_hops:
                continue
            for neighbor in adjacency.get(current, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        sub_entities = [self._entities[eid] for eid in sorted(visited)]
        sub_relations = [
            rel
            for rel in self._relations.values()
            if rel.source_id in visited and rel.target_id in visited
        ]
        return sub_entities, sub_relations

    def set_communities(self, communities: list[Community]) -> None:
        self._communities = {c.community_id: c for c in communities}
        for community in communities:
            for eid in community.entity_ids:
                if eid in self._entities:
                    self._entities[eid] = self._entities[eid].model_copy(
                        update={"community_id": community.community_id}
                    )

    def get_all_communities(self) -> list[Community]:
        return sorted(self._communities.values(), key=lambda c: c.community_id)

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for entity in self._entities.values():
            graph.add_node(
                entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                aliases=entity.aliases,
                source_chunk_ids=entity.source_chunk_ids,
                community_id=entity.community_id if entity.community_id is not None else -1,
            )
        for rel in self._relations.values():
            if rel.source_id in self._entities and rel.target_id in self._entities:
                graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_id=rel.relation_id,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight,
                    source_chunk_ids=rel.source_chunk_ids,
                )
        return graph

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "entities": [e.model_dump() for e in self._entities.values()],
            "relations": [r.model_dump() for r in self._relations.values()],
            "communities": [c.model_dump() for c in self.get_all_communities()],
            "hyperedges": [h.model_dump() for h in self._hyperedges.values()],
        }
        (directory / "graph.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, directory: Path) -> None:
        graph_file = directory / "graph.json"
        if not graph_file.is_file():
            raise FileNotFoundError(
                f"Graph store not found at {graph_file}. Please run `build` first."
            )
        payload = json.loads(graph_file.read_text(encoding="utf-8"))
        self._entities = {
            item["entity_id"]: Entity(**item) for item in payload.get("entities", [])
        }
        self._relations = {
            item["relation_id"]: Relation(**item) for item in payload.get("relations", [])
        }
        self._communities = {
            item["community_id"]: Community(**item) for item in payload.get("communities", [])
        }
        self._hyperedges = {
            item["hyperedge_id"]: HyperEdge(**item) for item in payload.get("hyperedges", [])
        }

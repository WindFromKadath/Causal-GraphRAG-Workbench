"""GraphRAG retrieval engine supporting Local, Global, and Hybrid modes with provenance."""

from __future__ import annotations

import re

from llm_graphrag_kg.core.interfaces import (
    BaseEmbeddingClient,
    BaseGraphStore,
    BaseLLMClient,
    BaseVectorStore,
)
from llm_graphrag_kg.core.schema import (
    Community,
    Entity,
    EvidenceItem,
    QueryResult,
    Relation,
    SubgraphTrace,
)

QA_SYSTEM_PROMPT = (
    "You are a grounded GraphRAG assistant. Answer the user's question accurately using ONLY "
    "the provided knowledge graph facts, community summaries, and source text chunks. "
    "Cite chunk IDs (e.g., [doc_xxxx::chunk_0000]) whenever possible. "
    "If the context is insufficient, state clearly what is missing."
)


class GraphRAGRetriever:
    """Executes Local (k-hop KG), Global (Community), and Hybrid (KG + Vector) GraphRAG queries."""

    def __init__(
        self,
        *,
        llm: BaseLLMClient,
        embedder: BaseEmbeddingClient,
        graph_store: BaseGraphStore,
        vector_store: BaseVectorStore,
        top_k_chunks: int = 4,
        top_k_entities: int = 5,
        k_hops: int = 1,
        top_k_communities: int = 3,
        temperature: float = 0.2,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.top_k_chunks = top_k_chunks
        self.top_k_entities = top_k_entities
        self.k_hops = k_hops
        self.top_k_communities = top_k_communities
        self.temperature = temperature

    def query(self, question: str, *, mode: str = "hybrid") -> QueryResult:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"local", "global", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode!r}. Use local | global | hybrid.")

        seed_entities: list[Entity] = []
        expanded_entities: list[Entity] = []
        sub_relations: list[Relation] = []
        selected_communities: list[Community] = []
        evidence_by_chunk: dict[str, EvidenceItem] = {}

        # 1. Vector retrieval (used in hybrid, and as fallback/enrichment for local)
        vector_hits = []
        if normalized_mode in {"hybrid", "local"}:
            q_vec = self.embedder.embed([question])[0]
            vector_hits = self.vector_store.search(q_vec, top_k=self.top_k_chunks)
            if normalized_mode == "hybrid":
                for chunk, score in vector_hits:
                    evidence_by_chunk[chunk.chunk_id] = EvidenceItem(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        source=chunk.source,
                        text=chunk.text,
                        score=round(score, 4),
                        matched_via="vector",
                    )

        # 2. Local KG entity matching + k-hop subgraph expansion
        if normalized_mode in {"local", "hybrid"}:
            seed_entities = self.graph_store.find_entities_by_query(
                question, top_k=self.top_k_entities
            )
            # If lexical entity match missed, bootstrap seeds from top vector chunks
            if not seed_entities and vector_hits:
                top_chunk_ids = {chunk.chunk_id for chunk, _ in vector_hits[:2]}
                for ent in self.graph_store.get_all_entities():
                    if any(cid in top_chunk_ids for cid in ent.source_chunk_ids):
                        seed_entities.append(ent)
                        if len(seed_entities) >= self.top_k_entities:
                            break

            seed_ids = [e.entity_id for e in seed_entities]
            expanded_entities, sub_relations = self.graph_store.get_k_hop_subgraph(
                seed_ids, k_hops=self.k_hops
            )

            # Collect provenance chunks linked to the retrieved subgraph
            graph_chunk_ids: list[str] = []
            for rel in sub_relations:
                graph_chunk_ids.extend(rel.source_chunk_ids)
            for ent in expanded_entities:
                graph_chunk_ids.extend(ent.source_chunk_ids)

            for cid in dict.fromkeys(graph_chunk_ids):
                if cid in evidence_by_chunk:
                    item = evidence_by_chunk[cid]
                    evidence_by_chunk[cid] = item.model_copy(
                        update={"matched_via": f"{item.matched_via}+graph"}
                    )
                    continue
                chunk = self.vector_store.get_chunk(cid)
                if chunk is not None:
                    evidence_by_chunk[cid] = EvidenceItem(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        source=chunk.source,
                        text=chunk.text,
                        score=1.0,
                        matched_via="graph",
                    )
                if len(evidence_by_chunk) >= self.top_k_chunks + 3:
                    break

        # 3. Global community summary retrieval
        if normalized_mode in {"global", "hybrid"}:
            selected_communities = self._rank_communities(
                question,
                self.graph_store.get_all_communities(),
                expanded_entities,
            )[: self.top_k_communities]
            if normalized_mode == "global":
                # Surface community member entities in the trace so HTML export highlights them
                member_ids = [
                    eid for comm in selected_communities for eid in comm.entity_ids
                ]
                expanded_entities, sub_relations = self.graph_store.get_k_hop_subgraph(
                    member_ids, k_hops=0
                )

        evidences = list(evidence_by_chunk.values())
        prompt = self._compose_prompt(
            question=question,
            mode=normalized_mode,
            entities=expanded_entities,
            relations=sub_relations,
            communities=selected_communities,
            evidences=evidences,
        )
        answer = self.llm.complete(
            prompt,
            system_prompt=QA_SYSTEM_PROMPT,
            temperature=self.temperature,
            json_mode=False,
        )

        trace = SubgraphTrace(
            seed_entity_ids=[e.entity_id for e in seed_entities],
            expanded_entity_ids=[e.entity_id for e in expanded_entities],
            relation_ids=[r.relation_id for r in sub_relations],
            community_ids=[c.community_id for c in selected_communities],
        )
        return QueryResult(
            question=question,
            mode=normalized_mode,
            answer=answer,
            trace=trace,
            evidences=evidences,
            metadata={
                "seed_entity_names": [e.name for e in seed_entities],
                "expanded_entity_count": len(expanded_entities),
                "relation_count": len(sub_relations),
                "community_count": len(selected_communities),
            },
        )

    @staticmethod
    def _rank_communities(
        question: str,
        communities: list[Community],
        matched_entities: list[Entity],
    ) -> list[Community]:
        if not communities:
            return []
        q_tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", question.lower()))
        matched_comm_ids = {
            e.community_id for e in matched_entities if e.community_id is not None
        }

        scored: list[tuple[float, Community]] = []
        for comm in communities:
            haystack = f"{comm.title} {comm.summary}".lower()
            overlap = sum(1 for tok in q_tokens if tok in haystack)
            boost = 3.0 if comm.community_id in matched_comm_ids else 0.0
            scored.append((boost + overlap + len(comm.entity_ids) * 0.05, comm))
        scored.sort(key=lambda item: (-item[0], item[1].community_id))
        return [comm for _, comm in scored]

    @staticmethod
    def _compose_prompt(
        *,
        question: str,
        mode: str,
        entities: list[Entity],
        relations: list[Relation],
        communities: list[Community],
        evidences: list[EvidenceItem],
    ) -> str:
        entity_map = {e.entity_id: e for e in entities}
        sections: list[str] = [f"Retrieval Mode: {mode}", f"Question: {question}"]

        if entities:
            ent_lines = [
                f"- {e.name} ({e.entity_type}): {e.description}" for e in entities[:20]
            ]
            sections.append("Knowledge Graph Entities:\n" + "\n".join(ent_lines))

        if relations:
            rel_lines = []
            for r in relations[:25]:
                src = entity_map.get(r.source_id)
                tgt = entity_map.get(r.target_id)
                s_name = src.name if src else r.source_id
                t_name = tgt.name if tgt else r.target_id
                rel_lines.append(
                    f"- {s_name} --[{r.relation_type}]--> {t_name} (weight={r.weight:.1f}): "
                    f"{r.description}"
                )
            sections.append("Knowledge Graph Relations:\n" + "\n".join(rel_lines))

        if communities:
            comm_lines = [f"- [{c.title}] {c.summary}" for c in communities]
            sections.append("Community Summaries:\n" + "\n".join(comm_lines))

        if evidences:
            ev_lines = [
                f"- [{ev.chunk_id} | {ev.source} | via={ev.matched_via}]: {ev.text}"
                for ev in evidences[:8]
            ]
            sections.append("Source Text Evidences:\n" + "\n".join(ev_lines))

        sections.append(
            "Please provide a clear, structured Chinese answer citing relevant chunk IDs."
        )
        return "\n\n".join(sections)

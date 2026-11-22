"""GraphRAG retrieval engine supporting Local, Global, Hybrid, and PPR modes with provenance."""

from __future__ import annotations

import re

import networkx as nx

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
from llm_graphrag_kg.harness.context_compactor import ContextCompactor
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.reasoning.cot import GraphCoTEngine
from llm_graphrag_kg.reasoning.tot import GraphToTEngine

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
        enable_compact: bool = True,
        compactor: ContextCompactor | None = None,
        harness: LLMHarness | None = None,
        tot_beam_width: int = 3,
        tot_max_depth: int = 2,
        tot_max_llm_calls: int = 6,
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
        self.enable_compact = enable_compact
        self.compactor = compactor or ContextCompactor()
        self.harness = harness or LLMHarness(llm, compactor=self.compactor)
        self.tot_beam_width = tot_beam_width
        self.tot_max_depth = tot_max_depth
        self.tot_max_llm_calls = tot_max_llm_calls

    def query(
        self,
        question: str,
        *,
        mode: str = "hybrid",
        reasoning: str = "direct",
    ) -> QueryResult:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"local", "global", "hybrid", "ppr"}:
            raise ValueError(
                f"Unsupported retrieval mode: {mode!r}. Use local | global | hybrid | ppr."
            )
        normalized_reasoning = reasoning.strip().lower()
        if normalized_reasoning not in {"direct", "cot", "tot"}:
            raise ValueError(
                f"Unsupported reasoning paradigm: {reasoning!r}. Use direct | cot | tot."
            )

        seed_entities: list[Entity] = []
        expanded_entities: list[Entity] = []
        sub_relations: list[Relation] = []
        selected_communities: list[Community] = []
        evidence_by_chunk: dict[str, EvidenceItem] = {}
        ppr_top_scores: dict[str, float] = {}

        # 1. Vector retrieval (used in hybrid, ppr, and as fallback/enrichment for local)
        vector_hits = []
        if normalized_mode in {"hybrid", "local", "ppr"}:
            q_vec = self.embedder.embed([question])[0]
            vector_hits = self.vector_store.search(q_vec, top_k=self.top_k_chunks)
            if normalized_mode in {"hybrid", "ppr"}:
                for chunk, score in vector_hits:
                    evidence_by_chunk[chunk.chunk_id] = EvidenceItem(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        source=chunk.source,
                        text=chunk.text,
                        score=round(score, 4),
                        matched_via="vector",
                    )

        # 2. Local KG entity matching + k-hop expansion OR HippoRAG-style PPR diffusion
        if normalized_mode in {"local", "hybrid", "ppr"}:
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
            if normalized_mode == "ppr":
                (
                    expanded_entities,
                    sub_relations,
                    ppr_chunk_ids,
                    ppr_top_scores,
                ) = self._run_ppr_retrieval(
                    seed_ids=seed_ids,
                    vector_hits=vector_hits,
                    max_entities=max(self.compactor.max_entities, self.top_k_entities * 3),
                    max_chunks=self.top_k_chunks + 3,
                )
                graph_chunk_ids = list(ppr_chunk_ids)
            else:
                expanded_entities, sub_relations = self.graph_store.get_k_hop_subgraph(
                    seed_ids, k_hops=self.k_hops
                )
                graph_chunk_ids = []
                for rel in sub_relations:
                    graph_chunk_ids.extend(rel.source_chunk_ids)
                for ent in expanded_entities:
                    graph_chunk_ids.extend(ent.source_chunk_ids)

            for cid in dict.fromkeys(graph_chunk_ids):
                if cid in evidence_by_chunk:
                    item = evidence_by_chunk[cid]
                    tag = "ppr" if normalized_mode == "ppr" else "graph"
                    evidence_by_chunk[cid] = item.model_copy(
                        update={"matched_via": f"{item.matched_via}+{tag}"}
                    )
                    continue
                chunk = self.vector_store.get_chunk(cid)
                if chunk is not None:
                    evidence_by_chunk[cid] = EvidenceItem(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        source=chunk.source,
                        text=chunk.text,
                        score=round(ppr_top_scores.get(f"chunk::{cid}", 1.0), 4),
                        matched_via="ppr" if normalized_mode == "ppr" else "graph",
                    )
                if len(evidence_by_chunk) >= self.top_k_chunks + 3:
                    break

        # 3. Global community summary retrieval
        if normalized_mode in {"global", "hybrid", "ppr"}:
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
        compaction_meta: dict[str, object] | None = None
        reasoning_meta: dict[str, object] | None = None

        if normalized_reasoning == "cot":
            cot_engine = GraphCoTEngine(
                self.harness,
                compactor=self.compactor,
                enable_compact=self.enable_compact,
                temperature=self.temperature,
            )
            cot_result = cot_engine.run(
                question=question,
                seed_entity_ids=[e.entity_id for e in seed_entities],
                entities=expanded_entities,
                relations=sub_relations,
                communities=selected_communities,
                evidences=evidences,
            )
            answer = cot_result.final_answer
            compaction_meta = cot_result.compaction
            reasoning_meta = cot_result.model_dump()
        elif normalized_reasoning == "tot":
            tot_engine = GraphToTEngine(
                self.harness,
                graph_store=self.graph_store,
                vector_store=self.vector_store,
                compactor=self.compactor,
                enable_compact=self.enable_compact,
                beam_width=self.tot_beam_width,
                max_depth=self.tot_max_depth,
                max_llm_calls=self.tot_max_llm_calls,
                temperature=self.temperature,
            )
            tot_result, expanded_entities, sub_relations, evidences = tot_engine.run(
                question=question,
                seed_entity_ids=[e.entity_id for e in seed_entities],
                entities=expanded_entities,
                relations=sub_relations,
                communities=selected_communities,
                evidences=evidences,
            )
            answer = tot_result.final_answer
            compaction_meta = tot_result.compaction
            reasoning_meta = tot_result.model_dump()
        else:
            if self.enable_compact:
                compacted = self.compactor.compact_graphrag_context(
                    question=question,
                    seed_entity_ids=[e.entity_id for e in seed_entities],
                    entities=expanded_entities,
                    relations=sub_relations,
                    communities=selected_communities,
                    evidences=evidences,
                )
                compaction_meta = compacted.report.model_dump()
                prompt = LLMHarness.compile_prompt(
                    context_text=compacted.full_text,
                    question=question,
                    task_instruction=(
                        f"Retrieval Mode: {normalized_mode}\n"
                        "Please provide a clear, structured Chinese answer citing relevant chunk IDs."
                    ),
                )
            else:
                prompt = self._compose_prompt(
                    question=question,
                    mode=normalized_mode,
                    entities=expanded_entities,
                    relations=sub_relations,
                    communities=selected_communities,
                    evidences=evidences,
                )

            self.harness.reset_ledger()
            answer = self.harness.run_text(
                prompt,
                step_name=f"graphrag_query_{normalized_mode}",
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
        metadata: dict[str, object] = {
            "seed_entity_names": [e.name for e in seed_entities],
            "expanded_entity_count": len(expanded_entities),
            "relation_count": len(sub_relations),
            "community_count": len(selected_communities),
            "reasoning_paradigm": normalized_reasoning,
            "harness_ledger": self.harness.ledger.summary(),
        }
        if compaction_meta is not None:
            metadata["compaction"] = compaction_meta
        if reasoning_meta is not None:
            metadata["reasoning"] = reasoning_meta
        if ppr_top_scores:
            metadata["ppr_top_nodes"] = dict(
                sorted(ppr_top_scores.items(), key=lambda kv: -kv[1])[:8]
            )

        return QueryResult(
            question=question,
            mode=normalized_mode,
            answer=answer,
            trace=trace,
            evidences=evidences,
            metadata=metadata,
        )

    def _run_ppr_retrieval(
        self,
        *,
        seed_ids: list[str],
        vector_hits: list[tuple[object, float]],
        max_entities: int = 18,
        max_chunks: int = 6,
        alpha: float = 0.85,
    ) -> tuple[list[Entity], list[Relation], list[str], dict[str, float]]:
        """HippoRAG-inspired Personalized PageRank over heterogeneous entity-chunk graph."""
        all_entities = self.graph_store.get_all_entities()
        all_relations = self.graph_store.get_all_relations()
        if not all_entities:
            return [], [], [], {}

        ent_map = {e.entity_id: e for e in all_entities}
        graph = nx.Graph()

        for ent in all_entities:
            graph.add_node(ent.entity_id)
            for cid in ent.source_chunk_ids:
                cnode = f"chunk::{cid}"
                graph.add_edge(ent.entity_id, cnode, weight=0.5)

        for rel in all_relations:
            if rel.source_id in ent_map and rel.target_id in ent_map:
                w = max(0.1, float(rel.weight))
                if graph.has_edge(rel.source_id, rel.target_id):
                    graph[rel.source_id][rel.target_id]["weight"] += w
                else:
                    graph.add_edge(rel.source_id, rel.target_id, weight=w)

        personalization: dict[str, float] = {}
        for eid in seed_ids:
            if eid in graph:
                personalization[eid] = personalization.get(eid, 0.0) + 1.0
        for chunk, score in vector_hits:
            cid = getattr(chunk, "chunk_id", "")
            cnode = f"chunk::{cid}"
            if cnode in graph:
                personalization[cnode] = personalization.get(cnode, 0.0) + max(0.2, float(score))

        if not personalization:
            return [], [], [], {}

        total_p = sum(personalization.values())
        norm_pers = {k: v / total_p for k, v in personalization.items()}

        pr_scores = self._personalized_pagerank(graph, norm_pers, alpha=alpha)

        ranked_eids = [
            nid
            for nid, _ in sorted(
                ((k, v) for k, v in pr_scores.items() if k in ent_map),
                key=lambda kv: -kv[1],
            )[:max_entities]
        ]
        # Always retain direct seeds
        kept_eid_order = list(dict.fromkeys([*seed_ids, *ranked_eids]))[:max_entities]
        kept_eid_set = set(kept_eid_order)
        kept_entities = [ent_map[eid] for eid in kept_eid_order if eid in ent_map]

        kept_relations = [
            rel
            for rel in all_relations
            if rel.source_id in kept_eid_set and rel.target_id in kept_eid_set
        ]

        ranked_cids = [
            nid.removeprefix("chunk::")
            for nid, _ in sorted(
                ((k, v) for k, v in pr_scores.items() if k.startswith("chunk::")),
                key=lambda kv: -kv[1],
            )[:max_chunks]
        ]
        return kept_entities, kept_relations, ranked_cids, pr_scores

    @staticmethod
    def _personalized_pagerank(
        graph: nx.Graph,
        personalization: dict[str, float],
        *,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """Dependency-free weighted Personalized PageRank via power iteration (no scipy required)."""
        nodes = list(graph.nodes())
        if not nodes:
            return {}
        out_weight: dict[str, float] = {
            u: sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges(u, data=True))
            for u in nodes
        }
        scores: dict[str, float] = {u: personalization.get(u, 0.0) for u in nodes}
        dangling_nodes = [u for u, w in out_weight.items() if w <= 0.0]

        for _ in range(max_iter):
            prev = scores
            scores = {u: (1.0 - alpha) * personalization.get(u, 0.0) for u in nodes}
            dangling_sum = alpha * sum(prev[u] for u in dangling_nodes)
            if dangling_sum > 0.0:
                for u, p_val in personalization.items():
                    scores[u] = scores.get(u, 0.0) + dangling_sum * p_val

            for u in nodes:
                w_sum = out_weight[u]
                if w_sum <= 0.0 or prev[u] == 0.0:
                    continue
                share = alpha * prev[u] / w_sum
                for _, v, data in graph.edges(u, data=True):
                    scores[v] += share * float(data.get("weight", 1.0))

            diff = sum(abs(scores[u] - prev[u]) for u in nodes)
            if diff < tol:
                break
        return scores

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

"""GraphRAG retrieval engine supporting Local, Global, Hybrid, and PPR modes with provenance."""

from __future__ import annotations

import math
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
from llm_graphrag_kg.pipelines.hipporag import SemanticGraphIndex
from llm_graphrag_kg.pipelines.hypergraph import (
    format_hyperedge_context,
    match_hyperedges,
)
from llm_graphrag_kg.pipelines.memory_store import (
    VerifiedMemoryStore,
    format_recalled_memories_context,
)
from llm_graphrag_kg.reasoning.cot import GraphCoTEngine
from llm_graphrag_kg.reasoning.cot_iterative import IterativeGraphCoTEngine
from llm_graphrag_kg.reasoning.router import route_query
from llm_graphrag_kg.reasoning.tot import GraphToTEngine
from llm_graphrag_kg.reasoning.tot_deep import DeepGraphToTEngine

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
        planner_harness: LLMHarness | None = None,
        tot_beam_width: int = 3,
        tot_max_depth: int = 2,
        tot_max_llm_calls: int = 6,
        enable_early_stop: bool = True,
        early_stop_coverage_threshold: float = 0.80,
        multi_branch_min_score: float = 0.25,
        semantic_index: SemanticGraphIndex | None = None,
        enable_hybrid_bm25: bool = True,
        hybrid_bm25_weight: float = 0.35,
        rrf_k: int = 60,
        enable_query_to_triple: bool = True,
        top_k_triples: int = 8,
        enable_hypergraph: bool = True,
        hyperedge_match_threshold: float = 0.60,
        memory_store: VerifiedMemoryStore | None = None,
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
        self.planner_harness = planner_harness or self.harness
        self.tot_beam_width = tot_beam_width
        self.tot_max_depth = tot_max_depth
        self.tot_max_llm_calls = tot_max_llm_calls
        self.enable_early_stop = enable_early_stop
        self.early_stop_coverage_threshold = early_stop_coverage_threshold
        self.multi_branch_min_score = multi_branch_min_score
        self.semantic_index = semantic_index
        self.enable_hybrid_bm25 = enable_hybrid_bm25
        self.hybrid_bm25_weight = hybrid_bm25_weight
        self.rrf_k = rrf_k
        self.enable_query_to_triple = enable_query_to_triple
        self.top_k_triples = top_k_triples
        self.enable_hypergraph = enable_hypergraph
        self.hyperedge_match_threshold = hyperedge_match_threshold
        self.memory_store = memory_store

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
        valid_paradigms = {
            "direct",
            "cot",
            "cot_iter",
            "tot",
            "tot_bfs",
            "tot_dfs",
            "multi_branch",
            "auto",
        }
        if normalized_reasoning not in valid_paradigms:
            raise ValueError(
                f"Unsupported reasoning paradigm: {reasoning!r}. "
                "Use auto | direct | cot | cot_iter | tot | tot_bfs | tot_dfs | multi_branch."
            )

        seed_entities: list[Entity] = []
        expanded_entities: list[Entity] = []
        sub_relations: list[Relation] = []
        selected_communities: list[Community] = []
        evidence_by_chunk: dict[str, EvidenceItem] = {}
        ppr_top_scores: dict[str, float] = {}

        # 1. Chunk retrieval: Stage 3.1 Dense + BM25 RRF hybrid (falls back to dense cosine)
        q_vec: list[float] | None = None
        vector_hits = []
        ppr_passage_pool = []
        used_hybrid_rrf = False
        if normalized_mode in {"hybrid", "local", "ppr"}:
            q_vec = self.embedder.embed([question])[0]
            pool_k = max(self.top_k_chunks * 3, 15) if normalized_mode == "ppr" else self.top_k_chunks
            if self.enable_hybrid_bm25 and hasattr(self.vector_store, "search_hybrid"):
                ppr_passage_pool = self.vector_store.search_hybrid(
                    q_vec,
                    question,
                    top_k=pool_k,
                    bm25_weight=self.hybrid_bm25_weight,
                    rrf_k=self.rrf_k,
                )
                vector_hits = ppr_passage_pool[: self.top_k_chunks]
                used_hybrid_rrf = True
            else:
                ppr_passage_pool = self.vector_store.search(q_vec, top_k=pool_k)
                vector_hits = ppr_passage_pool[: self.top_k_chunks]

            if normalized_mode in {"hybrid", "ppr"}:
                vec_tag = "hybrid_rrf" if used_hybrid_rrf else "vector"
                for chunk, score in vector_hits:
                    evidence_by_chunk[chunk.chunk_id] = EvidenceItem(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        source=chunk.source,
                        text=chunk.text,
                        score=round(score, 4),
                        matched_via=vec_tag,
                    )

        # 2. Local KG entity matching + Stage 3.1 HippoRAG 2 (Dense Entity + Query-to-Triple + PPR)
        dense_entity_weights: dict[str, float] = {}
        triple_seed_weights: dict[str, float] = {}
        triple_hits: list[dict[str, object]] = []
        if normalized_mode in {"local", "hybrid", "ppr"}:
            seed_entities = self.graph_store.find_entities_by_query(
                question, top_k=self.top_k_entities
            )
            seed_id_seen = {e.entity_id for e in seed_entities}
            max_seed_cap = max(self.top_k_entities * 2, self.top_k_entities + 4)

            if q_vec is not None and self.semantic_index is not None and self.semantic_index.is_ready:
                # 2a. Dense entity linking
                for eid, sc in self.semantic_index.search_entities(
                    q_vec, top_k=self.top_k_entities
                ):
                    dense_entity_weights[eid] = max(dense_entity_weights.get(eid, 0.0), sc)
                    if eid not in seed_id_seen and len(seed_entities) < max_seed_cap:
                        ent_obj = self.graph_store.get_entity(eid)
                        if ent_obj is not None:
                            seed_entities.append(ent_obj)
                            seed_id_seen.add(eid)

                # 2b. HippoRAG 2 Query-to-Triple seed anchoring
                if self.enable_query_to_triple and self.semantic_index.has_triples:
                    triple_hits = self.semantic_index.search_triples(
                        q_vec, top_k=self.top_k_triples
                    )
                    for hit in triple_hits:
                        sc = float(hit["score"])
                        for endpoint_key in ("source_id", "target_id"):
                            eid = str(hit[endpoint_key])
                            triple_seed_weights[eid] = max(
                                triple_seed_weights.get(eid, 0.0), sc
                            )
                            if eid not in seed_id_seen and len(seed_entities) < max_seed_cap:
                                ent_obj = self.graph_store.get_entity(eid)
                                if ent_obj is not None:
                                    seed_entities.append(ent_obj)
                                    seed_id_seen.add(eid)

            # If lexical + semantic entity match still missed, bootstrap seeds from top vector chunks
            if not seed_entities and vector_hits:
                top_chunk_ids = {chunk.chunk_id for chunk, _ in vector_hits[:2]}
                for ent in self.graph_store.get_all_entities():
                    if any(cid in top_chunk_ids for cid in ent.source_chunk_ids):
                        seed_entities.append(ent)
                        if len(seed_entities) >= self.top_k_entities:
                            break

            seed_ids = [e.entity_id for e in seed_entities]
            max_evidence_pool = self.top_k_chunks + 5 if normalized_mode == "ppr" else self.top_k_chunks + 3
            if normalized_mode == "ppr":
                (
                    expanded_entities,
                    sub_relations,
                    ppr_chunk_ids,
                    ppr_top_scores,
                ) = self._run_ppr_retrieval(
                    seed_ids=seed_ids,
                    vector_hits=ppr_passage_pool or vector_hits,
                    dense_entity_weights=dense_entity_weights,
                    triple_seed_weights=triple_seed_weights,
                    max_entities=max(self.compactor.max_entities, self.top_k_entities * 3),
                    max_chunks=max_evidence_pool,
                )
                graph_chunk_ids = list(ppr_chunk_ids)
                # Surface provenance chunks of the highest-scoring Query-to-Triple matches
                if triple_hits:
                    rel_map = {r.relation_id: r for r in self.graph_store.get_all_relations()}
                    for hit in triple_hits[:4]:
                        rel_obj = rel_map.get(str(hit["relation_id"]))
                        if rel_obj is not None:
                            graph_chunk_ids.extend(rel_obj.source_chunk_ids[:2])
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
                if len(evidence_by_chunk) >= max_evidence_pool:
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

        # Stage 3.4: Multi-condition HyperGraph matching & Verified Episodic Memory recall
        matched_hyperedges = []
        if self.enable_hypergraph and hasattr(self.graph_store, "get_all_hyperedges"):
            all_he = self.graph_store.get_all_hyperedges()
            if all_he:
                all_ent_map = {e.entity_id: e for e in self.graph_store.get_all_entities()}
                active_ids = list(
                    dict.fromkeys(
                        [e.entity_id for e in seed_entities]
                        + [e.entity_id for e in expanded_entities[:12]]
                    )
                )
                matched_hyperedges = match_hyperedges(
                    active_ids,
                    all_he,
                    all_ent_map,
                    min_match_ratio=self.hyperedge_match_threshold,
                )
                # Ensure matched hyperedge targets & their provenance chunks enter the reasoning pool
                exp_id_set = {e.entity_id for e in expanded_entities}
                for m in matched_hyperedges:
                    tgt_ent = all_ent_map.get(m.target_entity_id)
                    if tgt_ent is not None and tgt_ent.entity_id not in exp_id_set:
                        expanded_entities.append(tgt_ent)
                        exp_id_set.add(tgt_ent.entity_id)
                    for cid in m.source_chunk_ids:
                        if cid not in evidence_by_chunk:
                            ch = self.vector_store.get_chunk(cid)
                            if ch is not None:
                                evidence_by_chunk[cid] = EvidenceItem(
                                    chunk_id=ch.chunk_id,
                                    doc_id=ch.doc_id,
                                    source=ch.source,
                                    text=ch.text,
                                    score=round(m.joint_score, 4),
                                    matched_via="hypergraph",
                                )
                evidences = list(evidence_by_chunk.values())

        recalled_memories = (
            self.memory_store.retrieve_relevant(
                question, [e.entity_id for e in seed_entities], top_k=3
            )
            if self.memory_store is not None
            else []
        )
        hypergraph_note = format_hyperedge_context(matched_hyperedges)
        memory_note = format_recalled_memories_context(recalled_memories)
        if hypergraph_note or memory_note:
            supplemental_blocks = [b for b in (hypergraph_note, memory_note) if b]
            # Attach supplemental joint rules/memories onto community summary slot so all reasoning
            # engines (direct/cot/tot/multi_branch/cot_iter/tot_bfs/tot_dfs) seamlessly observe them
            selected_communities = [
                *selected_communities,
                Community(
                    community_id=-34,
                    title="Stage 3.4 HyperGraph & Verified Memory Context",
                    summary="\n\n".join(supplemental_blocks),
                    entity_ids=[m.target_entity_id for m in matched_hyperedges],
                    relation_count=len(matched_hyperedges),
                ),
            ]

        compaction_meta: dict[str, object] | None = None
        reasoning_meta: dict[str, object] | None = None
        route_decision_meta: dict[str, object] | None = None

        effective_reasoning = normalized_reasoning
        if normalized_reasoning == "auto":
            decision = route_query(
                question,
                seed_entities=seed_entities,
                relations=sub_relations,
                evidences=evidences,
                enable_early_stop=self.enable_early_stop,
                early_stop_coverage_threshold=self.early_stop_coverage_threshold,
            )
            effective_reasoning = decision.target_paradigm
            route_decision_meta = decision.model_dump()

        if effective_reasoning == "cot":
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
            if route_decision_meta is not None:
                cot_result = cot_result.model_copy(update={"route_decision": route_decision_meta})
            answer = cot_result.final_answer
            compaction_meta = cot_result.compaction
            reasoning_meta = cot_result.model_dump()
        elif effective_reasoning == "cot_iter":
            iter_engine = IterativeGraphCoTEngine(
                self.harness,
                planner_harness=self.planner_harness,
                graph_store=self.graph_store,
                vector_store=self.vector_store,
                embedder=self.embedder,
                semantic_index=self.semantic_index,
                compactor=self.compactor,
                max_iters=3,
                temperature=self.temperature,
            )
            iter_result, expanded_entities, sub_relations, evidences = iter_engine.run(
                question=question,
                seed_entity_ids=[e.entity_id for e in seed_entities],
                entities=expanded_entities,
                relations=sub_relations,
                communities=selected_communities,
                evidences=evidences,
            )
            answer = iter_result.final_answer
            compaction_meta = iter_result.compaction
            reasoning_meta = iter_result.model_dump()
        elif effective_reasoning in {"tot_bfs", "tot_dfs"}:
            deep_engine = DeepGraphToTEngine(
                self.harness,
                planner_harness=self.planner_harness,
                graph_store=self.graph_store,
                vector_store=self.vector_store,
                compactor=self.compactor,
                search_strategy="dfs" if effective_reasoning == "tot_dfs" else "bfs",
                beam_width=min(self.tot_beam_width, 3),
                max_depth=self.tot_max_depth,
                temperature=self.temperature,
            )
            deep_result, expanded_entities, sub_relations, evidences = deep_engine.run(
                question=question,
                seed_entity_ids=[e.entity_id for e in seed_entities],
                entities=expanded_entities,
                relations=sub_relations,
                communities=selected_communities,
                evidences=evidences,
            )
            answer = deep_result.final_answer
            compaction_meta = deep_result.compaction
            reasoning_meta = deep_result.model_dump()
        elif effective_reasoning in {"tot", "multi_branch"}:
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
                synthesis_mode="additive" if effective_reasoning == "multi_branch" else "competitive",
                min_branch_score=self.multi_branch_min_score,
            )
            tot_result, expanded_entities, sub_relations, evidences = tot_engine.run(
                question=question,
                seed_entity_ids=[e.entity_id for e in seed_entities],
                entities=expanded_entities,
                relations=sub_relations,
                communities=selected_communities,
                evidences=evidences,
            )
            if route_decision_meta is not None:
                tot_result = tot_result.model_copy(update={"route_decision": route_decision_meta})
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
                        "Please provide a clear, concise answer in the same language as the "
                        "question, citing relevant chunk IDs."
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
        merged_ledger = self.harness.ledger.summary()
        if isinstance(reasoning_meta, dict) and isinstance(reasoning_meta.get("trace_ledger"), dict):
            merged_ledger = dict(reasoning_meta["trace_ledger"])
        metadata: dict[str, object] = {
            "seed_entity_names": [e.name for e in seed_entities],
            "expanded_entity_count": len(expanded_entities),
            "relation_count": len(sub_relations),
            "community_count": len(selected_communities),
            "reasoning_paradigm": effective_reasoning,
            "requested_reasoning": normalized_reasoning,
            "harness_ledger": merged_ledger,
            "retrieval_stage3_1": {
                "hybrid_bm25_rrf": used_hybrid_rrf,
                "dense_entity_hits": len(dense_entity_weights),
                "query_to_triple_hits": [
                    {"text": str(h["text"]), "score": float(h["score"])} for h in triple_hits[:5]
                ],
            },
        }
        if compaction_meta is not None:
            metadata["compaction"] = compaction_meta
        if reasoning_meta is not None:
            metadata["reasoning"] = reasoning_meta
        if route_decision_meta is not None:
            metadata["route_decision"] = route_decision_meta
        if matched_hyperedges:
            metadata["matched_hyperedges"] = [m.model_dump() for m in matched_hyperedges]
        if recalled_memories:
            metadata["recalled_memories"] = [r.model_dump() for r in recalled_memories]
        if ppr_top_scores:
            metadata["ppr_top_nodes"] = dict(
                sorted(ppr_top_scores.items(), key=lambda kv: -kv[1])[:8]
            )
        expanded_eids_set = {e.entity_id for e in expanded_entities}
        metadata["traversal_trace"] = {
            "seed_entity_ids": [e.entity_id for e in seed_entities],
            "dense_entity_scores": {
                eid: round(float(sc), 4) for eid, sc in dense_entity_weights.items()
            },
            "triple_seed_scores": {
                eid: round(float(sc), 4) for eid, sc in triple_seed_weights.items()
            },
            "ppr_entity_scores": {
                eid: round(float(ppr_top_scores[eid]), 4)
                for eid in expanded_eids_set
                if eid in ppr_top_scores
            },
        }

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
        dense_entity_weights: dict[str, float] | None = None,
        triple_seed_weights: dict[str, float] | None = None,
        max_entities: int = 18,
        max_chunks: int = 6,
        alpha: float = 0.85,
    ) -> tuple[list[Entity], list[Relation], list[str], dict[str, float]]:
        """HippoRAG 2 Personalized PageRank over entity-synonym-chunk heterogeneous graph."""
        all_entities = self.graph_store.get_all_entities()
        all_relations = self.graph_store.get_all_relations()
        if not all_entities:
            return [], [], [], {}

        ent_map = {e.entity_id: e for e in all_entities}
        graph = nx.Graph()

        for ent in all_entities:
            graph.add_node(ent.entity_id)
            # Specificity-weighted entity-chunk link (dampens generic high-frequency hubs)
            deg_scale = math.sqrt(max(1, len(ent.source_chunk_ids)))
            ec_weight = 0.5 / deg_scale
            for cid in ent.source_chunk_ids:
                cnode = f"chunk::{cid}"
                graph.add_edge(ent.entity_id, cnode, weight=ec_weight)

        for rel in all_relations:
            if rel.source_id in ent_map and rel.target_id in ent_map:
                if rel.relation_type == "SYNONYM_OF":
                    w = max(0.3, float(rel.weight))
                else:
                    w = max(0.1, float(rel.weight))
                if graph.has_edge(rel.source_id, rel.target_id):
                    graph[rel.source_id][rel.target_id]["weight"] += w
                else:
                    graph.add_edge(rel.source_id, rel.target_id, weight=w)

        personalization: dict[str, float] = {}
        for eid in seed_ids:
            if eid in graph:
                personalization[eid] = personalization.get(eid, 0.0) + 1.0
        for eid, sc in (dense_entity_weights or {}).items():
            if eid in graph:
                personalization[eid] = personalization.get(eid, 0.0) + 0.8 * max(0.1, float(sc))
        for eid, sc in (triple_seed_weights or {}).items():
            if eid in graph:
                personalization[eid] = personalization.get(eid, 0.0) + 1.2 * max(0.1, float(sc))
        for chunk, score in vector_hits:
            cid = getattr(chunk, "chunk_id", "")
            cnode = f"chunk::{cid}"
            if cnode in graph:
                personalization[cnode] = personalization.get(cnode, 0.0) + max(
                    0.25, float(score) * 1.5
                )

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
            "Please provide a clear, concise answer in the same language as the question, "
            "citing relevant chunk IDs."
        )
        return "\n\n".join(sections)

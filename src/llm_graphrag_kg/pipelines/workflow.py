"""High-level GraphRAGPipeline orchestrating indexing, persistence, and querying."""

from __future__ import annotations

import json
from pathlib import Path

from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import (
    BaseEmbeddingClient,
    BaseGraphStore,
    BaseLLMClient,
    BaseVectorStore,
)
from llm_graphrag_kg.core.schema import GraphAuditStats, QueryResult
from llm_graphrag_kg.harness.context_compactor import ContextCompactor
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.pipelines.community import CommunityDetector
from llm_graphrag_kg.pipelines.extraction import KnowledgeExtractor
from llm_graphrag_kg.pipelines.hipporag import SemanticGraphIndex, build_synonym_edges
from llm_graphrag_kg.pipelines.hypergraph import mine_joint_hyperedges
from llm_graphrag_kg.pipelines.ingestion import DocumentIngestor
from llm_graphrag_kg.pipelines.retriever import GraphRAGRetriever
from llm_graphrag_kg.providers.factory import (
    create_embedding_client,
    create_fast_llm_client,
    create_graph_store,
    create_llm_client,
    create_memory_store,
    create_vector_store,
)


class GraphRAGPipeline:
    """Facade combining ingestion, KG extraction, community detection, storage, and retrieval."""

    def __init__(
        self,
        config: AppConfig,
        *,
        llm: BaseLLMClient | None = None,
        planner_llm: BaseLLMClient | None = None,
        embedder: BaseEmbeddingClient | None = None,
        graph_store: BaseGraphStore | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.config = config
        self.llm = llm or create_llm_client(config)
        self.planner_llm = planner_llm if planner_llm is not None else create_fast_llm_client(config)
        self.embedder = embedder or create_embedding_client(config)
        self.graph_store = graph_store or create_graph_store(config)
        self.vector_store = vector_store or create_vector_store(config)
        self.semantic_index = SemanticGraphIndex()
        self.memory_store = create_memory_store(config)

    def build(
        self,
        input_dir: Path | None = None,
        *,
        storage_dir: Path | None = None,
        summarize_communities: bool | None = None,
    ) -> GraphAuditStats:
        """Run full offline indexing pipeline and persist artifacts to `storage_dir`."""
        src_dir = input_dir or self.config.input_path
        dst_dir = storage_dir or self.config.storage_path

        ingestor = DocumentIngestor(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )
        documents = ingestor.load_directory(src_dir, base_dir=self.config.project_root)
        chunks = ingestor.chunk_documents(documents)

        # Reset fresh stores for deterministic rebuild
        self.graph_store = create_graph_store(self.config)
        self.vector_store = create_vector_store(self.config)
        self.semantic_index = SemanticGraphIndex()

        # 1. Embed and store chunks
        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            self.vector_store.add_chunks(chunks, vectors)

        # 2. Extract entities and relations per chunk
        extractor = KnowledgeExtractor(
            self.llm,
            entity_types=self.config.extraction.entity_types,
            temperature=self.config.extraction.temperature,
            max_entities_per_chunk=self.config.extraction.max_entities_per_chunk,
        )
        for chunk in chunks:
            entities, relations = extractor.extract_from_chunk(chunk)
            for entity in entities:
                self.graph_store.upsert_entity(entity)
            for relation in relations:
                self.graph_store.upsert_relation(relation)

        # 2.5 Stage 3.1 HippoRAG 2: SYNONYM_OF auto-linking + Semantic Entity/Triple Index
        synonym_edges_added = 0
        precomputed_entities = None
        if self.config.retrieval.enable_synonym_edges:
            synonym_edges_added, ent_ids, ent_vecs = build_synonym_edges(
                self.graph_store,
                self.embedder,
                cosine_threshold=self.config.retrieval.synonym_cosine_threshold,
                max_synonyms_per_entity=self.config.retrieval.max_synonyms_per_entity,
            )
            precomputed_entities = (ent_ids, ent_vecs)
        sem_stats = self.semantic_index.build(
            self.graph_store,
            self.embedder,
            precomputed_entities=precomputed_entities,
            max_indexed_triples=self.config.retrieval.max_indexed_triples,
        )

        # 3. Detect communities
        use_llm_summary = (
            self.config.community.summarize_with_llm
            if summarize_communities is None
            else summarize_communities
        )
        detector = CommunityDetector(
            resolution=self.config.community.resolution,
            seed=self.config.community.seed,
            summarize_with_llm=use_llm_summary,
        )
        detector.detect_and_summarize(self.graph_store, llm=self.llm)

        # Stage 3.4: Auto-mine multi-condition N-ary HyperEdges from grounded KG co-occurrence
        hyperedges_mined = 0
        if self.config.memory.enable_hypergraph:
            mined_he = mine_joint_hyperedges(
                self.graph_store,
                min_conditions=self.config.memory.hyperedge_min_conditions,
            )
            hyperedges_mined = len(mined_he)

        # 4. Persist stores and manifest
        dst_dir.mkdir(parents=True, exist_ok=True)
        self.graph_store.save(dst_dir)
        self.vector_store.save(dst_dir)
        self.semantic_index.save(dst_dir)
        self.memory_store.storage_dir = dst_dir
        manifest = {
            "doc_count": len(documents),
            "chunk_count": len(chunks),
            "embedding_provider": self.config.env.embedding_provider,
            "embedding_model": self.config.env.embedding_model,
            "synonym_edges_added": synonym_edges_added,
            "hyperedges_mined": hyperedges_mined,
            "semantic_index": sem_stats,
            "documents": [
                {"doc_id": d.doc_id, "source": d.source, "title": d.title} for d in documents
            ],
        }
        (dst_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        from llm_graphrag_kg.viz.audit_report import compute_audit_stats

        return compute_audit_stats(self.graph_store, self.vector_store, doc_count=len(documents))

    def _uses_enterprise_storage(self) -> bool:
        return (
            self.config.env.vector_backend.strip().lower() == "pgvector"
            or self.config.env.graph_backend.strip().lower() == "neo4j"
        )

    def load(self, storage_dir: Path | None = None) -> int:
        """Load persisted graph, vector, semantic, and verified memory stores."""
        src_dir = storage_dir or self.config.storage_path
        self.graph_store.load(src_dir)
        self.vector_store.load(src_dir)
        if not self._uses_enterprise_storage():
            self.semantic_index.load(src_dir)
        self.memory_store.load(src_dir)
        manifest_path = src_dir / "manifest.json"
        if not self._uses_enterprise_storage() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return int(manifest.get("doc_count", 0))
        return len({c.doc_id for c in self.vector_store.get_all_chunks()})

    def query(
        self,
        question: str,
        *,
        mode: str | None = None,
        storage_dir: Path | None = None,
        auto_load: bool = True,
        enable_compact: bool | None = None,
        reasoning: str | None = None,
        enable_writeback: bool | None = None,
    ) -> QueryResult:
        """Run an online GraphRAG query with full subgraph, compaction, reasoning, and optional A-MEM write-back."""
        if auto_load and not self.graph_store.get_all_entities():
            self.load(storage_dir)
        if storage_dir is not None and self.memory_store.storage_dir is None:
            self.memory_store.load(storage_dir)
        elif self.memory_store.storage_dir is None:
            self.memory_store.storage_dir = self.config.storage_path

        use_compact = (
            self.config.harness.enable_compact if enable_compact is None else enable_compact
        )
        compactor = ContextCompactor(
            max_context_tokens=self.config.harness.max_context_tokens,
            max_entities=self.config.harness.max_entities,
            max_relations=self.config.harness.max_subgraph_relations,
            max_evidence_chunks=self.config.harness.max_evidence_chunks,
        )
        harness = LLMHarness(
            self.llm,
            compactor=compactor,
            json_repair_retries=self.config.harness.json_repair_retries,
            max_calls=self.config.harness.max_calls,
            max_wall_seconds=self.config.harness.max_wall_seconds,
        )
        planner_harness = (
            LLMHarness(
                self.planner_llm,
                compactor=compactor,
                json_repair_retries=self.config.harness.json_repair_retries,
                max_calls=self.config.harness.max_calls,
                max_wall_seconds=self.config.harness.max_wall_seconds,
            )
            if self.planner_llm is not None
            else None
        )
        retriever = GraphRAGRetriever(
            llm=self.llm,
            embedder=self.embedder,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            top_k_chunks=self.config.retrieval.top_k_chunks,
            top_k_entities=self.config.retrieval.top_k_entities,
            k_hops=self.config.retrieval.k_hops,
            top_k_communities=self.config.retrieval.top_k_communities,
            temperature=self.config.retrieval.temperature,
            enable_compact=use_compact,
            compactor=compactor,
            harness=harness,
            planner_harness=planner_harness,
            tot_beam_width=self.config.reasoning.tot_beam_width,
            tot_max_depth=self.config.reasoning.tot_max_depth,
            tot_max_llm_calls=self.config.reasoning.tot_max_llm_calls,
            enable_early_stop=self.config.reasoning.enable_early_stop,
            early_stop_coverage_threshold=self.config.reasoning.early_stop_coverage_threshold,
            multi_branch_min_score=self.config.reasoning.multi_branch_min_score,
            semantic_index=self.semantic_index,
            enable_hybrid_bm25=self.config.retrieval.enable_hybrid_bm25,
            hybrid_bm25_weight=self.config.retrieval.hybrid_bm25_weight,
            rrf_k=self.config.retrieval.rrf_k,
            enable_query_to_triple=self.config.retrieval.enable_query_to_triple,
            top_k_triples=self.config.retrieval.top_k_triples,
            enable_hypergraph=self.config.memory.enable_hypergraph,
            hyperedge_match_threshold=self.config.memory.hyperedge_match_threshold,
            memory_store=self.memory_store,
        )
        result = retriever.query(
            question,
            mode=mode or self.config.retrieval.default_mode,
            reasoning=reasoning or self.config.reasoning.default_paradigm,
        )

        # Stage 3.4: Optional gated verified self-evolving memory write-back (A-MEM)
        use_wb = (
            self.config.memory.enable_verified_writeback
            if enable_writeback is None
            else enable_writeback
        )
        if use_wb:
            rm = result.metadata.get("reasoning")
            g_score: float | None = None
            ungrounded: list[str] = []
            conf = 0.0
            target_eids: list[str] = []
            cited_chunks: list[str] = []
            if isinstance(rm, dict):
                ga = rm.get("grounding_audit")
                if isinstance(ga, dict):
                    g_score = float(ga.get("grounding_score", 0.0))
                    ungrounded = list(ga.get("ungrounded_citations", []))
                conf = float(rm.get("confidence", 0.0))
                for br in rm.get("branches", []):
                    if isinstance(br, dict) and br.get("status") == "selected":
                        if br.get("target_entity_id"):
                            target_eids.append(str(br["target_entity_id"]))
                for st in rm.get("steps", []):
                    if isinstance(st, dict):
                        cited_chunks.extend(str(c) for c in st.get("cited_chunk_ids", []))
            if not cited_chunks:
                cited_chunks = [ev.chunk_id for ev in result.evidences[:4]]
            if not target_eids:
                for hm in result.metadata.get("matched_hyperedges", []):
                    if isinstance(hm, dict) and hm.get("target_entity_id"):
                        target_eids.append(str(hm["target_entity_id"]))
            decision = self.memory_store.evaluate_and_write_back(
                question=question,
                answer=result.answer,
                grounding_score=g_score,
                ungrounded_citations=ungrounded,
                confidence=conf,
                condition_entity_ids=result.trace.seed_entity_ids[:5],
                target_entity_ids=target_eids[:3],
                source_chunk_ids=cited_chunks,
                reasoning_paradigm=str(result.metadata.get("reasoning_paradigm", "auto")),
                min_grounding=self.config.memory.min_writeback_grounding,
                min_confidence=self.config.memory.min_writeback_confidence,
                enable_conflict_check=self.config.memory.enable_conflict_check,
                promote_to_hyperedge=self.config.memory.promote_memory_to_hyperedge,
                graph_store=self.graph_store,
            )
            result.metadata["writeback_decision"] = decision.model_dump()

        return result

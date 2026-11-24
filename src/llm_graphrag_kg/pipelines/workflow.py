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
from llm_graphrag_kg.pipelines.ingestion import DocumentIngestor
from llm_graphrag_kg.pipelines.retriever import GraphRAGRetriever
from llm_graphrag_kg.providers.factory import (
    create_embedding_client,
    create_graph_store,
    create_llm_client,
    create_vector_store,
)


class GraphRAGPipeline:
    """Facade combining ingestion, KG extraction, community detection, storage, and retrieval."""

    def __init__(
        self,
        config: AppConfig,
        *,
        llm: BaseLLMClient | None = None,
        embedder: BaseEmbeddingClient | None = None,
        graph_store: BaseGraphStore | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.config = config
        self.llm = llm or create_llm_client(config)
        self.embedder = embedder or create_embedding_client(config)
        self.graph_store = graph_store or create_graph_store()
        self.vector_store = vector_store or create_vector_store()
        self.semantic_index = SemanticGraphIndex()

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
        self.graph_store = create_graph_store()
        self.vector_store = create_vector_store()
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

        # 4. Persist stores and manifest
        dst_dir.mkdir(parents=True, exist_ok=True)
        self.graph_store.save(dst_dir)
        self.vector_store.save(dst_dir)
        self.semantic_index.save(dst_dir)
        manifest = {
            "doc_count": len(documents),
            "chunk_count": len(chunks),
            "embedding_provider": self.config.env.embedding_provider,
            "embedding_model": self.config.env.embedding_model,
            "synonym_edges_added": synonym_edges_added,
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

    def load(self, storage_dir: Path | None = None) -> int:
        """Load persisted graph, vector, and semantic stores from disk; returns document count."""
        src_dir = storage_dir or self.config.storage_path
        self.graph_store.load(src_dir)
        self.vector_store.load(src_dir)
        self.semantic_index.load(src_dir)
        manifest_path = src_dir / "manifest.json"
        if manifest_path.is_file():
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
    ) -> QueryResult:
        """Run an online GraphRAG query with full subgraph, compaction, and reasoning trace."""
        if auto_load and not self.graph_store.get_all_entities():
            self.load(storage_dir)

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
            tot_beam_width=self.config.reasoning.tot_beam_width,
            tot_max_depth=self.config.reasoning.tot_max_depth,
            tot_max_llm_calls=self.config.reasoning.tot_max_llm_calls,
            semantic_index=self.semantic_index,
            enable_hybrid_bm25=self.config.retrieval.enable_hybrid_bm25,
            hybrid_bm25_weight=self.config.retrieval.hybrid_bm25_weight,
            rrf_k=self.config.retrieval.rrf_k,
            enable_query_to_triple=self.config.retrieval.enable_query_to_triple,
            top_k_triples=self.config.retrieval.top_k_triples,
        )
        return retriever.query(
            question,
            mode=mode or self.config.retrieval.default_mode,
            reasoning=reasoning or self.config.reasoning.default_paradigm,
        )

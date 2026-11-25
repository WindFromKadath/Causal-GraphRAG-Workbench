"""Unit tests for paper-faithful Iterative Graph-CoT v2 & Multi-Depth Graph-ToT v2 (BFS / DFS)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm_graphrag_kg.config import AppConfig, EnvSettings
from llm_graphrag_kg.core.schema import Entity, EvidenceItem, Relation, TextChunk
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline
from llm_graphrag_kg.providers.embedding.local_hash import LocalHashEmbeddingClient
from llm_graphrag_kg.providers.factory import create_fast_llm_client
from llm_graphrag_kg.providers.llm.mock import MockLLMClient
from llm_graphrag_kg.providers.llm.ollama_native import OllamaNativeLLMClient
from llm_graphrag_kg.providers.storage.local_vector import LocalVectorStore
from llm_graphrag_kg.providers.storage.networkx_store import NetworkXGraphStore
from llm_graphrag_kg.reasoning.tot_deep import DeepGraphToTEngine


class _BacktrackForcingPlannerLLM:
    """Deterministic planner LLM that forces DFS dead-end on H1.* so backtracking to H2.* triggers."""

    model = "mock-fast-4b"

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        _ = (system_prompt, temperature, json_mode, max_tokens)
        cids = list(dict.fromkeys(re.findall(r"===\s*Candidate\s+([A-Za-z0-9_.]+)", prompt)))
        evals = []
        for idx, cid in enumerate(cids):
            if "depth=1" in prompt:
                evals.append(
                    {
                        "candidate_id": cid,
                        "verdict": "likely",
                        "semantic_score": 0.85 if idx == 0 else 0.78,
                        "hypothesis": f"D1 branch {cid}",
                        "prune_reason": "",
                    }
                )
            elif cid.startswith("H1."):
                # Force dead-end on H1's children so DFS must backtrack to H2!
                evals.append(
                    {
                        "candidate_id": cid,
                        "verdict": "impossible",
                        "semantic_score": 0.15,
                        "hypothesis": f"Dead-end {cid}",
                        "prune_reason": "Contradicted by frequency spectrum",
                    }
                )
            else:
                evals.append(
                    {
                        "candidate_id": cid,
                        "verdict": "sure",
                        "semantic_score": 0.95,
                        "hypothesis": f"Verified root cause {cid}",
                        "prune_reason": "",
                    }
                )
        return json.dumps({"evaluations": evals}, ensure_ascii=False)

    def health_check(self) -> tuple[bool, str]:
        return True, "ok"


def _build_two_hop_graph() -> tuple[
    NetworkXGraphStore, LocalVectorStore, list[Entity], list[Relation], list[EvidenceItem]
]:
    gs = NetworkXGraphStore()
    vs = LocalVectorStore()
    embedder = LocalHashEmbeddingClient()

    chunks = [
        TextChunk(
            chunk_id="d1::chunk_0000",
            doc_id="d1",
            source="pump.md",
            text="Pump exhibits 2x vibration linked to shaft misalignment and bearing heating.",
            index=0,
            start_char=0,
            end_char=76,
        ),
        TextChunk(
            chunk_id="d1::chunk_0001",
            doc_id="d1",
            source="pump.md",
            text="Shaft misalignment requires Laser Alignment SOP per ISO 20816.",
            index=1,
            start_char=77,
            end_char=139,
        ),
        TextChunk(
            chunk_id="d1::chunk_0002",
            doc_id="d1",
            source="pump.md",
            text="Bearing spalling requires envelope shock pulses and Bearing Replacement SOP.",
            index=2,
            start_char=140,
            end_char=216,
        ),
    ]
    vs.add_chunks(chunks, embedder.embed([c.text for c in chunks]))

    ents = [
        Entity(
            entity_id="ent_vib",
            name="2x vibration",
            entity_type="Symptom",
            source_chunk_ids=["d1::chunk_0000"],
        ),
        Entity(
            entity_id="ent_bear",
            name="bearing spalling",
            entity_type="FaultMode",
            source_chunk_ids=["d1::chunk_0000", "d1::chunk_0002"],
        ),
        Entity(
            entity_id="ent_mis",
            name="shaft misalignment",
            entity_type="FaultMode",
            source_chunk_ids=["d1::chunk_0000", "d1::chunk_0001"],
        ),
        Entity(
            entity_id="ent_sop_b",
            name="Bearing Replacement SOP",
            entity_type="RemediationSOP",
            source_chunk_ids=["d1::chunk_0002"],
        ),
        Entity(
            entity_id="ent_sop_m",
            name="Laser Alignment SOP",
            entity_type="RemediationSOP",
            source_chunk_ids=["d1::chunk_0001"],
        ),
    ]
    for e in ents:
        gs.upsert_entity(e)

    rels = [
        Relation(
            relation_id="rel_1",
            source_id="ent_vib",
            target_id="ent_bear",
            relation_type="POSSIBLE_CAUSE",
            weight=1.5,
            source_chunk_ids=["d1::chunk_0000"],
        ),
        Relation(
            relation_id="rel_2",
            source_id="ent_vib",
            target_id="ent_mis",
            relation_type="CAUSES",
            weight=2.0,
            source_chunk_ids=["d1::chunk_0000"],
        ),
        Relation(
            relation_id="rel_3",
            source_id="ent_bear",
            target_id="ent_sop_b",
            relation_type="REMEDIATED_BY",
            weight=1.0,
            source_chunk_ids=["d1::chunk_0002"],
        ),
        Relation(
            relation_id="rel_4",
            source_id="ent_mis",
            target_id="ent_sop_m",
            relation_type="REMEDIATED_BY",
            weight=2.0,
            source_chunk_ids=["d1::chunk_0001"],
        ),
    ]
    for r in rels:
        gs.upsert_relation(r)

    evs = [
        EvidenceItem(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            source=c.source,
            text=c.text,
            score=0.9,
            matched_via="hybrid_rrf",
        )
        for c in chunks
    ]
    return gs, vs, ents, rels, evs


def test_create_fast_llm_client_routing(tmp_path: Path) -> None:
    cfg_same = AppConfig(
        project_root=tmp_path,
        env=EnvSettings(llm_provider="ollama_native", llm_model="qwen3.5:9b", llm_fast_model="qwen3.5:9b"),
    )
    assert create_fast_llm_client(cfg_same) is None

    cfg_fast = AppConfig(
        project_root=tmp_path,
        env=EnvSettings(
            llm_provider="ollama_native",
            llm_model="qwen3.5:9b",
            llm_fast_model="qwen3.5:4b",
            llm_fast_num_ctx=8192,
            llm_num_gpu=99,
        ),
    )
    fast = create_fast_llm_client(cfg_fast)
    assert isinstance(fast, OllamaNativeLLMClient)
    assert fast.model == "qwen3.5:4b"
    assert fast.num_ctx == 8192


def test_deep_tot_dfs_executes_genuine_backtracking_and_dual_model_telemetry() -> None:
    gs, vs, ents, rels, evs = _build_two_hop_graph()
    planner_harness = LLMHarness(_BacktrackForcingPlannerLLM())
    synth_harness = LLMHarness(MockLLMClient())

    engine = DeepGraphToTEngine(
        synth_harness,
        planner_harness=planner_harness,
        graph_store=gs,
        vector_store=vs,
        search_strategy="dfs",
        beam_width=2,
        max_depth=2,
        backtrack_threshold=0.62,
    )
    res, _, _, _ = engine.run(
        question="Pump shows 2x vibration; which root cause and SOP apply?",
        seed_entity_ids=["ent_vib"],
        entities=ents,
        relations=rels,
        communities=[],
        evidences=evs,
    )
    assert res.paradigm == "tot_dfs"
    assert res.trace_ledger is not None
    assert res.trace_ledger["planner_model"] == "mock-fast-4b"
    assert int(res.trace_ledger["max_depth_reached"]) == 2
    assert int(res.trace_ledger["backtrack_count"]) >= 1
    assert any(b.backtracked for b in res.branches)
    assert any(b.depth == 2 and b.status == "selected" and b.parent_branch_id for b in res.branches)


def test_pipeline_end_to_end_cot_iter_and_tot_bfs(tmp_path: Path) -> None:
    in_dir = tmp_path / "docs"
    st_dir = tmp_path / "storage"
    in_dir.mkdir()
    (in_dir / "pump.md").write_text(
        "# Centrifugal Pump Diagnosis\n\n"
        "离心泵机组出现2倍频振动升高时，轴系不对中是主要根因，应执行激光对中 SOP。\n"
        "滚动轴承外圈剥落由高频冲击脉冲表征，执行轴承润滑与更换 SOP。\n",
        encoding="utf-8",
    )

    cfg = AppConfig(project_root=tmp_path)
    pipeline = GraphRAGPipeline(
        cfg,
        llm=MockLLMClient(),
        planner_llm=MockLLMClient(),
        embedder=LocalHashEmbeddingClient(),
    )
    pipeline.build(in_dir, storage_dir=st_dir)

    # 1. Iterative Graph-CoT v2 (Thought -> Graph-Action -> Observation loop)
    res_iter = pipeline.query(
        "离心泵出现2倍频振动升高应执行什么SOP？",
        mode="ppr",
        reasoning="cot_iter",
        auto_load=False,
    )
    assert res_iter.metadata["reasoning_paradigm"] == "cot_iter"
    rm_iter = res_iter.metadata["reasoning"]
    assert isinstance(rm_iter, dict)
    assert any(st.get("tool_action") in {"ExpandNeighbors", "FetchChunk", "Finish"} for st in rm_iter["steps"])
    assert rm_iter["grounding_audit"]["passed"] is True
    assert res_iter.metadata["harness_ledger"]["planner_calls"] >= 1

    # 2. Multi-Depth ToT-BFS
    res_bfs = pipeline.query(
        "离心泵出现2倍频振动升高应执行什么SOP？",
        mode="ppr",
        reasoning="tot_bfs",
        auto_load=False,
    )
    assert res_bfs.metadata["reasoning_paradigm"] == "tot_bfs"
    rm_bfs = res_bfs.metadata["reasoning"]
    assert isinstance(rm_bfs, dict)
    assert rm_bfs["grounding_audit"]["passed"] is True
    assert any(b.get("depth", 0) >= 1 for b in rm_bfs["branches"])

"""Unit and integration tests for Stage 2.1 LLMHarness and 3-tier ContextCompactor."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from llm_graphrag_kg.config import load_config
from llm_graphrag_kg.core.schema import Community, Entity, EvidenceItem, Relation
from llm_graphrag_kg.harness.context_compactor import ContextCompactor, estimate_tokens
from llm_graphrag_kg.harness.runtime import LLMHarness
from llm_graphrag_kg.pipelines.workflow import GraphRAGPipeline


class _SampleOutput(BaseModel):
    answers: list[str] = Field(min_length=1)


class _FlakyJSONLLM:
    """Returns invalid JSON on first call and valid JSON after receiving repair feedback."""

    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        if self.calls == 1:
            return "NOT VALID JSON AT ALL"
        assert "VALIDATION_ERROR" in prompt
        return '{"answers": ["B", "D"]}'

    def health_check(self) -> tuple[bool, str]:
        return True, "ok"


def test_context_compactor_l0_and_l1_preserve_facts_and_shrink_tokens() -> None:
    entities = [
        Entity(
            entity_id="e_seed",
            name="轴系不对中",
            entity_type="FaultMode",
            description="由于平行或角度偏差引发的轴系不对中故障，造成2倍频振动升高。",
            source_chunk_ids=["doc_1::chunk_0000"],
        ),
        Entity(
            entity_id="e_sop",
            name="激光对中 SOP",
            entity_type="RemediationSOP",
            description="使用激光对中仪在冷态与热态补偿条件下调整电机垫片。",
            source_chunk_ids=["doc_2::chunk_0001"],
        ),
    ]
    # Add 25 unrelated distractor entities and relations to simulate high-connectivity bloat
    relations = [
        Relation(
            relation_id="r_core",
            source_id="e_seed",
            target_id="e_sop",
            relation_type="REMEDIATED_BY",
            description="轴系不对中应执行激光对中 SOP 进行校正。",
            weight=2.0,
            source_chunk_ids=["doc_2::chunk_0001"],
        )
    ]
    for idx in range(25):
        eid = f"e_noise_{idx}"
        entities.append(
            Entity(
                entity_id=eid,
                name=f"无关部件_{idx}",
                entity_type="Component",
                description="这是一段用于模拟高连通图子图膨胀的冗余背景描述文本。" * 3,
                source_chunk_ids=[f"doc_noise::chunk_{idx:04d}"],
            )
        )
        relations.append(
            Relation(
                relation_id=f"r_noise_{idx}",
                source_id=eid,
                target_id=f"e_noise_{(idx + 1) % 25}",
                relation_type="CONNECTED_TO",
                description="冗余背景连接边描述文本。" * 3,
                weight=1.0,
                source_chunk_ids=[f"doc_noise::chunk_{idx:04d}"],
            )
        )

    long_chunk = (
        "背景介绍：本车间共有多台辅助设备，日常巡检按排班执行，相关记录全部存入月度巡检台账与班组交接手册中。"
        "一般情况设备平稳运行无需干预，记录存档备查，运维人员只需按周例会复核运行小时数与台账完整性。"
        "关键异常：当离心泵轴系不对中时，2倍频振动升高至 7.8 mm/s 且温度上升 14℃。"
        "补充说明：巡检人员进入泵房需佩戴安全帽、耳塞与便携式点检仪并规范填写交接班电子日志和点检卡。"
        "历史背景：该泵房于去年完成常规管路保温改造与基础防腐刷漆作业，不影响当前主轴对中参数。"
        "处置规程：停机后执行激光对中 SOP，复测振幅回到 ISO 20816 B 区。"
    )
    evidences = [
        EvidenceItem(
            chunk_id="doc_2::chunk_0001",
            doc_id="doc_2",
            source="data/sample_fault_diagnosis/05.md",
            text=long_chunk,
            score=0.92,
            matched_via="vector+graph",
        )
    ]
    communities = [
        Community(
            community_id=0,
            title="Community 0: 轴系不对中",
            summary="围绕轴系不对中与激光对中 SOP 的社区摘要。",
            entity_ids=["e_seed", "e_sop"],
            relation_count=1,
        )
    ]

    compactor = ContextCompactor(max_entities=8, max_relations=6, max_sentences_per_chunk=2)
    compacted = compactor.compact_graphrag_context(
        question="轴系不对中时应该执行什么激光对中 SOP？",
        seed_entity_ids=["e_seed"],
        entities=entities,
        relations=relations,
        communities=communities,
        evidences=evidences,
    )

    assert compacted.report.compacted_tokens_est < compacted.report.raw_tokens_est
    assert compacted.report.compression_ratio >= 0.4
    assert compacted.report.dropped_nodes > 0
    assert "L0_subgraph_pruning" in compacted.report.applied_levels
    assert "L1_extractive_evidence" in compacted.report.applied_levels
    assert "激光对中 SOP" in compacted.full_text
    assert "doc_2::chunk_0001" in compacted.full_text
    assert "7.8 mm/s" in compacted.evidences_block


def test_l2_state_folding_preserves_numbers_and_provenance() -> None:
    compactor = ContextCompactor()
    folded = compactor.fold_state_l2(
        label="H3_winding_fault",
        status="pruned",
        raw_text="测点记录定子绕组泄漏流量 31.8 L/h，绝缘电阻正常，与假设矛盾，已淘汰。",
        entities=["定子绕组", "绝缘电阻"],
        chunk_ids=["doc_1::chunk_0003"],
    )
    line = folded.to_compact_line()
    assert "PRUNED H3_winding_fault" in line
    assert "31.8 L/h" in folded.retained_numbers
    assert "doc_1::chunk_0003" in line


def test_harness_structured_auto_repair_and_guardrail() -> None:
    flaky = _FlakyJSONLLM()
    harness = LLMHarness(flaky, json_repair_retries=1, max_calls=3)
    parsed, _ = harness.run_structured(
        "Pick two options",
        _SampleOutput,
        step_name="mcqa_test",
        extra_validator=lambda out: None if len(out.answers) == 2 else "need 2 answers",
    )
    assert parsed.answers == ["B", "D"]
    assert harness.ledger.total_calls == 1
    assert harness.ledger.total_repairs == 1

    # Verify max_calls guardrail blocks runaway loops
    from llm_graphrag_kg.providers.llm.mock import MockLLMClient

    strict_harness = LLMHarness(MockLLMClient(), max_calls=1)
    strict_harness.run_text("hello", step_name="call_1")
    with pytest.raises(RuntimeError, match="max_calls"):
        strict_harness.run_text("hello again", step_name="call_2")


def test_pipeline_query_records_compaction_and_ledger_metadata(tmp_path: Path) -> None:
    cfg = load_config(
        config_path=Path("configs/fault_diagnosis.yaml"),
        project_root=Path.cwd(),
        llm_provider_override="mock",
        embedding_provider_override="local_hash",
    )
    pipeline = GraphRAGPipeline(cfg)
    pipeline.build(storage_dir=tmp_path / "storage")
    res = pipeline.query(
        "离心泵机组出现2倍频振动升高时应执行什么激光对中 SOP？",
        mode="hybrid",
        auto_load=False,
        enable_compact=True,
    )
    assert "compaction" in res.metadata
    assert "harness_ledger" in res.metadata
    assert estimate_tokens(res.answer) > 0

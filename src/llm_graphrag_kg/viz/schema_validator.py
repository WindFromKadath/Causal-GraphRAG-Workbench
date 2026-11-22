"""SHACL-lite full-graph ontology & provenance compliance validator (Stage 2.4).

Scans 100% of entities and relations in a `BaseGraphStore` + `BaseVectorStore` to detect:
- Generic KG hygiene issues: type whitelist violations, self-loops, dangling endpoints,
  ghost entities/triples missing resolvable chunk provenance, and isolated nodes.
- Strict 4-layer ISO FailureSensorIQ ontology constraints (auto-activated when the graph
  uses Equipment / FailureMode / FailureCategory / SignalChannel types).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from llm_graphrag_kg.adapters.failuresensoriq import normalize_sensor_name
from llm_graphrag_kg.config import AppConfig
from llm_graphrag_kg.core.interfaces import BaseGraphStore, BaseVectorStore

ISO_ENTITY_TYPES = {"Equipment", "FailureMode", "FailureCategory", "SignalChannel"}
ISO_RELATION_RULES: dict[str, tuple[str, str]] = {
    "HAS_FAILURE_MODE": ("Equipment", "FailureMode"),
    "INSTANCE_OF": ("FailureMode", "FailureCategory"),
    "MONITORED_BY": ("FailureMode", "SignalChannel"),
}


class SHACLCheckResult(BaseModel):
    """Single SHACL-lite rule check result."""

    check_id: str
    title: str
    severity: Literal["error", "warning", "info"] = "error"
    passed: bool
    violation_count: int = 0
    sample_violations: list[str] = Field(default_factory=list)
    description: str = ""


class SHACLValidationReport(BaseModel):
    """Full-graph SHACL-lite compliance report."""

    profile: str  # "iso_failuresensoriq" | "generic"
    overall_status: Literal["PASS", "WARN", "FAIL"]
    compliance_score: float = Field(ge=0.0, le=1.0)
    total_entities_scanned: int
    total_relations_scanned: int
    total_chunks_indexed: int
    error_count: int
    warning_count: int
    checks: list[SHACLCheckResult] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# SHACL-lite 本体合规质检报告（Profile: `{self.profile}`）",
            "",
            f"- **总体判定**：**{self.overall_status}**（合规率 **{self.compliance_score * 100:.1f}%**）",
            f"- **全量扫描范围**：`{self.total_entities_scanned}` 实体 / "
            f"`{self.total_relations_scanned}` 关系 / `{self.total_chunks_indexed}` 原文切片",
            f"- **违例汇总**：Error `{self.error_count}` | Warning `{self.warning_count}`",
            "",
            "| 状态 | 规则 ID | 质检项 | 严重级 | 违例数 | 说明 |",
            "|---|---|---|---|---:|---|",
        ]
        for chk in self.checks:
            icon = "PASS" if chk.passed else ("FAIL" if chk.severity == "error" else "WARN")
            lines.append(
                f"| `{icon}` | `{chk.check_id}` | {chk.title} | `{chk.severity}` | "
                f"{chk.violation_count} | {chk.description} |"
            )
        for chk in self.checks:
            if chk.sample_violations:
                lines.extend(["", f"### 违例抽样：`{chk.check_id}` ({chk.title})"])
                for item in chk.sample_violations[:10]:
                    lines.append(f"- {item}")
        return "\n".join(lines) + "\n"


def validate_kg_shacl_lite(
    graph_store: BaseGraphStore,
    vector_store: BaseVectorStore,
    config: AppConfig | None = None,
) -> SHACLValidationReport:
    """Execute 100% full-graph SHACL-lite compliance validation."""
    entities = graph_store.get_all_entities()
    relations = graph_store.get_all_relations()
    chunks = vector_store.get_all_chunks()

    ent_map = {e.entity_id: e for e in entities}
    chunk_ids = {c.chunk_id for c in chunks}
    present_types = {e.entity_type for e in entities}

    is_iso_profile = bool(present_types) and present_types.issubset(ISO_ENTITY_TYPES) and (
        "Equipment" in present_types and "FailureMode" in present_types
    )
    profile = "iso_failuresensoriq" if is_iso_profile else "generic"

    checks: list[SHACLCheckResult] = []

    # 1. Entity Type Whitelist Check
    whitelist = (
        set(config.extraction.entity_types)
        if (config is not None and config.extraction.entity_types)
        else (ISO_ENTITY_TYPES if is_iso_profile else set())
    )
    if whitelist:
        bad_types = [
            f"{e.name} (`{e.entity_id}`): unexpected type `{e.entity_type}`"
            for e in entities
            if e.entity_type not in whitelist
        ]
        checks.append(
            SHACLCheckResult(
                check_id="SHACL_01_TYPE_WHITELIST",
                title="实体类型白名单合规",
                severity="error",
                passed=len(bad_types) == 0,
                violation_count=len(bad_types),
                sample_violations=bad_types[:10],
                description=f"允许类型: {', '.join(sorted(whitelist))}",
            )
        )

    # 2. Dangling Relation Endpoints Check
    dangling = [
        f"{r.relation_id} ({r.source_id} --[{r.relation_type}]--> {r.target_id})"
        for r in relations
        if r.source_id not in ent_map or r.target_id not in ent_map
    ]
    checks.append(
        SHACLCheckResult(
            check_id="SHACL_02_NO_DANGLING_ENDPOINTS",
            title="关系端点完整性（无悬空边）",
            severity="error",
            passed=len(dangling) == 0,
            violation_count=len(dangling),
            sample_violations=dangling[:10],
            description="校验所有三元组的 source_id 与 target_id 均存在于实体表",
        )
    )

    # 3. No Self-Loop Edges
    self_loops = [
        f"{r.relation_id}: {ent_map[r.source_id].name if r.source_id in ent_map else r.source_id} "
        f"--[{r.relation_type}]--> self"
        for r in relations
        if r.source_id == r.target_id
    ]
    checks.append(
        SHACLCheckResult(
            check_id="SHACL_03_NO_SELF_LOOPS",
            title="无自环边校验",
            severity="warning",
            passed=len(self_loops) == 0,
            violation_count=len(self_loops),
            sample_violations=self_loops[:10],
            description="检测指向自身的无效/冗余关系边",
        )
    )

    # 4. Ghost Entities & Ghost Triples (Missing or Unresolvable Provenance Chunk IDs)
    ghost_items: list[str] = []
    for e in entities:
        if not e.source_chunk_ids:
            ghost_items.append(f"Entity `{e.name}` ({e.entity_id}): 缺少 source_chunk_ids")
        elif chunk_ids and not any(cid in chunk_ids for cid in e.source_chunk_ids):
            ghost_items.append(f"Entity `{e.name}` ({e.entity_id}): chunk_id 无法反查原文")
    for r in relations:
        if not r.source_chunk_ids:
            ghost_items.append(f"Relation `{r.relation_id}` ({r.relation_type}): 缺少 source_chunk_ids")
        elif chunk_ids and not any(cid in chunk_ids for cid in r.source_chunk_ids):
            ghost_items.append(f"Relation `{r.relation_id}` ({r.relation_type}): chunk_id 无法反查原文")
    checks.append(
        SHACLCheckResult(
            check_id="SHACL_04_PROVENANCE_COMPLETENESS",
            title="原文溯源完整性（零幽灵实体/三元组）",
            severity="error",
            passed=len(ghost_items) == 0,
            violation_count=len(ghost_items),
            sample_violations=ghost_items[:10],
            description="所有实体与三元组均须绑定可反查的原文 chunk_id",
        )
    )

    # 5. Isolated Entities (Degree == 0)
    deg: dict[str, int] = defaultdict(int)
    for r in relations:
        deg[r.source_id] += 1
        deg[r.target_id] += 1
    isolated = [
        f"{e.name} (`{e.entity_type}` | {e.entity_id})" for e in entities if deg[e.entity_id] == 0
    ]
    checks.append(
        SHACLCheckResult(
            check_id="SHACL_05_NO_ISOLATED_ENTITIES",
            title="无孤立实体校验",
            severity="warning",
            passed=len(isolated) == 0,
            violation_count=len(isolated),
            sample_violations=isolated[:10],
            description="检测度数为 0 的孤岛节点（通常需补充关系或合并同义词）",
        )
    )

    # 6-8. Strict 4-Layer ISO FailureSensorIQ Ontology Constraints
    if is_iso_profile:
        bad_domain_range: list[str] = []
        fm_has_asset_edge: set[str] = set()
        fm_has_sensor_edge: set[str] = set()
        for r in relations:
            src = ent_map.get(r.source_id)
            tgt = ent_map.get(r.target_id)
            if not src or not tgt:
                continue
            expected = ISO_RELATION_RULES.get(r.relation_type)
            if expected is None:
                bad_domain_range.append(
                    f"{r.relation_id}: 未知 ISO 关系类型 `{r.relation_type}` "
                    f"({src.entity_type} -> {tgt.entity_type})"
                )
            else:
                exp_s, exp_t = expected
                if src.entity_type != exp_s or tgt.entity_type != exp_t:
                    bad_domain_range.append(
                        f"{r.relation_id}: `{r.relation_type}` 期望 {exp_s}->{exp_t}，"
                        f"实际 {src.entity_type}->{tgt.entity_type}"
                    )
            if r.relation_type == "HAS_FAILURE_MODE" and src.entity_type == "Equipment":
                fm_has_asset_edge.add(r.target_id)
            if r.relation_type == "MONITORED_BY" and tgt.entity_type == "SignalChannel":
                fm_has_sensor_edge.add(r.source_id)

        checks.append(
            SHACLCheckResult(
                check_id="ISO_06_DOMAIN_RANGE_CONSTRAINTS",
                title="ISO 四层关系定义域/值域约束",
                severity="error",
                passed=len(bad_domain_range) == 0,
                violation_count=len(bad_domain_range),
                sample_violations=bad_domain_range[:10],
                description=(
                    "HAS_FAILURE_MODE(Equipment->FailureMode), "
                    "INSTANCE_OF(FailureMode->FailureCategory), "
                    "MONITORED_BY(FailureMode->SignalChannel)"
                ),
            )
        )

        fm_scoping_issues: list[str] = []
        for e in entities:
            if e.entity_type != "FailureMode":
                continue
            if not e.metadata.get("asset"):
                fm_scoping_issues.append(f"{e.name}: 缺少 asset 隔离元数据")
            if e.entity_id not in fm_has_asset_edge:
                fm_scoping_issues.append(f"{e.name}: 未挂接 Equipment (缺 HAS_FAILURE_MODE)")
            if e.entity_id not in fm_has_sensor_edge:
                fm_scoping_issues.append(f"{e.name}: 未挂接任何 SignalChannel (缺 MONITORED_BY)")
        checks.append(
            SHACLCheckResult(
                check_id="ISO_07_FAILURE_MODE_SCOPING",
                title="FailureMode 资产隔离与传感器覆盖约束",
                severity="error",
                passed=len(fm_scoping_issues) == 0,
                violation_count=len(fm_scoping_issues),
                sample_violations=fm_scoping_issues[:10],
                description="确保每个失效模式均按设备隔离且至少具备 1 条监控传感通道",
            )
        )

        raw_typo_leaks: list[str] = []
        for e in entities:
            if e.entity_type != "SignalChannel":
                continue
            canonical_s, merged = normalize_sensor_name(e.name)
            if merged and canonical_s != e.name.lower():
                raw_typo_leaks.append(
                    f"SignalChannel `{e.name}` 命中已知拼写错误词表（应归一为 `{canonical_s}`）"
                )
        checks.append(
            SHACLCheckResult(
                check_id="ISO_08_SIGNAL_ER_HYGIENE",
                title="SignalChannel 实体消歧（ER）洁净度校验",
                severity="error",
                passed=len(raw_typo_leaks) == 0,
                violation_count=len(raw_typo_leaks),
                sample_violations=raw_typo_leaks[:10],
                description="检查传感器规范名中是否存在未归一化的原始拼写错误",
            )
        )

    passed_count = sum(1 for c in checks if c.passed)
    error_count = sum(c.violation_count for c in checks if not c.passed and c.severity == "error")
    warning_count = sum(
        c.violation_count for c in checks if not c.passed and c.severity == "warning"
    )
    score = round(passed_count / max(len(checks), 1), 4)

    if error_count > 0:
        status: Literal["PASS", "WARN", "FAIL"] = "FAIL"
    elif warning_count > 0:
        status = "WARN"
    else:
        status = "PASS"

    return SHACLValidationReport(
        profile=profile,
        overall_status=status,
        compliance_score=score,
        total_entities_scanned=len(entities),
        total_relations_scanned=len(relations),
        total_chunks_indexed=len(chunks),
        error_count=error_count,
        warning_count=warning_count,
        checks=checks,
    )

"""Deterministic offline Mock LLM client for zero-cost verification and unit tests."""

from __future__ import annotations

import json
import re


class MockLLMClient:
    """Rule-based mock LLM that produces structured KG extractions and grounded QA answers."""

    # Curated technical vocabulary for realistic offline extraction on sample/tech docs.
    KNOWN_TERMS: list[tuple[str, str]] = [
        ("GraphRAG", "Method"),
        ("Knowledge Graph", "Concept"),
        ("知识图谱", "Concept"),
        ("LLM", "Technology"),
        ("大语言模型", "Technology"),
        ("Naive RAG", "Method"),
        ("向量检索", "Method"),
        ("Local Search", "Method"),
        ("Global Search", "Method"),
        ("Hybrid Search", "Method"),
        ("Louvain", "Method"),
        ("NetworkX", "Technology"),
        ("Neo4j", "Technology"),
        ("PyVis", "Technology"),
        ("Streamlit", "Technology"),
        ("Gephi", "Technology"),
        ("GraphML", "Technology"),
        ("uv", "Technology"),
        ("Pydantic", "Technology"),
        ("Entity Resolution", "Method"),
        ("实体对齐", "Method"),
        ("社区发现", "Method"),
        ("社区摘要", "Component"),
        ("多跳推理", "Concept"),
        ("原文溯源", "Concept"),
        ("BaseGraphStore", "Component"),
        ("BaseLLMClient", "Component"),
        ("BaseVectorStore", "Component"),
        ("OpenAI-compatible", "Technology"),
        ("vLLM", "Technology"),
        ("Ollama", "Technology"),
        # Industrial engineering deployment & fault diagnosis terms
        ("离心泵机组", "Equipment"),
        ("三相异步电机", "Equipment"),
        ("齿轮箱", "Component"),
        ("滚动轴承", "Component"),
        ("弹性联轴器", "Component"),
        ("叶轮", "Component"),
        ("润滑系统", "Component"),
        ("振动加速度传感器", "Sensor"),
        ("温度传感器", "Sensor"),
        ("电流互感器", "Sensor"),
        ("转速编码器", "Sensor"),
        ("声发射传感器", "Sensor"),
        ("高频冲击脉冲", "Symptom"),
        ("2倍频振动升高", "Symptom"),
        ("1倍频振动升高", "Symptom"),
        ("轴承温升异常", "Symptom"),
        ("电机定子电流波动", "Symptom"),
        ("频谱混叠失真", "Symptom"),
        ("特征频率 BPFO", "Symptom"),
        ("滚动轴承外圈剥落", "FaultMode"),
        ("轴系不对中", "FaultMode"),
        ("转子动不平衡", "FaultMode"),
        ("润滑脂劣化", "RootCause"),
        ("附加轴向载荷", "RootCause"),
        ("FFT 频谱分析", "Algorithm"),
        ("包络解调", "Algorithm"),
        ("小波包分解", "Algorithm"),
        ("孤立森林", "Algorithm"),
        ("Transformer 时序模型", "Algorithm"),
        ("物理引导弱监督诊断", "Algorithm"),
        ("GraphRAG 根因推理", "Algorithm"),
        ("边缘采集网关", "DeploymentComponent"),
        ("OPC-UA 协议", "DeploymentComponent"),
        ("MQTT 消息总线", "DeploymentComponent"),
        ("ONNX Runtime", "DeploymentComponent"),
        ("TensorRT 推理引擎", "DeploymentComponent"),
        ("Docker 容器平台", "DeploymentComponent"),
        ("InfluxDB 时序数据库", "DeploymentComponent"),
        ("Neo4j 图数据库", "DeploymentComponent"),
        ("模型漂移监控", "DeploymentComponent"),
        ("ISO 20816 振动标准", "Standard"),
        ("激光对中 SOP", "RemediationSOP"),
        ("现场动平衡校正 SOP", "RemediationSOP"),
        ("轴承润滑与更换 SOP", "RemediationSOP"),
        ("边缘配置回滚 SOP", "RemediationSOP"),
    ]

    RELATION_HINTS: list[tuple[str, str]] = [
        ("依赖", "DEPENDS_ON"),
        ("基于", "BASED_ON"),
        ("结合", "COMBINES"),
        ("使用", "USES"),
        ("导出", "EXPORTS_TO"),
        ("支持", "SUPPORTS"),
        ("解决", "SOLVES"),
        ("替代", "REPLACES"),
        ("迁移", "MIGRATES_TO"),
        ("包含", "CONTAINS"),
        ("连接", "CONNECTS"),
        ("uses", "USES"),
        ("combines", "COMBINES"),
        ("supports", "SUPPORTS"),
        ("solves", "SOLVES"),
        ("exports", "EXPORTS_TO"),
        ("migrates", "MIGRATES_TO"),
    ]

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        _ = (system_prompt, temperature, max_tokens)
        if "GRAPH_COT_TASK" in prompt:
            return self._mock_graph_cot(prompt)
        if "Options:\n" in prompt and (
            "Answer: <" in prompt or '{"answers":' in prompt or "MCQA_COT_TASK" in prompt
        ):
            return self._mock_mcqa_answer(prompt)
        if json_mode or "ENTITY_EXTRACTION_TASK" in prompt:
            return self._mock_extract(prompt)
        if "COMMUNITY_SUMMARY_TASK" in prompt:
            return self._mock_community_summary(prompt)
        return self._mock_answer(prompt)

    def health_check(self) -> tuple[bool, str]:
        return True, "MockLLMClient is ready (offline deterministic mode)."

    def _mock_extract(self, prompt: str) -> str:
        marker = "TEXT_START"
        text = prompt.split(marker, 1)[1] if marker in prompt else prompt
        if "TEXT_END" in text:
            text = text.split("TEXT_END", 1)[0]
        text = text.strip()

        found: list[tuple[str, str]] = []
        seen_lower: set[str] = set()
        for term, etype in self.KNOWN_TERMS:
            if term.lower() in text.lower() and term.lower() not in seen_lower:
                found.append((term, etype))
                seen_lower.add(term.lower())

        # Fallback: extract capitalized multi-letter English tokens if nothing matched.
        if not found:
            for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text):
                if token.lower() not in seen_lower and token not in {"TEXT", "START", "END", "JSON"}:
                    found.append((token, "Concept"))
                    seen_lower.add(token.lower())
                if len(found) >= 6:
                    break

        entities = [
            {
                "name": name,
                "entity_type": etype,
                "description": f"从文本片段中识别到的 {etype} 实体 `{name}`。",
            }
            for name, etype in found
        ]

        relations: list[dict[str, str]] = []
        sentences = [s.strip() for s in re.split(r"[。！？.!?\n]+", text) if s.strip()]
        seen_edges: set[tuple[str, str, str]] = set()
        for sent in sentences:
            in_sent = [name for name, _ in found if name.lower() in sent.lower()]
            if len(in_sent) < 2:
                continue
            rel_type = "RELATED_TO"
            for keyword, mapped in self.RELATION_HINTS:
                if keyword in sent.lower():
                    rel_type = mapped
                    break
            src = in_sent[0]
            for tgt in in_sent[1:]:
                key = (src, rel_type, tgt)
                if src != tgt and key not in seen_edges:
                    seen_edges.add(key)
                    relations.append(
                        {
                            "source": src,
                            "target": tgt,
                            "relation_type": rel_type,
                            "description": sent[:120],
                        }
                    )

        return json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False)

    def _mock_community_summary(self, prompt: str) -> str:
        lines = [line.strip("- ").strip() for line in prompt.splitlines() if line.strip().startswith("-")]
        preview = "、".join(lines[:5]) if lines else "核心实体簇"
        return f"[Mock 社区摘要] 该主题社区围绕 {preview} 等节点构建，聚合了相关的实体关系与上下文证据。"

    def _mock_answer(self, prompt: str) -> str:
        if "Options:\n" in prompt and "Answer: <LETTER>" in prompt:
            return self._mock_mcqa_answer(prompt)

        question = ""
        for line in prompt.splitlines():
            if line.startswith("Question:"):
                question = line.removeprefix("Question:").strip()
                break

        triple_lines = [
            line.strip()
            for line in prompt.splitlines()
            if "-->" in line or line.strip().startswith("- [")
        ]
        summary_points = "\n".join(triple_lines[:6]) if triple_lines else "（当前知识库未命中显著三元组）"
        return (
            f"[Mock 回答] 针对问题“{question}”，基于当前检索到的知识图谱子图与原文证据综合如下：\n"
            f"{summary_points}\n"
            "提示：当前处于离线 Mock 模式，切换到 `.env` 中的 OpenAI 兼容或本机 "
            "Ollama/vLLM 模型即可获得完整语义推理回答。"
        )

    @staticmethod
    def _mock_graph_cot(prompt: str) -> str:
        """Deterministic grounded 4-step Graph-CoT JSON responder for offline verification."""
        question = ""
        for line in prompt.splitlines():
            if line.startswith("Question:"):
                question = line.removeprefix("Question:").strip()
                break

        ent_ids = list(dict.fromkeys(re.findall(r"\[(ent_[a-z0-9]+)\]", prompt)))
        rel_ids = list(dict.fromkeys(re.findall(r"\[(rel_[a-z0-9]+)\]", prompt)))
        chunk_ids = list(
            dict.fromkeys(re.findall(r"([a-z0-9_]+::chunk_\d+)", prompt))
        )

        steps = [
            {
                "step_id": 1,
                "stage": "anchor",
                "claim": f"定位问题核心锚点实体（共命中 {len(ent_ids)} 个图谱实体）。",
                "cited_entity_ids": ent_ids[:3],
                "cited_relation_ids": [],
                "cited_chunk_ids": [],
            },
            {
                "step_id": 2,
                "stage": "graph_traverse",
                "claim": f"沿知识图谱显性边展开多跳关联路径（共引用 {len(rel_ids[:3])} 条关系）。",
                "cited_entity_ids": ent_ids[:3],
                "cited_relation_ids": rel_ids[:3],
                "cited_chunk_ids": [],
            },
            {
                "step_id": 3,
                "stage": "evidence_check",
                "claim": "核对来源原文切片中的诊断规则、数值阈值与运维细节。",
                "cited_entity_ids": [],
                "cited_relation_ids": [],
                "cited_chunk_ids": chunk_ids[:3],
            },
            {
                "step_id": 4,
                "stage": "synthesis",
                "claim": f"综合图谱路径与原文证据，输出关于“{question[:40]}”的扎根结论。",
                "cited_entity_ids": ent_ids[:2],
                "cited_relation_ids": rel_ids[:2],
                "cited_chunk_ids": chunk_ids[:2],
            },
        ]
        cite_str = f" [{', '.join(chunk_ids[:2])}]" if chunk_ids else ""
        payload = {
            "anchor_entities": ent_ids[:3],
            "steps": steps,
            "final_answer": (
                f"[Mock Graph-CoT] 针对问题“{question}”，已完成 4 步显性图谱推导并完成忠实度校验{cite_str}。"
            ),
            "confidence": 0.92 if (ent_ids or rel_ids or chunk_ids) else 0.5,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _mock_mcqa_answer(prompt: str) -> str:
        """Deterministic MCQA responder supporting single-choice, multi-choice, JSON & CoT mode."""
        opts_block = prompt.split("Options:\n", 1)[1].split("\n\n", 1)[0]
        options: list[tuple[str, str]] = []
        for line in opts_block.splitlines():
            m = re.match(r"^\s*([A-Za-z0-9]+)[\.\)]\s*(.+?)\s*$", line)
            if m:
                options.append((m.group(1).upper(), m.group(2).lower()))
        if not options:
            return json.dumps({"answers": ["A"]})

        count_match = re.search(r"Select\s+(?:all\s+|exactly\s+)?(\d+)", prompt, flags=re.I)
        expected = int(count_match.group(1)) if count_match else 1

        ctx = prompt.split("Question:", 1)[0].lower()
        q_part = prompt.split("Question:", 1)[1].split("Options:", 1)[0].lower()
        is_negative = any(
            kw in q_part
            for kw in (
                "not ",
                "irrelevant",
                "least ",
                "except",
                "eliminate",
                "disregard",
                "unimportant",
                "non-relevant",
                "exclude",
                "inapplicable",
            )
        )

        # Parse tri-state lines if Scope-Aware L0 Context is present
        linked_line = ""
        unlinked_line = ""
        for line in ctx.splitlines():
            if "other valid" in line and "not " in line:
                unlinked_line += " " + line
            elif "monitored_by" in line or "monitors only" in line:
                linked_line += " " + line

        option_checks: list[dict[str, str]] = []
        chosen: list[str] = []
        if "structured knowledge graph facts" in ctx or "retrieved text chunks" in ctx:
            for oid, opt in options:
                if linked_line or unlinked_line:
                    if opt in linked_line:
                        status = "LINKED"
                    elif opt in unlinked_line:
                        status = "IN_ASSET_UNLINKED"
                    else:
                        status = "OUT_OF_ASSET"
                else:
                    status = "LINKED" if opt in ctx else "IN_ASSET_UNLINKED"
                option_checks.append(
                    {
                        "option_id": oid,
                        "option_text": opt,
                        "kg_status": status,
                        "matched_kg_item": opt if status != "OUT_OF_ASSET" else "",
                    }
                )
                want = "IN_ASSET_UNLINKED" if is_negative else "LINKED"
                if status == want and len(chosen) < expected:
                    chosen.append(oid)

        while len(chosen) < expected and len(chosen) < len(options):
            for oid, _ in options:
                if oid not in chosen:
                    chosen.append(oid)
                    break

        if "MCQA_COT_TASK" in prompt:
            return json.dumps(
                {
                    "target_anchor": "mock_anchor",
                    "polarity": "irrelevant" if is_negative else "relevant",
                    "option_checks": option_checks,
                    "reasoning_summary": "Matched strictly against Scope-Aware KG links.",
                    "answers": chosen,
                },
                ensure_ascii=False,
            )
        if '{"answers":' in prompt:
            return json.dumps({"answers": chosen})
        return f"[Mock MCQA] Matched from context.\nAnswer: {', '.join(chosen)}"

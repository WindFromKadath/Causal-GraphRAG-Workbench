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
        ("Magpie", "Technology"),
        ("Ollama", "Technology"),
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
    ) -> str:
        _ = (system_prompt, temperature)
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
            "提示：当前处于离线 Mock 模式，切换到 `.env` 中的 OpenAI 兼容或本机 Magpie 模型即可获得完整语义推理回答。"
        )

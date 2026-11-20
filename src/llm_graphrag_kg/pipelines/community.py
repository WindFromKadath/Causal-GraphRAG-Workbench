"""Louvain community detection and optional LLM-based community summarization."""

from __future__ import annotations

import networkx as nx

from llm_graphrag_kg.core.interfaces import BaseGraphStore, BaseLLMClient
from llm_graphrag_kg.core.schema import Community


class CommunityDetector:
    """Partitions the KG into topical communities for Global GraphRAG retrieval and visualization."""

    def __init__(
        self,
        *,
        resolution: float = 1.0,
        seed: int = 42,
        summarize_with_llm: bool = False,
    ) -> None:
        self.resolution = resolution
        self.seed = seed
        self.summarize_with_llm = summarize_with_llm

    def detect_and_summarize(
        self,
        graph_store: BaseGraphStore,
        llm: BaseLLMClient | None = None,
    ) -> list[Community]:
        nx_graph = graph_store.to_networkx()
        if nx_graph.number_of_nodes() == 0:
            graph_store.set_communities([])
            return []

        undirected = nx_graph.to_undirected()
        if undirected.number_of_edges() == 0:
            raw_communities = [{node} for node in undirected.nodes()]
        else:
            raw_communities = list(
                nx.algorithms.community.louvain_communities(
                    undirected,
                    weight="weight",
                    resolution=self.resolution,
                    seed=self.seed,
                )
            )

        # Sort communities deterministically by size descending, then node IDs
        raw_communities.sort(key=lambda nodes: (-len(nodes), sorted(nodes)[0]))

        communities: list[Community] = []
        for idx, node_set in enumerate(raw_communities):
            entity_ids = sorted(node_set)
            subgraph = nx_graph.subgraph(entity_ids)
            names = [str(nx_graph.nodes[nid].get("name", nid)) for nid in entity_ids]
            title = f"Community {idx}: " + " / ".join(names[:3])

            if self.summarize_with_llm and llm is not None:
                summary = self._summarize_via_llm(nx_graph, entity_ids, llm)
            else:
                summary = self._extractive_summary(nx_graph, entity_ids)

            communities.append(
                Community(
                    community_id=idx,
                    title=title,
                    summary=summary,
                    entity_ids=entity_ids,
                    relation_count=subgraph.number_of_edges(),
                )
            )

        graph_store.set_communities(communities)
        return communities

    @staticmethod
    def _extractive_summary(graph: nx.DiGraph, entity_ids: list[str]) -> str:
        sub = graph.subgraph(entity_ids)
        names = [str(graph.nodes[n].get("name", n)) for n in entity_ids]
        rel_samples: list[str] = []
        for u, v, data in list(sub.edges(data=True))[:6]:
            u_name = graph.nodes[u].get("name", u)
            v_name = graph.nodes[v].get("name", v)
            rel_samples.append(f"{u_name} --[{data.get('relation_type', 'RELATED_TO')}]--> {v_name}")
        rel_text = "；".join(rel_samples) if rel_samples else "暂无内部边连接"
        return f"包含 {len(entity_ids)} 个实体（{'、'.join(names[:8])}）。核心关系：{rel_text}。"

    @staticmethod
    def _summarize_via_llm(graph: nx.DiGraph, entity_ids: list[str], llm: BaseLLMClient) -> str:
        sub = graph.subgraph(entity_ids)
        entity_lines = [
            f"- {graph.nodes[n].get('name', n)} ({graph.nodes[n].get('entity_type', 'Concept')}): "
            f"{graph.nodes[n].get('description', '')}"
            for n in entity_ids[:20]
        ]
        edge_lines = [
            f"- {graph.nodes[u].get('name', u)} --[{d.get('relation_type', 'RELATED_TO')}]--> "
            f"{graph.nodes[v].get('name', v)}"
            for u, v, d in list(sub.edges(data=True))[:20]
        ]
        prompt = (
            "COMMUNITY_SUMMARY_TASK\n"
            "请用 2-3 句中文简明总结以下知识图谱社区的核心主题、关键实体与相互关系：\n\n"
            "Entities:\n"
            + "\n".join(entity_lines)
            + "\n\nRelations:\n"
            + ("\n".join(edge_lines) if edge_lines else "- (none)")
        )
        return llm.complete(prompt, temperature=0.2, json_mode=False)

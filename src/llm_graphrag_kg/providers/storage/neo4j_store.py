"""Neo4j 5.26 enterprise Knowledge Graph & HyperGraph store (Stage 4.1).

Implements `BaseGraphStore` plus Stage 3.4/4.1 enterprise capabilities:
- Composite uniqueness constraints on `(tenant_id, entity_id)` and `(tenant_id, hyperedge_id)`
- Pre-Filter multi-tenant (`tenant_id`), soft-delete (`is_deleted = false`), and RBAC (`acl`)
  enforced inside every Cypher traversal (`get_k_hop_subgraph`, `find_entities_by_query`, etc.)
- First-class `:HyperEdge` mediator nodes (`(:Entity)-[:CONDITION]->(:HyperEdge)-[:TARGET]->(:Entity)`)
- ACL-partitioned `:Community` summaries (`acl_partition`) preventing cross-privilege summary leakage
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from llm_graphrag_kg.core.schema import Community, Entity, HyperEdge, Relation


def _import_neo4j() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore[import-not-found]

        return GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "Neo4j backend requires enterprise dependencies. Run: `uv sync --extra enterprise`."
        ) from exc


def _parse_roles(roles: list[str] | str | None, default_roles: list[str]) -> list[str]:
    if roles is None:
        return list(default_roles)
    if isinstance(roles, str):
        parsed = [r.strip() for r in roles.split(",") if r.strip()]
        return parsed or list(default_roles)
    return [r.strip() for r in roles if r.strip()] or list(default_roles)


class Neo4jGraphStore:
    """Enterprise Neo4j 5.26 adapter implementing `BaseGraphStore` + HyperEdge + ACL isolation."""

    def __init__(
        self,
        uri: str = "bolt://127.0.0.1:27687",
        *,
        user: str = "neo4j",
        password: str = "graphrag_dev_pass",
        database: str = "neo4j",
        default_tenant_id: str = "default",
        default_acl_roles: list[str] | str = "public,internal",
        auto_init_constraints: bool = True,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.default_tenant_id = default_tenant_id
        self.default_acl_roles = _parse_roles(default_acl_roles, ["public", "internal"])
        graph_db = _import_neo4j()
        self._driver = graph_db.driver(uri, auth=(user, password))
        if auto_init_constraints:
            self.ensure_constraints()

    def close(self) -> None:
        self._driver.close()

    def _session(self) -> Any:
        return self._driver.session(database=self.database)

    def health_check(self) -> tuple[bool, str]:
        try:
            with self._session() as sess:
                rec = sess.run(
                    "CALL dbms.components() YIELD name, versions, edition "
                    "RETURN name, versions[0] AS ver, edition LIMIT 1"
                ).single()
                if rec:
                    return (
                        True,
                        f"Neo4j ready ({rec['name']} {rec['ver']} {rec['edition']} @ {self.uri})",
                    )
                return True, f"Neo4j ready @ {self.uri}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Neo4j health check failed ({self.uri}): {exc}"

    def ensure_constraints(self) -> None:
        """Idempotently create Neo4j 5.x uniqueness constraints and lookup indexes."""
        statements = [
            "CREATE CONSTRAINT entity_tenant_id_uniq IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.tenant_id, e.entity_id) IS UNIQUE",
            "CREATE CONSTRAINT hyperedge_tenant_id_uniq IF NOT EXISTS "
            "FOR (h:HyperEdge) REQUIRE (h.tenant_id, h.hyperedge_id) IS UNIQUE",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.entity_type)",
            "CREATE INDEX community_tenant_idx IF NOT EXISTS "
            "FOR (c:Community) ON (c.tenant_id, c.community_id, c.acl_partition)",
        ]
        with self._session() as sess:
            for cypher in statements:
                sess.run(cypher)

    @staticmethod
    def _node_to_entity(props: dict[str, Any]) -> Entity:
        meta_raw = props.get("metadata_json") or "{}"
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
        except json.JSONDecodeError:
            meta = {}
        if props.get("acl"):
            meta.setdefault("acl", list(props["acl"]))
        cid = props.get("community_id")
        return Entity(
            entity_id=str(props["entity_id"]),
            name=str(props.get("name", props["entity_id"])),
            entity_type=str(props.get("entity_type", "Concept")),
            description=str(props.get("description", "")),
            aliases=list(props.get("aliases") or []),
            source_chunk_ids=list(props.get("source_chunk_ids") or []),
            community_id=int(cid) if cid is not None and int(cid) >= 0 else None,
            metadata=meta,
        )

    @staticmethod
    def _rel_to_relation(props: dict[str, Any], source_id: str, target_id: str) -> Relation:
        meta_raw = props.get("metadata_json") or "{}"
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
        except json.JSONDecodeError:
            meta = {}
        if props.get("acl"):
            meta.setdefault("acl", list(props["acl"]))
        return Relation(
            relation_id=str(props["relation_id"]),
            source_id=str(source_id),
            target_id=str(target_id),
            relation_type=str(props.get("relation_type", "RELATED_TO")),
            description=str(props.get("description", "")),
            weight=float(props.get("weight", 1.0)),
            source_chunk_ids=list(props.get("source_chunk_ids") or []),
            metadata=meta,
        )

    def upsert_entity(
        self,
        entity: Entity,
        *,
        tenant_id: str | None = None,
        acl: list[str] | None = None,
    ) -> Entity:
        tid = tenant_id or self.default_tenant_id
        eff_acl = _parse_roles(
            acl if acl is not None else entity.metadata.get("acl"),
            self.default_acl_roles,
        )
        meta_json = json.dumps(entity.metadata, ensure_ascii=False)
        cypher = """
        MERGE (e:Entity {tenant_id: $tid, entity_id: $eid})
        ON CREATE SET
            e.name = $name,
            e.entity_type = $etype,
            e.description = $desc,
            e.aliases = $aliases,
            e.source_chunk_ids = $chunks,
            e.community_id = $comm_id,
            e.acl = $acl,
            e.is_deleted = false,
            e.metadata_json = $meta_json
        ON MATCH SET
            e.entity_type = CASE WHEN e.entity_type = 'Concept' THEN $etype ELSE e.entity_type END,
            e.description = CASE WHEN size(coalesce(e.description, '')) = 0 THEN $desc ELSE e.description END,
            e.aliases = apoc_or_list,
            e.source_chunk_ids = [x IN coalesce(e.source_chunk_ids, []) WHERE NOT x IN $chunks] + $chunks,
            e.community_id = CASE WHEN $comm_id >= 0 THEN $comm_id ELSE coalesce(e.community_id, -1) END,
            e.acl = [x IN coalesce(e.acl, []) WHERE NOT x IN $acl] + $acl,
            e.is_deleted = false,
            e.metadata_json = $meta_json
        RETURN properties(e) AS props
        """.replace(
            "apoc_or_list",
            "[x IN coalesce(e.aliases, []) WHERE NOT x IN $aliases] + $aliases",
        )
        with self._session() as sess:
            rec = sess.run(
                cypher,
                tid=tid,
                eid=entity.entity_id,
                name=entity.name,
                etype=entity.entity_type,
                desc=entity.description[:600],
                aliases=entity.aliases,
                chunks=entity.source_chunk_ids,
                comm_id=entity.community_id if entity.community_id is not None else -1,
                acl=eff_acl,
                meta_json=meta_json,
            ).single()
            return self._node_to_entity(rec["props"]) if rec else entity

    def upsert_relation(
        self,
        relation: Relation,
        *,
        tenant_id: str | None = None,
        acl: list[str] | None = None,
    ) -> Relation:
        tid = tenant_id or self.default_tenant_id
        eff_acl = _parse_roles(
            acl if acl is not None else relation.metadata.get("acl"),
            self.default_acl_roles,
        )
        meta_json = json.dumps(relation.metadata, ensure_ascii=False)
        cypher = """
        MATCH (s:Entity {tenant_id: $tid, entity_id: $sid})
        MATCH (t:Entity {tenant_id: $tid, entity_id: $tid_ent})
        MERGE (s)-[r:RELATES_TO {tenant_id: $tid, relation_id: $rid}]->(t)
        ON CREATE SET
            r.relation_type = $rtype,
            r.description = $desc,
            r.weight = $weight,
            r.source_chunk_ids = $chunks,
            r.acl = $acl,
            r.is_deleted = false,
            r.metadata_json = $meta_json
        ON MATCH SET
            r.weight = coalesce(r.weight, 0.0) + $weight,
            r.source_chunk_ids = [x IN coalesce(r.source_chunk_ids, []) WHERE NOT x IN $chunks] + $chunks,
            r.acl = [x IN coalesce(r.acl, []) WHERE NOT x IN $acl] + $acl,
            r.is_deleted = false,
            r.metadata_json = $meta_json
        RETURN properties(r) AS props
        """
        with self._session() as sess:
            rec = sess.run(
                cypher,
                tid=tid,
                sid=relation.source_id,
                tid_ent=relation.target_id,
                rid=relation.relation_id,
                rtype=relation.relation_type,
                desc=relation.description[:600],
                weight=float(relation.weight),
                chunks=relation.source_chunk_ids,
                acl=eff_acl,
                meta_json=meta_json,
            ).single()
            return (
                self._rel_to_relation(rec["props"], relation.source_id, relation.target_id)
                if rec
                else relation
            )

    def upsert_hyperedge(
        self,
        hyperedge: HyperEdge,
        *,
        tenant_id: str | None = None,
        acl: list[str] | None = None,
    ) -> HyperEdge:
        """Persist an N-ary HyperEdge as a `:HyperEdge` mediator node connected to condition/target entities."""
        tid = tenant_id or self.default_tenant_id
        eff_acl = _parse_roles(
            acl if acl is not None else hyperedge.metadata.get("acl"),
            self.default_acl_roles,
        )
        meta_json = json.dumps(hyperedge.metadata, ensure_ascii=False)
        cypher = """
        MERGE (h:HyperEdge {tenant_id: $tid, hyperedge_id: $hid})
        SET h.relation_type = $rtype,
            h.description = $desc,
            h.confidence = $conf,
            h.weight = $weight,
            h.source_chunk_ids = $chunks,
            h.condition_entity_ids = $conds,
            h.target_entity_id = $target_id,
            h.acl = $acl,
            h.is_deleted = false,
            h.metadata_json = $meta_json
        WITH h
        OPTIONAL MATCH (t:Entity {tenant_id: $tid, entity_id: $target_id})
        FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
            MERGE (h)-[:TARGET {tenant_id: $tid}]->(t)
        )
        WITH h
        UNWIND $conds AS cid
        MATCH (c:Entity {tenant_id: $tid, entity_id: cid})
        MERGE (c)-[:CONDITION {tenant_id: $tid}]->(h)
        """
        with self._session() as sess:
            sess.run(
                cypher,
                tid=tid,
                hid=hyperedge.hyperedge_id,
                rtype=hyperedge.relation_type,
                desc=hyperedge.description[:600],
                conf=float(hyperedge.confidence),
                weight=float(hyperedge.weight),
                chunks=hyperedge.source_chunk_ids,
                conds=hyperedge.condition_entity_ids,
                target_id=hyperedge.target_entity_id,
                acl=eff_acl,
                meta_json=meta_json,
            )
        return hyperedge

    def get_all_hyperedges(
        self,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[HyperEdge]:
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        cypher = """
        MATCH (h:HyperEdge {tenant_id: $tid})
        WHERE coalesce(h.is_deleted, false) = false
          AND any(r IN coalesce(h.acl, ['public']) WHERE r IN $roles)
        RETURN properties(h) AS props
        """
        with self._session() as sess:
            out: list[HyperEdge] = []
            for rec in sess.run(cypher, tid=tid, roles=roles):
                p = rec["props"]
                try:
                    meta = json.loads(p.get("metadata_json") or "{}")
                except json.JSONDecodeError:
                    meta = {}
                out.append(
                    HyperEdge(
                        hyperedge_id=str(p["hyperedge_id"]),
                        condition_entity_ids=list(p.get("condition_entity_ids") or []),
                        target_entity_id=str(p.get("target_entity_id", "")),
                        relation_type=str(p.get("relation_type", "JOINT_INDICATES")),
                        description=str(p.get("description", "")),
                        confidence=float(p.get("confidence", 1.0)),
                        weight=float(p.get("weight", 1.0)),
                        source_chunk_ids=list(p.get("source_chunk_ids") or []),
                        metadata=meta,
                    )
                )
            return out

    def get_entity(
        self,
        entity_id: str,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> Entity | None:
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        cypher = """
        MATCH (e:Entity {tenant_id: $tid, entity_id: $eid})
        WHERE coalesce(e.is_deleted, false) = false
          AND any(r IN coalesce(e.acl, ['public']) WHERE r IN $roles)
        RETURN properties(e) AS props
        """
        with self._session() as sess:
            rec = sess.run(cypher, tid=tid, eid=entity_id, roles=roles).single()
            return self._node_to_entity(rec["props"]) if rec else None

    def find_entities_by_query(
        self,
        query: str,
        top_k: int = 5,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[Entity]:
        q_norm = re.sub(r"\s+", " ", query.strip()).lower()
        if not q_norm:
            return []
        q_tokens = list(dict.fromkeys(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", q_norm)))
        entities = self.get_all_entities(tenant_id=tenant_id, acl_roles=acl_roles)
        scored: list[tuple[float, Entity]] = []
        for ent in entities:
            score = 0.0
            for cand in (ent.name, *ent.aliases):
                c_norm = re.sub(r"\s+", " ", cand.strip()).lower()
                if not c_norm:
                    continue
                if c_norm in q_norm:
                    score = max(score, 3.0 + min(len(c_norm), 12) * 0.1)
                elif q_norm in c_norm:
                    score = max(score, 2.5)
                else:
                    c_toks = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", c_norm))
                    hits = len(set(q_tokens) & c_toks)
                    if hits > 0:
                        score = max(score, 1.0 + hits * 0.5)
            if score == 0.0 and ent.description:
                d_low = ent.description.lower()
                dhits = sum(1 for tok in q_tokens if tok in d_low)
                if dhits > 0:
                    score = 0.4 * dhits
            if score > 0.0:
                scored.append((score, ent))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [ent for _, ent in scored[:top_k]]

    def get_all_entities(
        self,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[Entity]:
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        cypher = """
        MATCH (e:Entity {tenant_id: $tid})
        WHERE coalesce(e.is_deleted, false) = false
          AND any(r IN coalesce(e.acl, ['public']) WHERE r IN $roles)
        RETURN properties(e) AS props
        """
        with self._session() as sess:
            return [self._node_to_entity(r["props"]) for r in sess.run(cypher, tid=tid, roles=roles)]

    def get_all_relations(
        self,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[Relation]:
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        cypher = """
        MATCH (s:Entity {tenant_id: $tid})-[r:RELATES_TO {tenant_id: $tid}]->(t:Entity {tenant_id: $tid})
        WHERE coalesce(s.is_deleted, false) = false
          AND coalesce(t.is_deleted, false) = false
          AND coalesce(r.is_deleted, false) = false
          AND any(role IN coalesce(s.acl, ['public']) WHERE role IN $roles)
          AND any(role IN coalesce(t.acl, ['public']) WHERE role IN $roles)
          AND any(role IN coalesce(r.acl, ['public']) WHERE role IN $roles)
        RETURN properties(r) AS props, s.entity_id AS sid, t.entity_id AS tid_ent
        """
        with self._session() as sess:
            return [
                self._rel_to_relation(rec["props"], rec["sid"], rec["tid_ent"])
                for rec in sess.run(cypher, tid=tid, roles=roles)
            ]

    def get_k_hop_subgraph(
        self,
        seed_entity_ids: list[str],
        k_hops: int = 1,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> tuple[list[Entity], list[Relation]]:
        """ACL-enforced Cypher k-hop traversal (VectorCypher pattern)."""
        if not seed_entity_ids:
            return [], []
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        hops = max(0, min(int(k_hops), 4))

        with self._session() as sess:
            if hops == 0:
                node_cypher = """
                MATCH (e:Entity {tenant_id: $tid})
                WHERE e.entity_id IN $seeds
                  AND coalesce(e.is_deleted, false) = false
                  AND any(role IN coalesce(e.acl, ['public']) WHERE role IN $roles)
                RETURN properties(e) AS props
                """
            else:
                node_cypher = f"""
                MATCH (s:Entity {{tenant_id: $tid}})
                WHERE s.entity_id IN $seeds
                  AND coalesce(s.is_deleted, false) = false
                  AND any(role IN coalesce(s.acl, ['public']) WHERE role IN $roles)
                OPTIONAL MATCH p = (s)-[:RELATES_TO*1..{hops}]-(n:Entity {{tenant_id: $tid}})
                WHERE all(x IN nodes(p) WHERE coalesce(x.is_deleted, false) = false
                          AND any(role IN coalesce(x.acl, ['public']) WHERE role IN $roles))
                  AND all(rel IN relationships(p) WHERE coalesce(rel.is_deleted, false) = false
                          AND any(role IN coalesce(rel.acl, ['public']) WHERE role IN $roles))
                WITH collect(DISTINCT s) + collect(DISTINCT n) AS all_nodes
                UNWIND all_nodes AS node
                WITH DISTINCT node WHERE node IS NOT NULL
                RETURN properties(node) AS props
                """
            entities = [
                self._node_to_entity(r["props"])
                for r in sess.run(node_cypher, tid=tid, seeds=seed_entity_ids, roles=roles)
            ]
            eids = [e.entity_id for e in entities]
            if not eids:
                return [], []

            rel_cypher = """
            MATCH (s:Entity {tenant_id: $tid})-[r:RELATES_TO {tenant_id: $tid}]->(t:Entity {tenant_id: $tid})
            WHERE s.entity_id IN $eids AND t.entity_id IN $eids
              AND coalesce(r.is_deleted, false) = false
              AND any(role IN coalesce(r.acl, ['public']) WHERE role IN $roles)
            RETURN properties(r) AS props, s.entity_id AS sid, t.entity_id AS tid_ent
            """
            relations = [
                self._rel_to_relation(rec["props"], rec["sid"], rec["tid_ent"])
                for rec in sess.run(rel_cypher, tid=tid, eids=eids, roles=roles)
            ]
            return entities, relations

    def set_communities(
        self,
        communities: list[Community],
        *,
        tenant_id: str | None = None,
        acl_partition: str = "public",
    ) -> None:
        """Persist ACL-partitioned community summaries so summaries never leak across privileges."""
        tid = tenant_id or self.default_tenant_id
        with self._session() as sess:
            sess.run(
                "MATCH (c:Community {tenant_id: $tid, acl_partition: $part}) DETACH DELETE c",
                tid=tid,
                part=acl_partition,
            )
            for comm in communities:
                sess.run(
                    """
                    CREATE (c:Community {
                        tenant_id: $tid,
                        community_id: $cid,
                        acl_partition: $part,
                        title: $title,
                        summary: $summary,
                        entity_ids: $eids,
                        relation_count: $rcount
                    })
                    """,
                    tid=tid,
                    cid=comm.community_id,
                    part=acl_partition,
                    title=comm.title,
                    summary=comm.summary,
                    eids=comm.entity_ids,
                    rcount=comm.relation_count,
                )
                if comm.entity_ids:
                    sess.run(
                        """
                        MATCH (e:Entity {tenant_id: $tid})
                        WHERE e.entity_id IN $eids
                        SET e.community_id = $cid
                        """,
                        tid=tid,
                        eids=comm.entity_ids,
                        cid=comm.community_id,
                    )

    def get_all_communities(
        self,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> list[Community]:
        tid = tenant_id or self.default_tenant_id
        roles = _parse_roles(acl_roles, self.default_acl_roles)
        cypher = """
        MATCH (c:Community {tenant_id: $tid})
        WHERE coalesce(c.acl_partition, 'public') IN $roles
        RETURN properties(c) AS props
        ORDER BY c.community_id ASC
        """
        with self._session() as sess:
            return [
                Community(
                    community_id=int(r["props"]["community_id"]),
                    title=str(r["props"].get("title", "")),
                    summary=str(r["props"].get("summary", "")),
                    entity_ids=list(r["props"].get("entity_ids") or []),
                    relation_count=int(r["props"].get("relation_count", 0)),
                )
                for r in sess.run(cypher, tid=tid, roles=roles)
            ]

    def to_networkx(
        self,
        *,
        tenant_id: str | None = None,
        acl_roles: list[str] | None = None,
    ) -> nx.DiGraph:
        entities = self.get_all_entities(tenant_id=tenant_id, acl_roles=acl_roles)
        relations = self.get_all_relations(tenant_id=tenant_id, acl_roles=acl_roles)
        graph = nx.DiGraph()
        ent_ids = {e.entity_id for e in entities}
        for e in entities:
            graph.add_node(
                e.entity_id,
                name=e.name,
                entity_type=e.entity_type,
                description=e.description,
                aliases=e.aliases,
                source_chunk_ids=e.source_chunk_ids,
                community_id=e.community_id if e.community_id is not None else -1,
            )
        for rel in relations:
            if rel.source_id in ent_ids and rel.target_id in ent_ids:
                graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_id=rel.relation_id,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight,
                    source_chunk_ids=rel.source_chunk_ids,
                )
        return graph

    def clear_tenant(self, tenant_id: str | None = None) -> None:
        tid = tenant_id or self.default_tenant_id
        with self._session() as sess:
            sess.run("MATCH (n {tenant_id: $tid}) DETACH DELETE n", tid=tid)

    def save(self, directory: Path) -> None:
        """No-op: Neo4j persists graph transactions directly."""
        _ = directory

    def load(self, directory: Path) -> None:
        """No-op: Neo4j queries live state directly."""
        _ = directory

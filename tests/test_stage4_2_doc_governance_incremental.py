"""Stage 4.2 offline unit tests: 3 Quality Gates, Parent-Child chunking, SHA-256 reuse, and Redis Worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_graphrag_kg.pipelines.doc_governance import (
    govern_document,
    run_file_gate,
    run_parse_gate,
    split_code_block,
    split_markdown_table,
)
from llm_graphrag_kg.pipelines.incremental_worker import (
    DocumentChangeEvent,
    IncrementalIngestionWorker,
)
from llm_graphrag_kg.providers.storage.pgvector_store import PgVectorStore


def test_file_gate_blocks_forged_pdf_and_binary_md(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "forged.pdf"
    fake_pdf.write_text("this is plain text pretending to be a PDF", encoding="utf-8")
    g_pdf = run_file_gate(fake_pdf)
    assert not g_pdf.passed
    assert any("Forged MIME" in e for e in g_pdf.errors)

    fake_md = tmp_path / "binary.md"
    fake_md.write_bytes(b"%PDF-1.7\n\x00\x01binary payload disguised as markdown")
    g_md = run_file_gate(fake_md)
    assert not g_md.passed
    assert g_md.errors

    valid_md = tmp_path / "valid.md"
    valid_md.write_text("# 标题\n这是一份正常的工业设备运维文档说明。", encoding="utf-8")
    assert run_file_gate(valid_md).passed


def test_parse_gate_blocks_garbled_text_and_tracks_partial_pages() -> None:
    from llm_graphrag_kg.pipelines.doc_governance import ParsedBlock

    garbled = [ParsedBlock(block_id="b1", content="正常" + "\ufffd" * 16)]
    g_bad = run_parse_gate(garbled)
    assert not g_bad.passed
    assert any("garbled" in e.lower() for e in g_bad.errors)

    partial = [
        ParsedBlock(
            block_id="b1",
            page_no=1,
            content="离心泵在2倍频振动升高时应优先复核轴系对中偏差与地脚螺栓紧固度。",
        )
    ]
    g_part = run_parse_gate(partial, total_pages=3, missing_pages=[2])
    assert g_part.passed
    assert g_part.warnings and "Partial extraction" in g_part.warnings[0]


def test_table_atomic_and_header_continuation_and_python_ast_split() -> None:
    # 1. Small table stays atomic
    small_tbl = "| 参数 | 阈值 |\n| --- | --- |\n| 温度 | 85°C |\n| 振动 | 4.5 mm/s |"
    assert len(split_markdown_table(small_tbl, atomic_max_tokens=600)) == 1

    # 2. Large table splits by rows and REPEATS header + separator on every split chunk
    rows = [
        f"| P-{i:03d} | 离心泵轴承位点{i}监测指标与告警处理规程说明 | {4.0 + i * 0.1:.1f} mm/s |"
        for i in range(45)
    ]
    big_tbl = "| 点位 | 描述 | 告警阈值 |\n| --- | --- | --- |\n" + "\n".join(rows)
    pieces = split_markdown_table(big_tbl, atomic_max_tokens=300, max_tokens=220)
    assert len(pieces) >= 3
    for p in pieces:
        lines = p.splitlines()
        assert lines[0] == "| 点位 | 描述 | 告警阈值 |"
        assert lines[1] == "| --- | --- | --- |"

    # 3. Python AST splitter respects function/class boundaries
    py_code = "\n\n".join(
        [
            f"def diagnose_rule_{i}(vib_2x: float, temp_c: float) -> str:\n"
            f'    """Diagnostic rule #{i} evaluating vibration and bearing temperature thresholds."""\n'
            f"    score = vib_2x * 1.5 + temp_c * 0.2 + {i}\n"
            f"    return 'misalignment' if score > 30 else 'normal'"
            for i in range(8)
        ]
    )
    code_pieces = split_code_block(py_code, language="python", max_tokens=120)
    assert len(code_pieces) >= 2
    for cp in code_pieces:
        assert cp.lstrip().startswith("def diagnose_rule_")


def test_govern_document_parent_child_breadcrumbs_and_chunk_gate() -> None:
    md = (
        "# 离心泵运维手册\n\n"
        "## 1. 振动监测\n"
        "离心泵机组出现 2 倍频径向振动升高时，通常由联轴器平行或角度轴系不对中引发附加轴向载荷。\n\n"
        "ok\n\n"  # tiny fragment that ChunkGate must drop (<10 tok)
        "## 2. 巡检阈值表\n"
        "| 测点 | 报警值 | 停机值 |\n"
        "| --- | --- | --- |\n"
        "| 驱动端轴承温度 | 80°C | 95°C |\n"
        "| 2X径向振动速度 | 4.5 mm/s | 7.1 mm/s |\n"
    )
    res = govern_document(
        doc_id="doc_pump_01",
        raw_text=md,
        title="离心泵运维手册",
        source_uri="manual/pump.md",
        version_id=1,
        acl=["public", "engineer"],
    )
    assert res.status == "ready"
    assert res.dropped_chunks >= 1  # "ok" fragment dropped
    assert len(res.parent_chunks) >= 2
    assert len(res.child_chunks) >= 2
    parent_ids = {p.parent_id for p in res.parent_chunks}
    for ch in res.child_chunks:
        assert ch.parent_id in parent_ids
        assert ch.breadcrumb_text.startswith("[离心泵运维手册 > ")
        assert len(ch.content_hash) == 64


class _CountingEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.5] * self.dim for _ in texts]


class _FakeRedis:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.acked: list[str] = []
        self._seq = 0

    def xgroup_create(self, **kwargs: Any) -> None:
        _ = kwargs

    def xadd(self, name: str, fields: dict[str, str]) -> str:
        self._seq += 1
        sid = f"{self._seq}-0"
        self.streams.setdefault(name, []).append((sid, dict(fields)))
        return sid

    def xreadgroup(self, streams: dict[str, str], count: int = 10, **kwargs: Any) -> list[Any]:
        _ = kwargs
        out: list[Any] = []
        for s_name in streams:
            items = self.streams.get(s_name, [])[:count]
            if items:
                self.streams[s_name] = self.streams[s_name][len(items) :]
                out.append((s_name, items))
        return out

    def xack(self, stream: str, group: str, msg_id: str) -> int:
        _ = (stream, group)
        self.acked.append(msg_id)
        return 1

    def xrevrange(self, name: str, count: int = 20) -> list[tuple[str, dict[str, str]]]:
        return list(reversed(self.streams.get(name, [])))[:count]

    def scan_iter(self, match: str = "*", count: int = 200) -> list[str]:
        _ = (match, count)
        return []


class _StatefulFakePgCursor:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self._next_one: Any = None
        self._next_all: list[Any] = []
        self.rowcount = 1

    def __enter__(self) -> _StatefulFakePgCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        s = " ".join(sql.split())
        self.state.setdefault("sql_log", []).append(s)
        self._next_one = None
        self._next_all = []
        if s.startswith("SELECT active_version, status, is_deleted FROM documents"):
            tid, did = params
            self._next_one = self.state["docs"].get((tid, did))
        elif s.startswith("SELECT content_hash FROM child_chunks"):
            tid, did, ver = params
            self._next_all = [
                (c["content_hash"],)
                for c in self.state["chunks"].values()
                if c["tid"] == tid and c["doc_id"] == did and c["ver"] == ver and c["status"] == "ready"
            ]
        elif s.startswith("SELECT content_hash, embedding::text FROM child_chunks"):
            tid, did, _, _ = params
            self._next_all = [
                (c["content_hash"], c["vec"])
                for c in self.state["chunks"].values()
                if c["tid"] == tid and c["doc_id"] == did and c.get("vec")
            ]
        elif s.startswith("INSERT INTO documents"):
            did, tid = params[0], params[1]
            self.state["docs"][(tid, did)] = (0, "building", False)
        elif s.startswith("INSERT INTO child_chunks"):
            cid, _, tid, did, ver = params[0], params[1], params[2], params[3], params[4]
            c_hash = params[11]
            self.state["chunks"][cid] = {
                "tid": tid,
                "doc_id": did,
                "ver": ver,
                "content_hash": c_hash,
                "status": "pending",
                "vec": None,
            }
        elif s.startswith("UPDATE child_chunks SET index_status = 'processing'"):
            _, tid, did, ver = params
            for c in self.state["chunks"].values():
                if c["tid"] == tid and c["doc_id"] == did and c["ver"] == ver:
                    c["status"] = "processing"
        elif s.startswith("UPDATE child_chunks SET embedding ="):
            vec_lit, _, _, cid, _, _ = params
            if cid in self.state["chunks"]:
                self.state["chunks"][cid]["vec"] = vec_lit
                self.state["chunks"][cid]["status"] = "ready"
        elif s.startswith("SELECT COUNT(*) FROM child_chunks"):
            tid, did, ver = params
            unready = sum(
                1
                for c in self.state["chunks"].values()
                if c["tid"] == tid and c["doc_id"] == did and c["ver"] == ver and c["status"] != "ready"
            )
            self._next_one = (unready,)
        elif s.startswith("UPDATE documents SET acl ="):
            ver, status, _, tid, did = params[4], params[5], params[6], params[7], params[8]
            self.state["docs"][(tid, did)] = (ver, status, False)
        elif s.startswith("SELECT embedding_model, embedding_dim"):
            self._next_one = ("test-embed", 4, "child_chunks", "active")

    def fetchone(self) -> Any:
        return self._next_one

    def fetchall(self) -> list[Any]:
        return list(self._next_all)


class _StatefulFakePgConn:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def __enter__(self) -> _StatefulFakePgConn:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def cursor(self) -> _StatefulFakePgCursor:
        return _StatefulFakePgCursor(self.state)

    def commit(self) -> None:
        return None


def test_incremental_worker_idempotency_hash_reuse_stale_version_and_dlq(monkeypatch: Any) -> None:
    state: dict[str, Any] = {"docs": {}, "chunks": {}}

    class _FakePsycopg:
        @staticmethod
        def connect(dsn: str) -> _StatefulFakePgConn:
            _ = dsn
            return _StatefulFakePgConn(state)

    monkeypatch.setattr(
        "llm_graphrag_kg.providers.storage.pgvector_store._import_psycopg",
        lambda: _FakePsycopg,
    )

    pg = PgVectorStore(
        "postgresql://u:p@127.0.0.1:25432/db",
        embedding_dim=4,
        embedding_model="test-embed",
        auto_init_schema=False,
    )
    embedder = _CountingEmbedder(dim=4)
    fake_redis = _FakeRedis()
    worker = IncrementalIngestionWorker(
        pg_store=pg,
        embedder=embedder,
        redis_client=fake_redis,
        max_retries=2,
    )

    doc_v1 = (
        "# 手册\n\n"
        "## 第一节 振动\n离心泵在 2 倍频振动升高时需立即核查联轴器轴系对中偏差。\n\n"
        "## 第二节 润滑\n滚动轴承温度超过 85 度时需排查润滑脂劣化与油位不足问题。\n"
    )
    # 1. Initial v1 via Redis Stream -> computes 2 embeddings
    evt_v1 = DocumentChangeEvent(doc_id="doc_inc", version_id=1, raw_text=doc_v1, title="手册")
    worker.enqueue_event(evt_v1)
    r1 = worker.consume_batch(max_messages=5)[0]
    assert r1.outcome == "applied"
    assert r1.computed_embeddings == 2 and r1.reused_embeddings == 0

    # 2. Duplicate v1 delivery -> 100% idempotent no-op (0 extra embed calls!)
    calls_before = len(embedder.calls)
    r_dup = worker.process_event(evt_v1)
    assert r_dup.outcome == "idempotent_noop"
    assert len(embedder.calls) == calls_before

    # 3. Partial edit v2 (only Section 2 changed!) -> reuses Section 1 embedding, computes only 1!
    doc_v2 = (
        "# 手册\n\n"
        "## 第一节 振动\n离心泵在 2 倍频振动升高时需立即核查联轴器轴系对中偏差。\n\n"
        "## 第二节 润滑\n滚动轴承温度超过 90 度时必须立即停机更换全新耐高温锂基润滑脂。\n"
    )
    r2 = worker.process_event(
        DocumentChangeEvent(doc_id="doc_inc", version_id=2, raw_text=doc_v2, title="手册")
    )
    assert r2.outcome == "applied"
    assert r2.reused_embeddings == 1
    assert r2.computed_embeddings == 1

    # 4. Out-of-order stale v1 arriving after v2 -> blocked cleanly!
    r_stale = worker.process_event(evt_v1)
    assert r_stale.outcome == "stale_version_ignored"

    # 5. Poisoned event exceeding max_retries -> routed to DLQ!
    class _BrokenEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            _ = texts
            raise RuntimeError("simulated ollama crash")

    worker_fail = IncrementalIngestionWorker(
        pg_store=pg,
        embedder=_BrokenEmbedder(),  # type: ignore[arg-type]
        redis_client=fake_redis,
        max_retries=2,
    )
    bad_evt = DocumentChangeEvent(doc_id="doc_fail", version_id=1, raw_text=doc_v1, title="坏文档")
    worker_fail.enqueue_event(bad_evt)
    # First batch retries (attempt 0 -> 1); second batch routes to DLQ!
    worker_fail.consume_batch(max_messages=5)
    res_dlq = worker_fail.consume_batch(max_messages=5)
    assert res_dlq and res_dlq[0].outcome == "failed_dlq"
    dlq_items = worker_fail.list_dlq()
    assert len(dlq_items) == 1
    assert "simulated ollama crash" in dlq_items[0]["error"]

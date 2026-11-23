"""Unit tests for Stage 3.0 base model defaults, OllamaNative num_gpu control, and privacy guard."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from typing import Any

import pytest

from llm_graphrag_kg.config import AppConfig, EnvSettings
from llm_graphrag_kg.providers.factory import create_llm_client
from llm_graphrag_kg.providers.llm.ollama_native import OllamaNativeLLMClient


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> _DummyResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_env_settings_stage3_base_model_defaults() -> None:
    env = EnvSettings(_env_file=None)
    assert env.llm_model == "qwen3.5:9b"
    assert env.llm_num_gpu is None
    assert env.llm_timeout == 180


def test_factory_routes_to_ollama_native_when_num_gpu_set() -> None:
    cfg = AppConfig(
        env=EnvSettings(
            _env_file=None,
            llm_provider="openai_compat",
            llm_base_url="http://127.0.0.1:11434/v1",
            llm_model="qwen3.5:9b",
            llm_num_ctx=16384,
            llm_num_gpu=99,
            llm_think=False,
        )
    )
    client = create_llm_client(cfg)
    assert isinstance(client, OllamaNativeLLMClient)
    assert client.base_url == "http://127.0.0.1:11434"
    assert client.model == "qwen3.5:9b"
    assert client.num_ctx == 16384
    assert client.num_gpu == 99
    assert client.think is False
    assert cfg.redacted_summary()["llm_num_gpu"] == 99


def test_ollama_native_sends_num_gpu_and_records_prefill_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_urlopen(req: Any, timeout: int = 180) -> _DummyResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _DummyResponse(
            {
                "message": {"content": "ok"},
                "load_duration": 500_000_000,
                "prompt_eval_count": 2000,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 50,
                "eval_duration": 1_000_000_000,
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    client = OllamaNativeLLMClient(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:9b",
        think=False,
        num_ctx=16384,
        num_gpu=99,
    )
    out = client.complete("hello", system_prompt="sys", temperature=0.0, max_tokens=64)
    assert out == "ok"
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["num_ctx"] == 16384
    assert captured["body"]["options"]["num_gpu"] == 99
    assert captured["body"]["options"]["num_predict"] == 64
    assert client.last_telemetry["load_seconds"] == 0.5
    assert client.last_telemetry["prefill_tok_per_s"] == 2000.0
    assert client.last_telemetry["eval_tok_per_s"] == 50.0


def test_privacy_check_blocks_local_bench_dir(tmp_path: Path) -> None:
    script_path = Path("scripts/privacy_check.py").resolve()
    spec = importlib.util.spec_from_file_location("privacy_check_local_bench_mod", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    issues = mod.scan_files(tmp_path, ["local_bench/ollama_perf/bench_ollama.py"])
    assert any("[FORBIDDEN DIR]" in item for item in issues)

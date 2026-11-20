"""Unit tests for the privacy and changelog hygiene checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_privacy_module():
    script_path = Path("scripts/privacy_check.py").resolve()
    spec = importlib.util.spec_from_file_location("privacy_check_test_mod", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_privacy_check_detects_leaks_and_missing_changelog(tmp_path: Path) -> None:
    mod = _load_privacy_module()

    leaked = tmp_path / "bad.py"
    fake_secret = "sk" + "-1234567890abcdef1234567890"
    fake_win_path = "C:" + "\\Users\\Alice\\secret.txt"
    fake_email = "alice" + "@qq.com"
    leaked.write_text(
        f'KEY = "{fake_secret}"\nPATH = r"{fake_win_path}"\nEMAIL = "{fake_email}"\n',
        encoding="utf-8",
    )

    issues = mod.scan_files(tmp_path, ["bad.py", ".env"], require_changelog=True)
    joined = "\n".join(issues)
    assert "[CHANGELOG]" in joined
    assert "[FORBIDDEN FILE]" in joined
    assert "[SECRET]" in joined
    assert "[LOCAL PATH]" in joined
    assert "[EMAIL]" in joined


def test_privacy_check_passes_clean_files_with_noreply_email(tmp_path: Path) -> None:
    mod = _load_privacy_module()
    (tmp_path / "CHANGELOG.md").write_text("## [2026-11-20] clean\n", encoding="utf-8")
    (tmp_path / "clean.py").write_text(
        'AUTHOR = "294740142+WindFromKadath@users.noreply.github.com"\n',
        encoding="utf-8",
    )
    issues = mod.scan_files(tmp_path, ["CHANGELOG.md", "clean.py"], require_changelog=True)
    assert issues == []

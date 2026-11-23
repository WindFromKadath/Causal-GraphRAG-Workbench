"""Privacy and commit hygiene checker for LLM-GraphRAG-KG.

Checks staged or repository files for:
1. Forbidden files/directories (.env, .vscode/, data/private/, storage/, outputs/)
2. Potential API keys / tokens
3. Hardcoded local machine user paths
4. Personal email addresses (allowing GitHub noreply and example.com)
5. Optional requirement that CHANGELOG.md is updated on every commit
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_EXACT = {".env"}
FORBIDDEN_PREFIXES = (
    ".vscode/",
    "data/private/",
    "data/raw/",
    "local_bench/",
    "storage/",
    "outputs/",
)

# Built via concatenation so this script never triggers its own secret rules.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI/DeepSeek-like API Key", re.compile(r"\bsk" + r"-[A-Za-z0-9_-]{16,}")),
    ("GitHub OAuth/PAT Token", re.compile(r"\b(?:gho|ghp|ghu|ghs|github_pat)" + r"_[A-Za-z0-9_]{12,}")),
    ("Google API Key", re.compile(r"\bAIza" + r"[0-9A-Za-z_-]{20,}")),
    ("AWS Access Key", re.compile(r"\bAKIA" + r"[0-9A-Z]{16}\b")),
]

PATH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "Windows local user path",
        re.compile(r"[A-Za-z]:[\\/]" + r"Users[\\/][^\\/\s\"'`]+"),
    ),
    (
        "Unix home directory path",
        re.compile(r"(?:^|[\s\"'`])/(?:Users|home)/" + r"[A-Za-z0-9._-]+/"),
    ),
]

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ALLOWED_EMAIL_DOMAINS = (
    "users.noreply.github.com",
    "example.com",
    "example.org",
    "localhost",
)

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".ini",
    ".cfg",
    ".env",
    ".example",
    ".sh",
    ".ps1",
    ".csv",
    ".html",
    ".xml",
    ".graphml",
}


def run_git(args: list[str], cwd: Path) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def collect_files(repo_root: Path, staged_only: bool) -> list[str]:
    if staged_only:
        return run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"], repo_root)
    tracked = run_git(["ls-files"], repo_root)
    if tracked:
        return tracked
    # Fallback when repo has no commits yet: inspect non-ignored files
    return run_git(["ls-files", "--others", "--exclude-standard"], repo_root)


def is_text_candidate(rel_path: str) -> bool:
    p = Path(rel_path)
    if p.name in {".gitignore", ".env.example", ".python-version", "Makefile", "Dockerfile"}:
        return True
    return p.suffix.lower() in TEXT_EXTENSIONS


def scan_files(repo_root: Path, rel_paths: list[str], require_changelog: bool = False) -> list[str]:
    issues: list[str] = []

    if require_changelog and "CHANGELOG.md" not in rel_paths:
        issues.append(
            "[CHANGELOG] Missing `CHANGELOG.md` in staged files. "
            "Every Git commit must include a modification log entry in CHANGELOG.md."
        )

    for rel_path in rel_paths:
        norm = rel_path.replace("\\", "/")
        if norm in FORBIDDEN_EXACT or (
            norm.startswith(".env.") and norm != ".env.example"
        ):
            issues.append(f"[FORBIDDEN FILE] {norm} must never be committed.")
            continue
        if any(norm.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            issues.append(f"[FORBIDDEN DIR] {norm} belongs to a private/local-only directory.")
            continue

        full_path = repo_root / norm
        if not full_path.is_file() or not is_text_candidate(norm):
            continue

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for lineno, line in enumerate(content.splitlines(), start=1):
            if "privacy-ignore" in line:
                continue

            for label, pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    issues.append(f"[SECRET] {norm}:{lineno} matched {label}.")

            for label, pattern in PATH_PATTERNS:
                if pattern.search(line):
                    issues.append(f"[LOCAL PATH] {norm}:{lineno} matched {label}.")

            for match in EMAIL_PATTERN.finditer(line):
                email = match.group(0).lower()
                domain = email.split("@", 1)[1]
                if not any(domain == d or domain.endswith("." + d) for d in ALLOWED_EMAIL_DOMAINS):
                    issues.append(f"[EMAIL] {norm}:{lineno} contains personal email `{email}`.")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check staged or repo files for privacy leaks.")
    parser.add_argument("--staged", action="store_true", help="Scan only git staged files.")
    parser.add_argument(
        "--require-changelog",
        action="store_true",
        help="Fail if CHANGELOG.md is not included in the scanned file list.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory.",
    )
    args = parser.parse_args(argv)

    rel_paths = collect_files(args.repo_root, staged_only=args.staged)
    issues = scan_files(args.repo_root, rel_paths, require_changelog=args.require_changelog)

    if issues:
        print("Privacy / Hygiene Check FAILED:", file=sys.stderr)
        for item in issues:
            print(f"  - {item}", file=sys.stderr)
        return 1

    scope = "staged" if args.staged else "repository"
    print(f"Privacy / Hygiene Check PASSED ({len(rel_paths)} {scope} files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

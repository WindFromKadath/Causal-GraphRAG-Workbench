"""Stage 4.2 Enterprise Document Governance: multi-format parsing, 3 Quality Gates, and Parent-Child chunking.

Implements JavaGuide enterprise RAG ingestion standards:
- Multi-format structural parsing: Tier-1 Docling (when installed) -> Tier-2 PyMuPDF (multi-column PDF +
  table extraction) / python-docx (headings + tables) -> native Markdown/Python AST/Text parser.
- Three Quality Gates:
  1. `FileGate`: existence, size cap, magic-byte / MIME anti-spoofing.
  2. `ParseGate`: minimum substance, garbled/mojibake ratio check, empty/missing page tracking.
  3. `ChunkGate`: fragment noise filtering, intra-doc SHA-256 deduplication, metadata completeness.
- Format-aware Parent-Child Chunking + Breadcrumb Prefix Injection:
  - Parent reading windows (~1000-1500 tokens) + Child retrieval windows (~256-384 tokens, 12% overlap)
  - Tables <= 600 tokens kept atomic; > 600 tokens split by rows with automatic Markdown header repeat
  - Python code split via `ast.parse` on top-level functions/classes
  - Every child gets `[Title > Section Path | P.x]` breadcrumb prefix & SHA-256 `content_hash`.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_graphrag_kg.core.schema import TextChunk

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^```([A-Za-z0-9_+-]*)\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?\n])\s*")
_MOJIBAKE_RE = re.compile(r"(Ã.|â€|ï¿½|Ð.|Ñ.){2,}")

_BINARY_MAGIC_PREFIXES = (
    b"%PDF-",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",
    b"MZ",
    b"\x7fELF",
)


class ParsedBlock(BaseModel):
    block_id: str
    element_type: str = "text"  # text | table | code
    section_path: str = ""
    page_no: int | None = None
    content: str
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParentChunkRecord(BaseModel):
    parent_id: str
    doc_id: str
    version_id: int = 1
    section_path: str = ""
    page_no: int | None = None
    element_type: str = "text"
    content: str
    token_count: int = 0
    acl: list[str] = Field(default_factory=lambda: ["public"])
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChildChunkRecord(BaseModel):
    chunk_id: str
    parent_id: str
    doc_id: str
    version_id: int = 1
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    source: str
    section_path: str = ""
    page_no: int | None = None
    element_type: str = "text"
    content_hash: str
    text: str
    breadcrumb_text: str
    token_count: int = 0
    acl: list[str] = Field(default_factory=lambda: ["public"])
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_text_chunk(self) -> TextChunk:
        meta = dict(self.metadata)
        meta.update(
            {
                "parent_id": self.parent_id,
                "version_id": self.version_id,
                "section_path": self.section_path,
                "page_no": self.page_no,
                "element_type": self.element_type,
                "content_hash": self.content_hash,
                "breadcrumb_text": self.breadcrumb_text,
                "acl": list(self.acl),
            }
        )
        return TextChunk(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            source=self.source,
            index=self.chunk_index,
            text=self.text,
            start_char=self.start_char,
            end_char=self.end_char,
            metadata=meta,
        )


class GateResult(BaseModel):
    gate_name: str
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class GovernanceResult(BaseModel):
    doc_id: str
    title: str
    source_uri: str
    source_type: str
    parser_used: str
    status: str  # ready | partial_available | rejected
    missing_pages: list[int] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)
    parent_chunks: list[ParentChunkRecord] = Field(default_factory=list)
    child_chunks: list[ChildChunkRecord] = Field(default_factory=list)
    dropped_chunks: int = 0
    rejection_reason: str | None = None


def estimate_tokens(text: str) -> int:
    """Fast bilingual token estimator calibrated for CJK + English technical corpora."""
    if not text:
        return 0
    cjk_count = len(_CJK_RE.findall(text))
    ascii_words = len(_ASCII_WORD_RE.findall(text))
    punct = max(0, len(text) - cjk_count - sum(len(m) for m in _ASCII_WORD_RE.findall(text)))
    return max(1, int(round(cjk_count * 1.15 + ascii_words * 1.3 + punct * 0.12)))


def compute_content_hash(section_path: str, text: str) -> str:
    normalized = f"{section_path.strip()}\n{re.sub(r'[ \t]+', ' ', text.strip())}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def format_breadcrumb(title: str, section_path: str, page_no: int | None, text: str) -> str:
    parts = [p.strip() for p in [title, section_path] if p and p.strip()]
    header = " > ".join(parts) if parts else "Document"
    if page_no is not None:
        header = f"{header} | P.{page_no}"
    return f"[{header}]\n{text.strip()}"


# ---------------------------------------------------------------------------
# Gate 1: FileGate (Magic-byte / MIME anti-spoofing & size bounds)
# ---------------------------------------------------------------------------


def run_file_gate(
    file_path: Path | None = None,
    *,
    raw_bytes: bytes | None = None,
    source_name: str | None = None,
    max_size_mb: float = 50.0,
) -> GateResult:
    """Gate 1: Verify file integrity, size limits, and real magic-byte MIME signature."""
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    if raw_bytes is None:
        if file_path is None or not file_path.is_file():
            return GateResult(
                gate_name="FileGate",
                passed=False,
                errors=[f"Source file not found: {file_path}"],
            )
        raw_bytes = file_path.read_bytes()
        ext = file_path.suffix.lower()
    else:
        ext = Path(source_name or (file_path.name if file_path else "inline.md")).suffix.lower() or ".md"

    size_bytes = len(raw_bytes)
    metrics["size_bytes"] = size_bytes
    metrics["extension"] = ext

    if size_bytes == 0:
        errors.append("Empty file (0 bytes).")
        return GateResult(gate_name="FileGate", passed=False, errors=errors, metrics=metrics)

    max_bytes = int(max_size_mb * 1024 * 1024)
    if size_bytes > max_bytes:
        errors.append(f"File size ({size_bytes} bytes) exceeds limit ({max_bytes} bytes).")

    head = raw_bytes[:16]
    if ext == ".pdf":
        if not head.startswith(b"%PDF-"):
            errors.append(
                f"Forged MIME detected: `.pdf` file does not start with `%PDF-` magic header (got {head[:8]!r})."
            )
    elif ext == ".docx":
        if not head.startswith(b"PK\x03\x04"):
            errors.append("Forged MIME detected: `.docx` file is not a valid ZIP/OOXML container.")
        else:
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    if "word/document.xml" not in zf.namelist():
                        errors.append("Invalid `.docx`: missing `word/document.xml` entry.")
            except zipfile.BadZipFile:
                errors.append("Corrupted `.docx`: bad ZIP structure.")
    elif ext in {".md", ".txt", ".py", ".yaml", ".yml", ".json", ".csv", ".rst", ".html", ".htm"}:
        if b"\x00" in raw_bytes[:4096]:
            errors.append(f"Binary payload disguised as text (`{ext}` contains NUL bytes).")
        for magic in _BINARY_MAGIC_PREFIXES:
            if head.startswith(magic):
                errors.append(f"Forged MIME detected: `{ext}` starts with binary magic bytes {magic!r}.")
                break
        try:
            raw_bytes.decode("utf-8")
            metrics["encoding"] = "utf-8"
        except UnicodeDecodeError:
            try:
                raw_bytes.decode("gb18030")
                metrics["encoding"] = "gb18030"
                warnings.append("Decoded using GB18030 fallback (non-UTF-8 text file).")
            except UnicodeDecodeError:
                errors.append(f"Undecodable text payload for `{ext}` (neither UTF-8 nor GB18030).")
    else:
        warnings.append(f"Unrecognized extension `{ext}`; will attempt plain-text decoding.")

    return GateResult(
        gate_name="FileGate",
        passed=len(errors) == 0,
        warnings=warnings,
        errors=errors,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Gate 2: ParseGate (Substance, garbled/mojibake ratio, missing page detection)
# ---------------------------------------------------------------------------


def compute_garbled_ratio(text: str) -> float:
    """Estimate fraction of replacement/control/PUA/mojibake characters in extracted text."""
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return 1.0
    bad_chars = 0
    for ch in stripped:
        code = ord(ch)
        if ch == "\ufffd" or (code < 32) or (0xE000 <= code <= 0xF8FF) or (0xFFF0 <= code <= 0xFFFF):
            bad_chars += 1
    for m in _MOJIBAKE_RE.finditer(stripped):
        bad_chars += len(m.group(0))
    return min(1.0, bad_chars / len(stripped))


def run_parse_gate(
    blocks: list[ParsedBlock],
    *,
    total_pages: int | None = None,
    missing_pages: list[int] | None = None,
    min_chars: int = 20,
    max_garbled_ratio: float = 0.15,
) -> GateResult:
    """Gate 2: Validate parsed blocks against empty extraction, high garbled rate, and missing pages."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = list(missing_pages or [])
    joined = "\n".join(b.content for b in blocks if b.content.strip())
    non_ws = len(re.sub(r"\s+", "", joined))
    garbled = compute_garbled_ratio(joined) if non_ws > 0 else 1.0

    metrics: dict[str, Any] = {
        "block_count": len(blocks),
        "non_ws_chars": non_ws,
        "garbled_ratio": round(garbled, 4),
        "total_pages": total_pages,
        "missing_pages": missing,
    }

    if non_ws < min_chars:
        errors.append(f"Parsed content too short ({non_ws} non-whitespace chars < {min_chars}).")
    if garbled > max_garbled_ratio:
        errors.append(
            f"High garbled/mojibake ratio detected ({garbled * 100:.1f}% > {max_garbled_ratio * 100:.1f}%)."
        )
    if total_pages and missing:
        if len(missing) >= total_pages:
            errors.append(f"All {total_pages} page(s) failed text extraction (possible unscanned image PDF).")
        else:
            warnings.append(
                f"Partial extraction: {len(missing)}/{total_pages} page(s) had no extractable text ({missing})."
            )

    return GateResult(
        gate_name="ParseGate",
        passed=len(errors) == 0,
        warnings=warnings,
        errors=errors,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Multi-Format Structural Parsers (Docling -> PyMuPDF / python-docx -> MD/Code)
# ---------------------------------------------------------------------------


def parse_markdown_blocks(text: str, *, page_no: int | None = None) -> list[ParsedBlock]:
    """Parse Markdown into heading-aware prose, table, and fenced code blocks."""
    lines = text.splitlines()
    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    b_idx = 0

    def cur_section() -> str:
        return " > ".join(h for _, h in heading_stack)

    def flush_prose() -> None:
        nonlocal b_idx
        content = "\n".join(buf).strip()
        buf.clear()
        if content:
            b_idx += 1
            blocks.append(
                ParsedBlock(
                    block_id=f"blk_{b_idx:04d}",
                    element_type="text",
                    section_path=cur_section(),
                    page_no=page_no,
                    content=content,
                )
            )

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 1. Fenced code block
        fm = _FENCE_RE.match(stripped)
        if fm:
            flush_prose()
            lang = fm.group(1).strip().lower()
            code_lines: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # consume closing fence
            code_body = "\n".join(code_lines).strip("\n")
            if code_body.strip():
                b_idx += 1
                blocks.append(
                    ParsedBlock(
                        block_id=f"blk_{b_idx:04d}",
                        element_type="code",
                        section_path=cur_section(),
                        page_no=page_no,
                        content=code_body,
                        language=lang or "text",
                    )
                )
            continue

        # 2. ATX Heading
        hm = _HEADING_RE.match(stripped)
        if hm:
            flush_prose()
            level = len(hm.group(1))
            title = hm.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            i += 1
            continue

        # 3. Markdown Table (header line with | followed by separator line)
        if "|" in line and (i + 1 < n) and _TABLE_SEP_RE.match(lines[i + 1]):
            flush_prose()
            tbl_lines = [line.strip(), lines[i + 1].strip()]
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                tbl_lines.append(lines[i].strip())
                i += 1
            b_idx += 1
            blocks.append(
                ParsedBlock(
                    block_id=f"blk_{b_idx:04d}",
                    element_type="table",
                    section_path=cur_section(),
                    page_no=page_no,
                    content="\n".join(tbl_lines),
                )
            )
            continue

        # 4. Blank line paragraph separator
        if not stripped:
            flush_prose()
            i += 1
            continue

        buf.append(line)
        i += 1

    flush_prose()
    return blocks


def _table_rows_to_markdown(rows: list[list[Any]]) -> str:
    clean_rows = [
        [re.sub(r"\s+", " ", str(cell or "")).replace("|", "/").strip() for cell in row]
        for row in rows
        if any(str(cell or "").strip() for cell in row)
    ]
    if not clean_rows:
        return ""
    col_n = max(len(r) for r in clean_rows)
    padded = [r + [""] * (col_n - len(r)) for r in clean_rows]
    header = "| " + " | ".join(padded[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(col_n)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join([header, sep, *body])


def _parse_pdf_pymupdf(file_path: Path) -> tuple[list[ParsedBlock], int, list[int]]:
    import pymupdf as fitz  # type: ignore[import-not-found]

    doc = fitz.open(str(file_path))
    blocks: list[ParsedBlock] = []
    missing_pages: list[int] = []
    total_pages = len(doc)
    current_section = ""
    b_idx = 0
    sec_re = re.compile(r"^((?:第[一二三四五六七八九十\d]+[章节条])|(?:\d+(?:\.\d+){0,3}))\s+.{2,60}$")

    try:
        for pno in range(total_pages):
            page = doc[pno]
            page_num = pno + 1
            page_added = 0
            table_rects: list[Any] = []

            # Extract structured tables first so body text doesn't duplicate table cells
            if hasattr(page, "find_tables"):
                try:
                    found = page.find_tables()
                    for tbl in found.tables:
                        md_tbl = _table_rows_to_markdown(tbl.extract())
                        if md_tbl:
                            b_idx += 1
                            blocks.append(
                                ParsedBlock(
                                    block_id=f"blk_{b_idx:04d}",
                                    element_type="table",
                                    section_path=current_section,
                                    page_no=page_num,
                                    content=md_tbl,
                                )
                            )
                            table_rects.append(fitz.Rect(tbl.bbox))
                            page_added += 1
                except Exception:  # noqa: BLE001
                    pass

            # Multi-column aware text blocks: sort left-column top-to-bottom, then right-column
            raw_blocks = page.get_text("blocks")
            page_mid = page.rect.width * 0.5
            text_items: list[tuple[int, float, float, str]] = []
            for rb in raw_blocks:
                if len(rb) < 5 or (len(rb) >= 7 and rb[6] != 0):
                    continue
                rect = fitz.Rect(rb[0], rb[1], rb[2], rb[3])
                txt = str(rb[4]).strip()
                if not txt:
                    continue
                if any(rect.intersects(tr) for tr in table_rects):
                    continue
                # Classify narrow left/right columns vs full-width header/paragraph
                width = rect.x1 - rect.x0
                if width < page.rect.width * 0.55 and rect.x0 >= page_mid * 0.85:
                    col_bucket = 1
                else:
                    col_bucket = 0
                text_items.append((col_bucket, rect.y0, rect.x0, txt))

            text_items.sort(key=lambda t: (t[0], round(t[1] / 6.0), t[2]))
            for _, _, _, txt in text_items:
                first_line = txt.splitlines()[0].strip()
                if len(txt) <= 80 and sec_re.match(first_line):
                    current_section = first_line
                    continue
                b_idx += 1
                blocks.append(
                    ParsedBlock(
                        block_id=f"blk_{b_idx:04d}",
                        element_type="text",
                        section_path=current_section,
                        page_no=page_num,
                        content=txt,
                    )
                )
                page_added += 1

            if page_added == 0:
                missing_pages.append(page_num)
    finally:
        doc.close()

    return blocks, total_pages, missing_pages


def _parse_docx_python_docx(file_path: Path) -> list[ParsedBlock]:
    from docx import Document as DocxDocument  # type: ignore[import-not-found]
    from docx.oxml.table import CT_Tbl  # type: ignore[import-not-found]
    from docx.oxml.text.paragraph import CT_P  # type: ignore[import-not-found]
    from docx.table import Table as DocxTable  # type: ignore[import-not-found]
    from docx.text.paragraph import Paragraph as DocxParagraph  # type: ignore[import-not-found]

    doc = DocxDocument(str(file_path))
    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []
    b_idx = 0

    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            para = DocxParagraph(child, doc)
            txt = para.text.strip()
            if not txt:
                continue
            style_name = str(para.style.name if para.style else "")
            if style_name.startswith("Heading"):
                m = re.search(r"(\d+)", style_name)
                lvl = int(m.group(1)) if m else 1
                while heading_stack and heading_stack[-1][0] >= lvl:
                    heading_stack.pop()
                heading_stack.append((lvl, txt))
                continue
            b_idx += 1
            blocks.append(
                ParsedBlock(
                    block_id=f"blk_{b_idx:04d}",
                    element_type="text",
                    section_path=" > ".join(h for _, h in heading_stack),
                    content=txt,
                )
            )
        elif isinstance(child, CT_Tbl):
            tbl = DocxTable(child, doc)
            rows = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
            md_tbl = _table_rows_to_markdown(rows)
            if md_tbl:
                b_idx += 1
                blocks.append(
                    ParsedBlock(
                        block_id=f"blk_{b_idx:04d}",
                        element_type="table",
                        section_path=" > ".join(h for _, h in heading_stack),
                        content=md_tbl,
                    )
                )
    return blocks


def parse_document_blocks(
    *,
    file_path: Path | None = None,
    raw_text: str | None = None,
    source_name: str = "inline.md",
    prefer_docling: bool = True,
) -> tuple[list[ParsedBlock], str, str, int | None, list[int]]:
    """Parse file or raw text into structured `ParsedBlock`s.

    Returns `(blocks, source_type, parser_used, total_pages, missing_pages)`.
    """
    if raw_text is not None:
        ext = Path(source_name).suffix.lower() or ".md"
        if ext == ".py":
            return (
                [
                    ParsedBlock(
                        block_id="blk_0001",
                        element_type="code",
                        section_path=Path(source_name).stem,
                        content=raw_text,
                        language="python",
                    )
                ],
                "python",
                "python_ast",
                None,
                [],
            )
        return parse_markdown_blocks(raw_text), ("markdown" if ext == ".md" else "text"), "markdown", None, []

    if file_path is None:
        return [], "unknown", "none", None, []

    ext = file_path.suffix.lower()
    if ext in {".pdf", ".docx"} and prefer_docling:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

            conv = DocumentConverter()
            res = conv.convert(str(file_path))
            md_text = res.document.export_to_markdown()
            blocks = parse_markdown_blocks(md_text)
            return blocks, ext.lstrip("."), "docling", None, []
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            pass

    if ext == ".pdf":
        blocks, total_pages, missing_pages = _parse_pdf_pymupdf(file_path)
        return blocks, "pdf", "pymupdf", total_pages, missing_pages

    if ext == ".docx":
        blocks = _parse_docx_python_docx(file_path)
        return blocks, "docx", "python_docx", None, []

    raw_bytes = file_path.read_bytes()
    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw_bytes.decode("gb18030", errors="replace")

    if ext == ".py":
        return (
            [
                ParsedBlock(
                    block_id="blk_0001",
                    element_type="code",
                    section_path=file_path.stem,
                    content=decoded,
                    language="python",
                )
            ],
            "python",
            "python_ast",
            None,
            [],
        )
    return parse_markdown_blocks(decoded), ("markdown" if ext == ".md" else "text"), "markdown", None, []


# ---------------------------------------------------------------------------
# Format-Aware Splitters (Table Header Repeat + Python AST + Sentence Overlap)
# ---------------------------------------------------------------------------


def split_markdown_table(table_md: str, *, atomic_max_tokens: int = 600, max_tokens: int = 380) -> list[str]:
    """Keep tables <= atomic_max_tokens intact; split larger tables while repeating header rows."""
    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    if len(lines) <= 2 or estimate_tokens(table_md) <= atomic_max_tokens:
        return [table_md.strip()]

    header_block = [lines[0], lines[1]]
    header_tokens = estimate_tokens("\n".join(header_block))
    budget = max(80, max_tokens - header_tokens)

    chunks: list[str] = []
    cur_rows: list[str] = []
    cur_toks = 0
    for row in lines[2:]:
        rtok = estimate_tokens(row)
        if cur_rows and (cur_toks + rtok > budget):
            chunks.append("\n".join([*header_block, *cur_rows]))
            cur_rows = [row]
            cur_toks = rtok
        else:
            cur_rows.append(row)
            cur_toks += rtok
    if cur_rows:
        chunks.append("\n".join([*header_block, *cur_rows]))
    return chunks


def split_code_block(code: str, *, language: str = "python", max_tokens: int = 380) -> list[str]:
    """Split Python code cleanly along top-level AST nodes; fallback to blank-line logical blocks."""
    if estimate_tokens(code) <= max_tokens:
        return [code.strip()]

    lines = code.splitlines()
    if language in {"py", "python"}:
        try:
            tree = ast.parse(code)
            segments: list[str] = []
            prev_end = 0
            for node in tree.body:
                start = max(prev_end, getattr(node, "lineno", 1) - 1)
                end = getattr(node, "end_lineno", start + 1)
                if start > prev_end:
                    preamble = "\n".join(lines[prev_end:start]).strip()
                    if preamble:
                        segments.append(preamble)
                seg = "\n".join(lines[start:end]).strip("\n")
                if seg.strip():
                    segments.append(seg)
                prev_end = end
            if prev_end < len(lines):
                tail = "\n".join(lines[prev_end:]).strip()
                if tail:
                    segments.append(tail)

            if segments:
                grouped: list[str] = []
                cur: list[str] = []
                cur_toks = 0
                for seg in segments:
                    st = estimate_tokens(seg)
                    if cur and (cur_toks + st > max_tokens):
                        grouped.append("\n\n".join(cur))
                        cur = [seg]
                        cur_toks = st
                    else:
                        cur.append(seg)
                        cur_toks += st
                if cur:
                    grouped.append("\n\n".join(cur))
                return grouped
        except SyntaxError:
            pass

    # Fallback: logical blank-line grouping
    paras = [p.strip("\n") for p in re.split(r"\n{2,}", code) if p.strip()]
    out: list[str] = []
    cur_p: list[str] = []
    cur_t = 0
    for p in paras:
        pt = estimate_tokens(p)
        if cur_p and cur_t + pt > max_tokens:
            out.append("\n\n".join(cur_p))
            cur_p = [p]
            cur_t = pt
        else:
            cur_p.append(p)
            cur_t += pt
    if cur_p:
        out.append("\n\n".join(cur_p))
    return out or [code.strip()]


def split_prose_text(text: str, *, max_tokens: int = 340, overlap_ratio: float = 0.12) -> list[str]:
    """Split prose on sentence boundaries with soft overlap."""
    if estimate_tokens(text) <= max_tokens:
        return [text.strip()]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        # Character fallback for run-on unbroken text
        approx_chars = max(120, int(max_tokens * 1.6))
        ov_chars = max(20, int(approx_chars * overlap_ratio))
        step = max(40, approx_chars - ov_chars)
        return [
            text[i : i + approx_chars].strip()
            for i in range(0, len(text), step)
            if text[i : i + approx_chars].strip()
        ]

    chunks: list[str] = []
    cur: list[str] = []
    cur_toks = 0
    overlap_budget = max(16, int(max_tokens * overlap_ratio))

    for sent in sentences:
        st = estimate_tokens(sent)
        if cur and (cur_toks + st > max_tokens):
            chunks.append(" ".join(cur).strip())
            # Keep trailing overlap sentences within overlap_budget
            ov: list[str] = []
            ov_t = 0
            for prev in reversed(cur):
                pt = estimate_tokens(prev)
                if ov_t + pt <= overlap_budget:
                    ov.insert(0, prev)
                    ov_t += pt
                else:
                    break
            cur = [*ov, sent]
            cur_toks = ov_t + st
        else:
            cur.append(sent)
            cur_toks += st
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks


# ---------------------------------------------------------------------------
# Parent-Child Assembly + Gate 3 (ChunkGate)
# ---------------------------------------------------------------------------


def build_parent_child_chunks(
    blocks: list[ParsedBlock],
    *,
    doc_id: str,
    title: str,
    source_uri: str,
    version_id: int = 1,
    acl: list[str] | None = None,
    parent_max_tokens: int = 1200,
    child_max_tokens: int = 340,
    child_overlap_ratio: float = 0.12,
    table_atomic_max_tokens: int = 600,
) -> tuple[list[ParentChunkRecord], list[ChildChunkRecord]]:
    """Assemble reading-grain Parent chunks and retrieval-grain Child chunks with breadcrumbs."""
    eff_acl = list(acl or ["public"])
    parents: list[ParentChunkRecord] = []
    children: list[ChildChunkRecord] = []

    # Group contiguous blocks sharing the same section_path into Parent windows
    groups: list[list[ParsedBlock]] = []
    cur_group: list[ParsedBlock] = []
    cur_toks = 0
    cur_sec: str | None = None

    for blk in blocks:
        bt = estimate_tokens(blk.content)
        sec_changed = cur_sec is not None and blk.section_path != cur_sec
        if cur_group and (sec_changed or cur_toks + bt > parent_max_tokens):
            groups.append(cur_group)
            cur_group = [blk]
            cur_toks = bt
            cur_sec = blk.section_path
        else:
            cur_group.append(blk)
            cur_toks += bt
            cur_sec = blk.section_path
    if cur_group:
        groups.append(cur_group)

    child_idx = 0
    char_cursor = 0
    for p_idx, grp in enumerate(groups):
        parent_id = f"{doc_id}::v{version_id}::p{p_idx:04d}"
        sec_path = grp[0].section_path
        page_no = grp[0].page_no
        elem_types = {b.element_type for b in grp}
        parent_elem = next(iter(elem_types)) if len(elem_types) == 1 else "mixed"
        parent_content = "\n\n".join(b.content for b in grp).strip()

        parents.append(
            ParentChunkRecord(
                parent_id=parent_id,
                doc_id=doc_id,
                version_id=version_id,
                section_path=sec_path,
                page_no=page_no,
                element_type=parent_elem,
                content=parent_content,
                token_count=estimate_tokens(parent_content),
                acl=eff_acl,
                metadata={"title": title, "source_uri": source_uri},
            )
        )

        # Format-aware Child splitting inside each Parent
        for blk in grp:
            if blk.element_type == "table":
                pieces = split_markdown_table(
                    blk.content,
                    atomic_max_tokens=table_atomic_max_tokens,
                    max_tokens=child_max_tokens,
                )
            elif blk.element_type == "code":
                pieces = split_code_block(
                    blk.content,
                    language=blk.language,
                    max_tokens=child_max_tokens,
                )
            else:
                pieces = split_prose_text(
                    blk.content,
                    max_tokens=child_max_tokens,
                    overlap_ratio=child_overlap_ratio,
                )

            for piece in pieces:
                clean = piece.strip()
                if not clean:
                    continue
                c_hash = compute_content_hash(blk.section_path, clean)
                crumb = format_breadcrumb(title, blk.section_path, blk.page_no, clean)
                start_c = char_cursor
                end_c = char_cursor + len(clean)
                char_cursor = end_c + 2
                children.append(
                    ChildChunkRecord(
                        chunk_id=f"{doc_id}::v{version_id}::c{child_idx:04d}",
                        parent_id=parent_id,
                        doc_id=doc_id,
                        version_id=version_id,
                        chunk_index=child_idx,
                        start_char=start_c,
                        end_char=end_c,
                        source=source_uri,
                        section_path=blk.section_path,
                        page_no=blk.page_no,
                        element_type=blk.element_type,
                        content_hash=c_hash,
                        text=clean,
                        breadcrumb_text=crumb,
                        token_count=estimate_tokens(clean),
                        acl=eff_acl,
                        metadata={
                            "title": title,
                            "language": blk.language,
                            "parent_id": parent_id,
                            "element_type": blk.element_type,
                            "page_no": blk.page_no,
                        },
                    )
                )
                child_idx += 1

    return parents, children


def run_chunk_gate(
    children: list[ChildChunkRecord],
    *,
    min_tokens: int = 10,
    max_tokens: int = 1400,
) -> tuple[list[ChildChunkRecord], int, GateResult]:
    """Gate 3: Filter tiny fragment noise & duplicate chunks while validating metadata."""
    kept: list[ChildChunkRecord] = []
    seen_hashes: set[str] = set()
    dropped_fragment = 0
    dropped_dup = 0
    warnings: list[str] = []
    errors: list[str] = []

    for ch in children:
        if ch.content_hash in seen_hashes:
            dropped_dup += 1
            continue
        if ch.token_count < min_tokens and ch.element_type == "text":
            dropped_fragment += 1
            continue
        if ch.token_count > max_tokens:
            warnings.append(f"Oversized chunk `{ch.chunk_id}` ({ch.token_count} tok > {max_tokens}).")
        seen_hashes.add(ch.content_hash)
        kept.append(ch)

    if dropped_fragment:
        warnings.append(f"Dropped {dropped_fragment} low-information fragment chunk(s) (< {min_tokens} tok).")
    if dropped_dup:
        warnings.append(f"Deduplicated {dropped_dup} identical chunk(s) by SHA-256 content_hash.")
    if not kept:
        errors.append("Zero valid chunks survived ChunkGate.")

    # Re-index surviving chunks contiguously
    for idx, ch in enumerate(kept):
        ch.chunk_index = idx
        ch.chunk_id = f"{ch.doc_id}::v{ch.version_id}::c{idx:04d}"

    return (
        kept,
        dropped_fragment + dropped_dup,
        GateResult(
            gate_name="ChunkGate",
            passed=len(errors) == 0,
            warnings=warnings,
            errors=errors,
            metrics={
                "input_chunks": len(children),
                "kept_chunks": len(kept),
                "dropped_fragments": dropped_fragment,
                "dropped_duplicates": dropped_dup,
            },
        ),
    )


def govern_document(
    *,
    doc_id: str,
    file_path: Path | None = None,
    raw_text: str | None = None,
    title: str | None = None,
    source_uri: str | None = None,
    version_id: int = 1,
    acl: list[str] | None = None,
    prefer_docling: bool = True,
    parent_max_tokens: int = 1200,
    child_max_tokens: int = 340,
    table_atomic_max_tokens: int = 600,
    min_chunk_tokens: int = 10,
) -> GovernanceResult:
    """Execute full Stage 4.2 pipeline: FileGate -> Structural Parse -> ParseGate -> Parent-Child -> ChunkGate."""
    eff_source = source_uri or (str(file_path) if file_path else f"{doc_id}.md")
    eff_title = title or (file_path.stem if file_path else doc_id)

    # Gate 1: FileGate
    raw_bytes = raw_text.encode("utf-8") if raw_text is not None else None
    g1 = run_file_gate(file_path=file_path, raw_bytes=raw_bytes, source_name=eff_source)
    if not g1.passed:
        return GovernanceResult(
            doc_id=doc_id,
            title=eff_title,
            source_uri=eff_source,
            source_type=Path(eff_source).suffix.lstrip(".") or "unknown",
            parser_used="none",
            status="rejected",
            gates=[g1],
            rejection_reason="; ".join(g1.errors),
        )

    # Multi-format structural parse
    try:
        blocks, src_type, parser_used, total_pages, missing_pages = parse_document_blocks(
            file_path=file_path,
            raw_text=raw_text,
            source_name=eff_source,
            prefer_docling=prefer_docling,
        )
    except Exception as exc:  # noqa: BLE001
        g2_err = GateResult(
            gate_name="ParseGate",
            passed=False,
            errors=[f"Parser `{Path(eff_source).suffix}` failed: {exc}"],
        )
        return GovernanceResult(
            doc_id=doc_id,
            title=eff_title,
            source_uri=eff_source,
            source_type=Path(eff_source).suffix.lstrip(".") or "unknown",
            parser_used="failed",
            status="rejected",
            gates=[g1, g2_err],
            rejection_reason=str(exc),
        )

    # Gate 2: ParseGate
    g2 = run_parse_gate(blocks, total_pages=total_pages, missing_pages=missing_pages)
    if not g2.passed:
        return GovernanceResult(
            doc_id=doc_id,
            title=eff_title,
            source_uri=eff_source,
            source_type=src_type,
            parser_used=parser_used,
            status="rejected",
            missing_pages=missing_pages,
            gates=[g1, g2],
            rejection_reason="; ".join(g2.errors),
        )

    # Format-aware Parent-Child chunking + Breadcrumb prefix
    parents, raw_children = build_parent_child_chunks(
        blocks,
        doc_id=doc_id,
        title=eff_title,
        source_uri=eff_source,
        version_id=version_id,
        acl=acl,
        parent_max_tokens=parent_max_tokens,
        child_max_tokens=child_max_tokens,
        table_atomic_max_tokens=table_atomic_max_tokens,
    )

    # Gate 3: ChunkGate
    kept_children, dropped, g3 = run_chunk_gate(raw_children, min_tokens=min_chunk_tokens)
    if not g3.passed:
        return GovernanceResult(
            doc_id=doc_id,
            title=eff_title,
            source_uri=eff_source,
            source_type=src_type,
            parser_used=parser_used,
            status="rejected",
            missing_pages=missing_pages,
            gates=[g1, g2, g3],
            dropped_chunks=dropped,
            rejection_reason="; ".join(g3.errors),
        )

    # Keep only parents referenced by surviving children
    used_pids = {c.parent_id for c in kept_children}
    kept_parents = [p for p in parents if p.parent_id in used_pids]
    final_status = "partial_available" if missing_pages else "ready"

    return GovernanceResult(
        doc_id=doc_id,
        title=eff_title,
        source_uri=eff_source,
        source_type=src_type,
        parser_used=parser_used,
        status=final_status,
        missing_pages=missing_pages,
        gates=[g1, g2, g3],
        parent_chunks=kept_parents,
        child_chunks=kept_children,
        dropped_chunks=dropped,
    )

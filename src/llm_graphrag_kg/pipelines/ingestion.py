"""Document loading and sliding-window chunking with character-level provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

from llm_graphrag_kg.core.schema import Document, TextChunk

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}


class DocumentIngestor:
    """Loads UTF-8 text/Markdown files and splits them into traceable `TextChunk`s."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_directory(self, input_dir: Path, base_dir: Path | None = None) -> list[Document]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        files = sorted(
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        documents: list[Document] = []
        ref_root = (base_dir or input_dir).resolve()

        for file_path in files:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            try:
                rel_source = file_path.resolve().relative_to(ref_root).as_posix()
            except ValueError:
                rel_source = file_path.name

            doc_hash = hashlib.sha1(rel_source.encode("utf-8")).hexdigest()[:8]
            doc_id = f"doc_{doc_hash}"
            title = self._extract_title(content, fallback=file_path.stem)
            documents.append(
                Document(
                    doc_id=doc_id,
                    source=rel_source,
                    title=title,
                    content=content,
                    metadata={"extension": file_path.suffix.lower()},
                )
            )
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for doc in documents:
            chunks.extend(self.chunk_document(doc))
        return chunks

    def chunk_document(self, doc: Document) -> list[TextChunk]:
        text = doc.content
        n = len(text)
        if n == 0:
            return []

        result: list[TextChunk] = []
        start = 0
        idx = 0
        step_floor = max(1, self.chunk_size - self.chunk_overlap)

        while start < n:
            raw_end = min(n, start + self.chunk_size)
            end = self._adjust_boundary(text, start, raw_end)
            segment = text[start:end].strip()
            if segment:
                result.append(
                    TextChunk(
                        chunk_id=f"{doc.doc_id}::chunk_{idx:04d}",
                        doc_id=doc.doc_id,
                        source=doc.source,
                        index=idx,
                        text=segment,
                        start_char=start,
                        end_char=end,
                        metadata={"title": doc.title},
                    )
                )
                idx += 1

            if end >= n:
                break
            next_start = max(start + 1, end - self.chunk_overlap)
            # Prevent tiny tail jitter when boundary adjustment moved far back
            if next_start <= start:
                next_start = start + step_floor
            start = next_start

        return result

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or fallback
        return fallback

    @staticmethod
    def _adjust_boundary(text: str, start: int, raw_end: int) -> int:
        if raw_end >= len(text):
            return len(text)
        # Prefer ending on a paragraph or sentence terminator within the last 30% of the window.
        min_end = start + int((raw_end - start) * 0.7)
        window = text[min_end:raw_end]
        for sep in ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? "):
            pos = window.rfind(sep)
            if pos != -1:
                return min_end + pos + len(sep)
        return raw_end

"""Structure-aware, token-aware chunking.

Two-level strategy:

1. **Split on Markdown headings** so a chunk never straddles two sections and
   always knows its section path ("torch.utils.data > Dataset Types > ...").
2. **Recursively split long sections** with LangChain's
   ``RecursiveCharacterTextSplitter``, measuring length in *tokens of the
   embedding model* (not characters), preferring paragraph, then line, then
   sentence boundaries.

Each chunk's text starts with its section path (a "contextual chunk header").
Without it, a chunk like "Set ``num_workers`` to 0 ..." loses the information
that it is about DataLoader reproducibility, which hurts both BM25 and dense
retrieval. The header counts toward the token budget, so chunks still fit the
encoder window.

LangChain's ``MarkdownHeaderTextSplitter`` is deliberately not used: it strips
leading whitespace from every line, which destroys the indentation of Python
code examples in technical documentation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import Chunk, ChunkMetadata, Document

TokenCounter = Callable[[str], int]

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
# Paragraphs, then lines, then sentences, then words: keeps code and prose intact when possible.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SECTION_SEPARATOR = " > "


@dataclass(frozen=True)
class Section:
    """A run of text under one heading path."""

    path: tuple[str, ...]
    text: str


def split_sections(markdown: str, max_depth: int = 4) -> list[Section]:
    """Split Markdown into sections at headings up to ``max_depth``.

    Headings inside fenced code blocks (e.g. ``# comment`` in Python) are ignored.
    Deeper headings stay inside their parent section's text.
    """
    sections: list[Section] = []
    path: list[str] = []
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(path=tuple(path), text=text))
        buffer.clear()

    for line in markdown.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        heading = None if in_code else _HEADING_RE.match(line)
        if heading and len(heading.group("hashes")) <= max_depth:
            flush()
            level = len(heading.group("hashes"))
            del path[level - 1 :]
            path.extend([""] * (level - 1 - len(path)))  # tolerate skipped levels (# then ###)
            path.append(heading.group("title").strip("` "))
            continue
        buffer.append(line)
    flush()
    return sections


class MarkdownChunker:
    """Turns `Document`s into token-bounded `Chunk`s with full provenance metadata."""

    def __init__(self, chunk_size: int, chunk_overlap: int, count_tokens: TokenCounter) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.count_tokens = count_tokens

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        return [chunk for doc in documents for chunk in self.chunk_document(doc)]

    def chunk_document(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in split_sections(document.content):
            section_name = _section_name(section.path, document.title)
            header = f"{section_name}\n\n"
            for body in self._split_body(section.text, header):
                content = header + body
                chunks.append(self._make_chunk(document, section_name, content, len(chunks)))
        return chunks

    def _split_body(self, text: str, header: str) -> list[str]:
        """Split a section body so that ``header + body`` fits in ``chunk_size`` tokens."""
        # Keep at least half the budget for content even under a very long heading path.
        budget = max(self.chunk_size - self.count_tokens(header), self.chunk_size // 2)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=min(self.chunk_overlap, budget // 2),
            length_function=self.count_tokens,
            separators=_SEPARATORS,
            keep_separator=True,
            strip_whitespace=True,
        )
        return [piece for piece in splitter.split_text(text) if piece.strip()]

    def _make_chunk(self, document: Document, section: str, content: str, index: int) -> Chunk:
        page_part = f"p{document.page}-" if document.page is not None else ""
        metadata = ChunkMetadata(
            source=document.source,
            document=document.title,
            section=section,
            chunk_id=f"{document.source}#{page_part}{index:04d}",
            chunk_index=index,
            token_count=self.count_tokens(content),
            page=document.page,
        )
        return Chunk(content=content, metadata=metadata)


def _section_name(path: tuple[str, ...], title: str) -> str:
    """Human-readable section path; documents without headings fall back to their title."""
    parts = [part for part in path if part]
    if not parts:
        return title
    if parts[0] != title:  # text before/without an H1: anchor the path at the document title
        parts.insert(0, title)
    return SECTION_SEPARATOR.join(parts)

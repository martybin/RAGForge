"""Document loaders.

Each loader turns one file into one or more `Document`s. Loaders are looked up
by file extension, so supporting a new format (PDF, HTML) means writing one
function and registering it in ``LOADERS``. Paginated formats should emit one
`Document` per page with ``page`` set, which the chunker carries into metadata.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path

from app.ingestion.cleaning import clean_markdown
from app.models.schemas import Document

logger = logging.getLogger(__name__)

_H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*#*\s*$")

Loader = Callable[[Path, str], list[Document]]


def load_markdown(path: Path, source: str) -> list[Document]:
    """Load a Markdown/MyST file, cleaning doc-builder markup."""
    content = clean_markdown(path.read_text(encoding="utf-8"))
    return [Document(content=content, source=source, title=_markdown_title(content, path))]


def load_text(path: Path, source: str) -> list[Document]:
    """Load a plain-text file; normalizes line endings and trailing whitespace."""
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    content = "\n".join(line.rstrip() for line in raw.split("\n")).strip() + "\n"
    return [Document(content=content, source=source, title=_title_from_filename(path))]


LOADERS: dict[str, Loader] = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
}


def load_documents(root: Path) -> list[Document]:
    """Load every supported file under ``root`` (recursively, in sorted order).

    Sorting makes ingestion deterministic, so chunk ids are stable across runs.
    Unsupported or unreadable files are logged and skipped rather than aborting
    the whole ingestion.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {root}")

    documents: list[Document] = []
    for path in _iter_files(root):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.debug("skipping_unsupported_file", extra={"path": str(path)})
            continue
        source = path.relative_to(root).as_posix()
        try:
            loaded = loader(path, source)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("document_load_failed", extra={"source": source, "error": str(exc)})
            continue
        documents.extend(doc for doc in loaded if doc.content.strip())

    logger.info("documents_loaded", extra={"count": len(documents), "root": str(root)})
    return documents


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path


def _markdown_title(content: str, path: Path) -> str:
    """First level-1 heading outside code blocks, else a name derived from the file."""
    in_code = False
    for line in content.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code and (match := _H1_RE.match(line)):
            return match.group("title").strip("` ")
    return _title_from_filename(path)


def _title_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().capitalize()

"""Persistence for the two index artifacts written at ingestion time.

* A LangChain `Chroma` vector store (cosine space) for dense search.
* A JSONL snapshot of all chunks, from which the in-memory BM25 index is rebuilt
  at startup (lexical search). JSONL keeps it human-inspectable and diffable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.models.schemas import Chunk


class IndexNotFoundError(RuntimeError):
    """Raised when the index has not been built yet (run scripts/ingest.py)."""


def open_vectorstore(
    chroma_dir: Path, name: str, embeddings: Embeddings, *, reset: bool = False
) -> Chroma:
    """Open (or create) the Chroma vector store using cosine distance."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )
    if reset and name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)
    return Chroma(
        client=client,
        collection_name=name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def save_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    """Write chunks as JSON lines; returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
            count += 1
    return count


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise IndexNotFoundError(f"No chunk snapshot at {path}. Run `python scripts/ingest.py`.")
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate(json.loads(line)) for line in handle if line.strip()]

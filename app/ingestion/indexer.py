"""Indexing: embed chunks and persist them for dense (Chroma) and lexical (BM25) search."""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Sequence

from pydantic import BaseModel

from app.config.settings import Settings
from app.embeddings import Embedder
from app.ingestion.chunker import MarkdownChunker
from app.ingestion.loaders import load_documents
from app.models.schemas import Chunk
from app.storage import open_vectorstore, save_chunks

logger = logging.getLogger(__name__)

_CHROMA_BATCH = 256


class IngestionReport(BaseModel):
    documents: int
    chunks: int
    mean_tokens: float
    max_tokens: int
    min_tokens: int
    elapsed_s: float


def index_chunks(chunks: Sequence[Chunk], embedder: Embedder, settings: Settings) -> None:
    """Rebuild the vector index and chunk snapshot from scratch.

    A full rebuild keeps the index consistent with the source files (deleted or
    edited docs leave no stale vectors); incremental upserts are not worth the
    complexity at this corpus size.
    """
    if not chunks:
        raise ValueError("No chunks to index; is the raw data directory empty?")
    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk ids; chunk ids must be unique")

    store = open_vectorstore(settings.chroma_dir, settings.chroma_collection, embedder, reset=True)
    started = time.perf_counter()
    for start in range(0, len(chunks), _CHROMA_BATCH):
        batch = chunks[start : start + _CHROMA_BATCH]
        store.add_documents([c.to_document() for c in batch], ids=[c.chunk_id for c in batch])
    logger.info(
        "chunks_embedded",
        extra={"count": len(chunks), "embed_ms": round((time.perf_counter() - started) * 1000)},
    )
    save_chunks(chunks, settings.chunks_path)
    logger.info(
        "index_built",
        extra={"vectors": store._collection.count(), "snapshot": str(settings.chunks_path)},
    )


def run_ingestion(settings: Settings, embedder: Embedder | None = None) -> IngestionReport:
    """Load -> clean -> chunk -> embed -> index everything under ``raw_data_dir``."""
    started = time.perf_counter()
    embedder = embedder or Embedder(
        settings.embedding_model,
        query_instruction=settings.embedding_query_instruction,
        device=settings.device,
        batch_size=settings.embedding_batch_size,
    )
    documents = load_documents(settings.raw_data_dir)
    chunker = MarkdownChunker(settings.chunk_size, settings.chunk_overlap, embedder.count_tokens)
    chunks = chunker.chunk_documents(documents)
    index_chunks(chunks, embedder, settings)

    token_counts = [c.metadata.token_count for c in chunks]
    return IngestionReport(
        documents=len(documents),
        chunks=len(chunks),
        mean_tokens=round(statistics.fmean(token_counts), 1),
        max_tokens=max(token_counts),
        min_tokens=min(token_counts),
        elapsed_s=round(time.perf_counter() - started, 2),
    )

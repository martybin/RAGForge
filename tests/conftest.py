"""Shared fixtures: a tiny in-memory corpus and helpers that need no model downloads."""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.models.schemas import Chunk, ChunkMetadata, RetrievalMethod, RetrievedChunk


def make_chunk(chunk_id: str, content: str, source: str | None = None) -> Chunk:
    source = source or chunk_id.split("#")[0]
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            source=source,
            document=source.rsplit("/", 1)[-1],
            section=f"{source} > section",
            chunk_id=chunk_id,
            chunk_index=int(chunk_id.rsplit("#", 1)[-1]) if "#" in chunk_id else 0,
            token_count=len(content.split()),
        ),
    )


def make_result(
    chunk_id: str,
    score: float,
    content: str | None = None,
    method: RetrievalMethod = RetrievalMethod.RERANKED,
    rank: int = 1,
) -> RetrievedChunk:
    chunk = make_chunk(chunk_id, content or f"content of {chunk_id}")
    return RetrievedChunk(
        content=chunk.content, metadata=chunk.metadata, score=score, method=method, rank=rank
    )


@pytest.fixture
def corpus() -> list[Chunk]:
    return [
        make_chunk(
            "data.md#0000", "DataLoader supports num_workers for multi-process data loading."
        ),
        make_chunk("data.md#0001", "Set pin_memory=True to enable fast host to GPU copies."),
        make_chunk(
            "randomness.md#0000", "Use torch.manual_seed to seed the random number generator."
        ),
        make_chunk(
            "optim.md#0000", "Call optimizer.zero_grad before loss.backward in each iteration."
        ),
        make_chunk("amp.md#0000", "Automatic mixed precision uses autocast and GradScaler."),
    ]


@pytest.fixture
def kb_settings(tmp_path) -> Settings:
    """Settings pointing at a tiny on-disk knowledge base and a temporary index directory."""
    raw = tmp_path / "raw"
    (raw / "notes").mkdir(parents=True)
    (raw / "notes" / "randomness.md").write_text(
        "# Reproducibility\n\n## Seeds\n\nUse {func}`torch.manual_seed` to seed the RNG.\n\n"
        "## DataLoader\n\nUse worker_init_fn and a generator for reproducible workers.\n",
        encoding="utf-8",
    )
    (raw / "optim.md").write_text(
        "# torch.optim\n\nCall optimizer.zero_grad before loss.backward.\n", encoding="utf-8"
    )
    return Settings(
        _env_file=None,
        raw_data_dir=raw,
        processed_data_dir=tmp_path / "processed",
        chunk_size=50,
        chunk_overlap=10,
    )

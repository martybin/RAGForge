"""Context filtering: the last deterministic step before the LLM sees anything.

Given reranked chunks (most relevant first) it

1. drops chunks whose reranker score is below a threshold,
2. drops duplicates and near-duplicates (same text indexed twice, or heavily
   overlapping neighbours), keeping the higher-ranked copy,
3. enforces a total token budget for the prompt context.

Every dropped chunk is reported with a reason, so the final context is fully
inspectable in the UI / API debug output.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from app.models.schemas import DroppedChunk, RetrievedChunk

_WORD_RE = re.compile(r"\w+")


class FilteredContext(BaseModel):
    kept: list[RetrievedChunk]
    dropped: list[DroppedChunk]


def shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    """Word n-grams used for near-duplicate detection."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < size:
        return {tuple(words)}
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def filter_context(
    chunks: Sequence[RetrievedChunk],
    *,
    min_score: float,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    near_duplicate_threshold: float = 0.8,
) -> FilteredContext:
    """Select the final LLM context from reranked chunks (input order = priority order)."""
    kept: list[RetrievedChunk] = []
    kept_shingles: list[set] = []
    dropped: list[DroppedChunk] = []
    used_tokens = 0

    def drop(chunk: RetrievedChunk, reason: str) -> None:
        dropped.append(DroppedChunk(chunk_id=chunk.chunk_id, reason=reason))

    for chunk in chunks:
        if chunk.score < min_score:
            drop(chunk, f"score {chunk.score:.3f} below threshold {min_score}")
            continue

        chunk_shingles = shingles(chunk.content)
        duplicate_of = next(
            (
                previous.chunk_id
                for previous, previous_shingles in zip(kept, kept_shingles, strict=True)
                if previous.chunk_id == chunk.chunk_id
                or jaccard(chunk_shingles, previous_shingles) >= near_duplicate_threshold
            ),
            None,
        )
        if duplicate_of:
            drop(chunk, f"near-duplicate of {duplicate_of}")
            continue

        tokens = count_tokens(chunk.content)
        if used_tokens + tokens > max_tokens:
            drop(chunk, f"context token budget exceeded ({used_tokens} + {tokens} > {max_tokens})")
            continue

        kept.append(chunk)
        kept_shingles.append(chunk_shingles)
        used_tokens += tokens

    return FilteredContext(kept=kept, dropped=dropped)

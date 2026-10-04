"""Lexical retrieval with BM25 (Okapi).

Dense embeddings are good at paraphrase but weak at exact identifiers such as
``pin_memory``, ``worker_init_fn`` or ``CUBLAS_WORKSPACE_CONFIG``, which is
exactly what users of technical documentation type. BM25 covers that gap.

The tokenizer is tailored to code-heavy text: identifiers are kept whole *and*
split into their parts, so ``num_workers`` matches both the exact identifier
(high IDF, very precise) and prose mentioning "workers".
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from app.models.schemas import Chunk, RetrievalMethod, RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    """a about above after again all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into
    is it its itself just me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your
    yours yourself yourselves""".split()  # noqa: SIM905 (a word list reads better than a literal)
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into identifier-aware terms, drop stopwords."""
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        parts = [p for p in token.split("_") if p]
        if len(parts) > 1:
            tokens.append(token.strip("_"))  # whole identifier: num_workers
        tokens.extend(parts)  # and its parts: num, workers
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Retriever:
    """In-memory BM25 index over all chunks (rebuilt from the chunk snapshot at startup)."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self._chunks = list(chunks)
        self._position = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}
        self._index = BM25Okapi([tokenize(c.content) for c in self._chunks], k1=k1, b=b)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Top-k chunks by BM25 score; chunks sharing no term with the query are excluded."""
        scores = self._scores(query)
        # Stable sort on -score keeps corpus order for ties, so results are deterministic.
        order = np.argsort(-scores, kind="stable")[:top_k]
        results = []
        for idx in order:
            if scores[idx] <= 0:
                break
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=float(scores[idx]),
                    method=RetrievalMethod.BM25,
                    rank=len(results) + 1,
                )
            )
        return results

    def score(self, query: str, chunk_ids: Iterable[str]) -> dict[str, float]:
        """BM25 scores of specific chunks (scores hybrid candidates found by dense search)."""
        scores = self._scores(query)
        return {
            cid: float(scores[self._position[cid]]) for cid in chunk_ids if cid in self._position
        }

    def _scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        if not terms:
            return np.zeros(len(self._chunks))
        return np.asarray(self._index.get_scores(terms), dtype=float)

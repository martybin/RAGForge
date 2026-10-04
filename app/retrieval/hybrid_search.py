"""Hybrid retrieval: weighted fusion of dense and BM25 scores.

    hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm

Raw scores live on incompatible scales (cosine similarity is roughly 0.3-0.9 for
BGE, BM25 is unbounded and query-length dependent), so both are min-max
normalized over the candidate set before mixing.

Candidates are the union of the dense top-k and BM25 top-k. Every candidate is
then scored by *both* retrievers. A chunk found only by BM25 gets its real
cosine similarity rather than an implicit 0, so it is not punished just for
ranking 11th in dense search.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

from pydantic import BaseModel

from app.models.schemas import RetrievalMethod, RetrievedChunk
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.vector_search import DenseRetriever

logger = logging.getLogger(__name__)


class HybridResult(BaseModel):
    """All three result lists, so the pipeline can expose each stage."""

    dense: list[RetrievedChunk]
    bm25: list[RetrievedChunk]
    hybrid: list[RetrievedChunk]


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Scale scores to [0, 1]; if all scores are equal, they all map to 1."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-12:
        return dict.fromkeys(scores, 1.0)
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def fuse_scores(
    dense_scores: Mapping[str, float],
    bm25_scores: Mapping[str, float],
    alpha: float,
) -> dict[str, dict[str, float]]:
    """Combine per-candidate raw scores into hybrid scores.

    Both mappings should cover the same candidate ids; an id missing from one
    side is treated as the minimum of that side (normalized 0).
    Returns ``{chunk_id: {"hybrid", "dense", "bm25", "dense_norm", "bm25_norm"}}``.
    """
    dense_norm = min_max_normalize(dense_scores)
    bm25_norm = min_max_normalize(bm25_scores)
    fused = {}
    for cid in dense_scores.keys() | bm25_scores.keys():
        d, b = dense_norm.get(cid, 0.0), bm25_norm.get(cid, 0.0)
        fused[cid] = {
            "hybrid": alpha * d + (1 - alpha) * b,
            "dense": dense_scores.get(cid, 0.0),
            "bm25": bm25_scores.get(cid, 0.0),
            "dense_norm": d,
            "bm25_norm": b,
        }
    return fused


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        alpha: float,
        top_k_dense: int,
        top_k_bm25: int,
        top_k_hybrid: int,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.dense = dense
        self.bm25 = bm25
        self.alpha = alpha
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_hybrid = top_k_hybrid

    def search(self, query: str) -> HybridResult:
        started = time.perf_counter()
        query_embedding = self.dense.embed_query(query)
        dense_results = self.dense.search_by_embedding(query_embedding, self.top_k_dense)
        bm25_results = self.bm25.search(query, self.top_k_bm25)

        # Score the union of candidates with both retrievers.
        candidates = {r.chunk_id: r for r in bm25_results} | {r.chunk_id: r for r in dense_results}
        dense_scores = {r.chunk_id: r.score for r in dense_results}
        dense_scores |= self.dense.score(query_embedding, candidates.keys() - dense_scores.keys())
        bm25_scores = self.bm25.score(query, candidates.keys())

        fused = fuse_scores(dense_scores, bm25_scores, self.alpha)
        # Sort by hybrid score; chunk_id breaks ties so ordering is deterministic.
        ranked = sorted(fused.items(), key=lambda item: (-item[1]["hybrid"], item[0]))
        hybrid_results = [
            RetrievedChunk(
                content=candidates[cid].content,
                metadata=candidates[cid].metadata,
                score=scores["hybrid"],
                method=RetrievalMethod.HYBRID,
                rank=rank,
                component_scores={k: v for k, v in scores.items() if k != "hybrid"},
            )
            for rank, (cid, scores) in enumerate(ranked[: self.top_k_hybrid], start=1)
        ]
        logger.info(
            "hybrid_retrieval",
            extra={
                "dense_hits": len(dense_results),
                "bm25_hits": len(bm25_results),
                "candidates": len(candidates),
                "returned": len(hybrid_results),
                "alpha": self.alpha,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return HybridResult(dense=dense_results, bm25=bm25_results, hybrid=hybrid_results)

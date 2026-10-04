import numpy as np
import pytest

from app.models.schemas import RetrievalMethod, RetrievedChunk
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever, fuse_scores, min_max_normalize


class FakeDenseRetriever:
    """Stands in for DenseRetriever with fixed cosine similarities."""

    def __init__(self, corpus, similarities: dict[str, float]):
        self._chunks = {c.chunk_id: c for c in corpus}
        self._similarities = similarities
        self.scored_ids: set[str] = set()

    def embed_query(self, query):
        return np.zeros(3)

    def search_by_embedding(self, query_embedding, top_k):
        ranked = sorted(self._similarities.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            RetrievedChunk(
                content=self._chunks[cid].content,
                metadata=self._chunks[cid].metadata,
                score=score,
                method=RetrievalMethod.DENSE,
                rank=rank,
            )
            for rank, (cid, score) in enumerate(ranked, start=1)
        ]

    def score(self, query_embedding, chunk_ids):
        ids = set(chunk_ids)
        self.scored_ids |= ids
        return {cid: self._similarities[cid] for cid in ids}


SIMILARITIES = {
    "data.md#0000": 0.82,
    "data.md#0001": 0.74,
    "randomness.md#0000": 0.55,
    "optim.md#0000": 0.50,
    "amp.md#0000": 0.45,
}


def test_min_max_normalize_maps_to_unit_interval():
    assert min_max_normalize({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}


def test_min_max_normalize_handles_constant_and_empty_input():
    assert min_max_normalize({"a": 0.7, "b": 0.7}) == {"a": 1.0, "b": 1.0}
    assert min_max_normalize({}) == {}


def test_fuse_scores_applies_alpha_weighting():
    fused = fuse_scores({"a": 0.9, "b": 0.5}, {"a": 0.0, "b": 10.0}, alpha=0.6)
    assert fused["a"]["hybrid"] == pytest.approx(0.6)  # dense best, bm25 worst
    assert fused["b"]["hybrid"] == pytest.approx(0.4)  # dense worst, bm25 best
    assert fused["b"]["bm25"] == 10.0 and fused["b"]["bm25_norm"] == 1.0


@pytest.mark.parametrize(
    ("alpha", "expected_first"),
    [(1.0, "data.md#0000"), (0.0, "data.md#0001")],
)
def test_alpha_extremes_reduce_to_single_retriever(corpus, alpha, expected_first):
    retriever = HybridRetriever(
        FakeDenseRetriever(corpus, SIMILARITIES), BM25Retriever(corpus), alpha, 2, 2, 5
    )
    result = retriever.search("pin_memory")
    assert result.hybrid[0].chunk_id == expected_first


def test_bm25_only_candidates_get_their_real_dense_score(corpus):
    dense = FakeDenseRetriever(corpus, SIMILARITIES)
    retriever = HybridRetriever(dense, BM25Retriever(corpus), 0.5, 1, 1, 5)
    result = retriever.search("GradScaler")

    # Dense top-1 is data.md#0000; BM25 top-1 is amp.md#0000, which dense search missed.
    assert {r.chunk_id for r in result.hybrid} == {"data.md#0000", "amp.md#0000"}
    assert dense.scored_ids == {"amp.md#0000"}
    amp = next(r for r in result.hybrid if r.chunk_id == "amp.md#0000")
    assert amp.component_scores["dense"] == pytest.approx(0.45)


def test_hybrid_results_are_ranked_and_exposed_per_stage(corpus):
    retriever = HybridRetriever(
        FakeDenseRetriever(corpus, SIMILARITIES), BM25Retriever(corpus), 0.6, 3, 3, 4
    )
    result = retriever.search("seed the random number generator")
    assert len(result.dense) == 3
    assert result.bm25[0].chunk_id == "randomness.md#0000"
    scores = [r.score for r in result.hybrid]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in result.hybrid] == list(range(1, len(result.hybrid) + 1))
    assert all(r.method is RetrievalMethod.HYBRID for r in result.hybrid)
    assert len(result.hybrid) <= 4


def test_invalid_alpha_is_rejected(corpus):
    with pytest.raises(ValueError):
        HybridRetriever(
            FakeDenseRetriever(corpus, SIMILARITIES), BM25Retriever(corpus), 1.2, 1, 1, 1
        )

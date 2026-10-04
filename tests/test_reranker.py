"""Reranker logic tests with the cross-encoder replaced by a deterministic fake."""

import math

import pytest

from app.models.schemas import RetrievalMethod
from app.retrieval import reranker as reranker_module
from app.retrieval.reranker import Reranker, sigmoid
from tests.conftest import make_result


class FakeCrossEncoder:
    """Scores a pair by how many query words the passage contains (as a logit)."""

    def __init__(self, model_name, device=None, max_length=512):
        self.model_name = model_name

    def predict(self, pairs, **kwargs):
        return [
            float(sum(w in passage.lower() for w in query.lower().split())) - 1.0
            for query, passage in pairs
        ]


@pytest.fixture
def reranker(monkeypatch):
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    return Reranker("fake-reranker")


def candidates():
    return [
        make_result("a#0", 0.9, "nothing relevant here", method=RetrievalMethod.HYBRID, rank=1),
        make_result(
            "b#0", 0.5, "pin memory for faster cuda copies", method=RetrievalMethod.HYBRID, rank=2
        ),
        make_result("c#0", 0.4, "pin the version", method=RetrievalMethod.HYBRID, rank=3),
    ]


def test_rerank_reorders_by_cross_encoder_and_keeps_top_k(reranker):
    results = reranker.rerank("pin memory cuda", candidates(), top_k=2)
    assert [r.chunk_id for r in results] == ["b#0", "c#0"]
    assert [r.rank for r in results] == [1, 2]
    assert all(r.method is RetrievalMethod.RERANKED for r in results)


def test_scores_are_sigmoid_of_logits_and_previous_stage_is_kept(reranker):
    top = reranker.rerank("pin memory cuda", candidates(), top_k=1)[0]
    assert top.component_scores["rerank_logit"] == 2.0
    assert top.score == pytest.approx(sigmoid(2.0))
    assert top.component_scores["hybrid"] == 0.5
    assert top.component_scores["hybrid_rank"] == 2.0


def test_empty_candidates_return_empty(reranker):
    assert reranker.rerank("anything", [], top_k=5) == []


def test_sigmoid_is_bounded():
    assert sigmoid(0) == 0.5
    assert 0 < sigmoid(-20) < sigmoid(20) < 1
    assert sigmoid(1) == pytest.approx(1 / (1 + math.e**-1))

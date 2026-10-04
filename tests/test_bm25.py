import pytest

from app.models.schemas import RetrievalMethod
from app.retrieval.bm25_search import BM25Retriever, tokenize


def test_tokenize_keeps_identifiers_whole_and_split():
    assert tokenize("Set num_workers > 0") == ["set", "num_workers", "num", "workers", "0"]


def test_tokenize_splits_dotted_names_and_drops_stopwords():
    assert tokenize("The torch.manual_seed function") == [
        "torch",
        "manual_seed",
        "manual",
        "seed",
        "function",
    ]


def test_exact_identifier_ranks_first(corpus):
    results = BM25Retriever(corpus).search("what does pin_memory do?", top_k=3)
    assert results[0].chunk_id == "data.md#0001"
    assert results[0].method is RetrievalMethod.BM25
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_results_are_sorted_and_exclude_zero_scores(corpus):
    results = BM25Retriever(corpus).search("GradScaler autocast", top_k=5)
    assert [r.chunk_id for r in results] == ["amp.md#0000"]


def test_query_without_known_terms_returns_nothing(corpus):
    assert BM25Retriever(corpus).search("the of and", top_k=5) == []


def test_score_returns_scores_for_requested_ids_only(corpus):
    scores = BM25Retriever(corpus).score(
        "zero_grad backward", ["optim.md#0000", "amp.md#0000", "missing#1"]
    )
    assert set(scores) == {"optim.md#0000", "amp.md#0000"}
    assert scores["optim.md#0000"] > scores["amp.md#0000"] == 0.0


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        BM25Retriever([])

"""Dense retrieval against a real Chroma collection built with a fake embedder."""

import pytest

from app.ingestion.indexer import run_ingestion
from app.retrieval.vector_search import DenseRetriever
from app.storage import IndexNotFoundError, open_collection
from tests.fakes import HashingEmbedder


def test_dense_retriever_finds_the_matching_chunk(kb_settings):
    embedder = HashingEmbedder()
    run_ingestion(kb_settings, embedder=embedder)
    retriever = DenseRetriever(
        embedder, open_collection(kb_settings.chroma_dir, kb_settings.chroma_collection)
    )

    results = retriever.search("worker_init_fn generator reproducible workers", top_k=2)
    assert results[0].chunk_id == "notes/randomness.md#0001"
    assert results[0].metadata.section == "Reproducibility > DataLoader"
    assert results[0].score > results[1].score
    assert [r.rank for r in results] == [1, 2]

    exact = retriever.score(embedder.embed_query("worker_init_fn"), [results[1].chunk_id])
    assert set(exact) == {results[1].chunk_id}


def test_empty_collection_raises_index_not_found(kb_settings):
    with pytest.raises(IndexNotFoundError):
        DenseRetriever(HashingEmbedder(), open_collection(kb_settings.chroma_dir, "empty"))

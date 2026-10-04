"""Ingestion end to end (load -> clean -> chunk -> embed -> Chroma + JSONL) with a fake embedder."""

import pytest

from app.ingestion.indexer import index_chunks, run_ingestion
from app.storage import IndexNotFoundError, load_chunks, open_collection
from tests.fakes import HashingEmbedder


def test_ingestion_builds_consistent_vector_index_and_snapshot(kb_settings):
    report = run_ingestion(kb_settings, embedder=HashingEmbedder())
    chunks = load_chunks(kb_settings.chunks_path)
    collection = open_collection(kb_settings.chroma_dir, kb_settings.chroma_collection)

    assert report.documents == 2
    assert report.chunks == len(chunks) == collection.count() == 3
    stored = collection.get(ids=[chunks[0].chunk_id], include=["metadatas", "documents"])
    assert stored["documents"][0] == chunks[0].content
    assert stored["metadatas"][0]["section"] == chunks[0].metadata.section
    assert "page" not in stored["metadatas"][0]  # None values are not sent to Chroma
    assert "{func}" not in chunks[0].content  # cleaned before chunking


def test_reingestion_replaces_the_index_instead_of_appending(kb_settings):
    run_ingestion(kb_settings, embedder=HashingEmbedder())
    run_ingestion(kb_settings, embedder=HashingEmbedder())
    assert open_collection(kb_settings.chroma_dir, kb_settings.chroma_collection).count() == 3


def test_missing_snapshot_raises_a_clear_error(kb_settings):
    with pytest.raises(IndexNotFoundError):
        load_chunks(kb_settings.chunks_path)


def test_duplicate_chunk_ids_are_rejected(kb_settings, corpus):
    with pytest.raises(ValueError, match="Duplicate"):
        index_chunks([corpus[0], corpus[0]], HashingEmbedder(), kb_settings)

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ChunkMetadata,
    QueryRequest,
    RetrievalMethod,
    RetrievedChunk,
)


def make_metadata(**overrides) -> ChunkMetadata:
    fields = {
        "source": "pytorch/notes/autograd.md",
        "document": "Autograd mechanics",
        "section": "Autograd mechanics > Saved tensors",
        "chunk_id": "pytorch/notes/autograd.md#0003",
        "chunk_index": 3,
        "token_count": 120,
    }
    return ChunkMetadata(**(fields | overrides))


def test_chroma_metadata_drops_none_values():
    metadata = make_metadata(page=None)
    assert "page" not in metadata.to_chroma()
    assert make_metadata(page=2).to_chroma()["page"] == 2


def test_retrieved_chunk_exposes_id_and_source_in_json():
    result = RetrievedChunk(
        content="text", metadata=make_metadata(), score=0.9, method=RetrievalMethod.DENSE, rank=1
    )
    payload = result.model_dump(mode="json")
    assert payload["chunk_id"] == "pytorch/notes/autograd.md#0003"
    assert payload["source"] == "pytorch/notes/autograd.md"
    assert payload["method"] == "dense"


def test_retrieved_chunk_round_trips_to_chunk():
    result = RetrievedChunk(
        content="text", metadata=make_metadata(), score=0.9, method=RetrievalMethod.BM25, rank=2
    )
    assert result.as_chunk().chunk_id == result.chunk_id


def test_query_request_strips_whitespace():
    assert QueryRequest(question="  What is autograd?  ").question == "What is autograd?"


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_query_request_rejects_invalid_questions(question):
    with pytest.raises(ValidationError):
        QueryRequest(question=question)

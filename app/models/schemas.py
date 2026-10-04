"""Pydantic data models shared by every stage of the pipeline.

Data flows through the system as:

    Document  --chunker-->  Chunk  --retrievers-->  RetrievedChunk
              --pipeline--> QueryResponse (answer + Citations + RetrievalTrace)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class Document(BaseModel):
    """A loaded source document (or one page of a paginated document)."""

    content: str
    source: str = Field(description="Path relative to the raw data directory, POSIX style.")
    title: str
    page: int | None = None


class ChunkMetadata(BaseModel):
    """Provenance of a chunk; everything a citation needs."""

    model_config = ConfigDict(frozen=True)

    source: str
    document: str
    section: str
    chunk_id: str
    chunk_index: int = Field(ge=0)
    token_count: int = Field(ge=0)
    page: int | None = None

    def to_chroma(self) -> dict[str, str | int]:
        """Chroma metadata values must be scalars and cannot be ``None``."""
        return self.model_dump(exclude_none=True)


class Chunk(BaseModel):
    """A retrievable unit of text with its provenance."""

    content: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id


class RetrievalMethod(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class RetrievedChunk(BaseModel):
    """A chunk returned by one retrieval stage, with that stage's score."""

    content: str
    metadata: ChunkMetadata
    score: float
    method: RetrievalMethod
    rank: int = Field(ge=1, description="1-based rank within the stage's result list.")
    component_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores from earlier stages, e.g. normalized dense/BM25 scores behind a hybrid score.",
    )

    @computed_field
    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    @computed_field
    @property
    def source(self) -> str:
        return self.metadata.source

    def as_chunk(self) -> Chunk:
        return Chunk(content=self.content, metadata=self.metadata)


class Citation(BaseModel):
    """A numbered source the answer refers to as ``[index]``."""

    index: int = Field(ge=1)
    source: str
    document: str
    section: str
    chunk_id: str
    page: int | None = None
    score: float = Field(description="Reranker relevance score of the cited chunk.")


class DroppedChunk(BaseModel):
    """A reranked chunk that context filtering excluded, and why."""

    chunk_id: str
    reason: str


class RetrievalTrace(BaseModel):
    """Every intermediate result of the retrieval pipeline, for inspection and debugging."""

    dense: list[RetrievedChunk] = Field(default_factory=list)
    bm25: list[RetrievedChunk] = Field(default_factory=list)
    hybrid: list[RetrievedChunk] = Field(default_factory=list)
    reranked: list[RetrievedChunk] = Field(default_factory=list)
    context: list[RetrievedChunk] = Field(
        default_factory=list, description="Final filtered context actually sent to the LLM."
    )
    filtered_out: list[DroppedChunk] = Field(default_factory=list)


class Timings(BaseModel):
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class QueryResponse(BaseModel):
    answer: str
    insufficient_evidence: bool = Field(
        description="True when the system declined to answer for lack of supporting context."
    )
    sources: list[Citation]
    retrieval: RetrievalTrace
    timings: Timings
    warnings: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Models and retrieval parameters in effect, shown by /health and the UI."""

    llm_provider: str
    llm_model: str | None
    embedding_model: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_dense: int
    top_k_bm25: int
    top_k_hybrid: int
    top_k_reranked: int
    hybrid_alpha: float
    min_rerank_score: float
    max_context_tokens: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    detail: str | None = None
    indexed_chunks: int = 0
    llm_reachable: bool = False
    llm_model_available: bool = False
    config: PipelineConfig

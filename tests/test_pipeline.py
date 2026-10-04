"""Full pipeline wiring with fakes for the model-backed parts (embedder, cross-encoder, LLM)."""

import pytest

from app.config.settings import Settings
from app.generation.generator import AnswerGenerator
from app.generation.prompts import REFUSAL_MESSAGE
from app.ingestion.indexer import run_ingestion
from app.models.schemas import RetrievalMethod, RetrievedChunk
from app.pipeline.rag_pipeline import RAGPipeline
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.vector_search import DenseRetriever
from app.storage import load_chunks, open_vectorstore
from tests.fakes import FakeLLM, HashingEmbedder


class FakeReranker:
    """Gives every candidate the same fixed relevance score."""

    def __init__(self, score: float):
        self.score = score

    def rerank(self, query, candidates, top_k):
        return [
            RetrievedChunk(
                content=c.content,
                metadata=c.metadata,
                score=self.score,
                method=RetrievalMethod.RERANKED,
                rank=rank,
            )
            for rank, c in enumerate(candidates[:top_k], start=1)
        ]


def build_pipeline(settings: Settings, rerank_score: float, llm: FakeLLM) -> RAGPipeline:
    embedder = HashingEmbedder()
    run_ingestion(settings, embedder=embedder)
    dense = DenseRetriever(
        embedder, open_vectorstore(settings.chroma_dir, settings.chroma_collection, embedder)
    )
    retriever = HybridRetriever(
        dense, BM25Retriever(load_chunks(settings.chunks_path)), 0.5, 3, 3, 3
    )
    return RAGPipeline(
        settings,
        retriever,
        FakeReranker(rerank_score),
        embedder.count_tokens,
        AnswerGenerator(llm),
    )


def test_answer_path_returns_cited_answer_and_full_trace(kb_settings):
    llm = FakeLLM("Use `worker_init_fn` and a generator [1].")
    response = build_pipeline(kb_settings, 0.9, llm).query("reproducible DataLoader workers?")

    assert not response.insufficient_evidence
    assert response.sources[0].index == 1
    trace = response.retrieval
    assert trace.dense and trace.bm25 and trace.hybrid and trace.reranked
    assert trace.context == trace.reranked  # nothing filtered at score 0.9
    assert response.sources[0].chunk_id == trace.context[0].chunk_id
    timings = response.timings
    # The fake LLM is instant, so generation_ms may round to 0.0; check consistency instead
    # (each timing is rounded to 0.1 ms independently, hence the tolerance).
    parts = timings.retrieval_ms + timings.rerank_ms + timings.generation_ms
    assert timings.total_ms >= parts - 0.5
    assert len(llm.calls) == 1


def test_low_relevance_refuses_without_calling_the_llm(kb_settings):
    llm = FakeLLM("should not be called")
    response = build_pipeline(kb_settings, 0.01, llm).query("capital of France?")

    assert response.insufficient_evidence
    assert response.answer == REFUSAL_MESSAGE
    assert response.sources == [] and response.retrieval.context == []
    assert len(response.retrieval.filtered_out) == len(response.retrieval.reranked)
    assert response.timings.generation_ms == 0
    assert llm.calls == []


def test_query_requires_a_generator(kb_settings):
    pipeline = build_pipeline(kb_settings, 0.9, FakeLLM("x"))
    pipeline.generator = None
    with pytest.raises(RuntimeError):
        pipeline.query("q")
    assert pipeline.retrieve("reproducible workers").trace.reranked  # retrieval still works

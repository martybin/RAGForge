"""End-to-end RAG pipeline.

    question
      -> hybrid retrieval (dense + BM25, fused)       top_k_hybrid candidates
      -> cross-encoder reranking                      top_k_reranked
      -> context filtering (threshold, dedup, budget) final context
      -> grounded generation + citation validation    answer + sources

Each stage's output is kept in a `RetrievalTrace`, so the whole path from
question to answer is inspectable (API response, Streamlit debug mode).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from pydantic import BaseModel

from app.config.settings import Settings
from app.embeddings import Embedder
from app.generation.generator import AnswerGenerator
from app.generation.llm import LLMHealth, create_llm_client
from app.models.schemas import QueryResponse, RetrievalTrace, Timings
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.context_filter import filter_context
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import DenseRetriever
from app.storage import load_chunks, open_vectorstore

logger = logging.getLogger(__name__)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class RetrievalOutcome(BaseModel):
    trace: RetrievalTrace
    retrieval_ms: float
    rerank_ms: float


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: Reranker,
        count_tokens: Callable[[str], int],
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.count_tokens = count_tokens
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings, *, with_generator: bool = True) -> RAGPipeline:
        """Load models and open the index; ``with_generator=False`` skips the LLM."""
        embedder = Embedder(
            settings.embedding_model,
            query_instruction=settings.embedding_query_instruction,
            device=settings.device,
            batch_size=settings.embedding_batch_size,
        )
        store = open_vectorstore(settings.chroma_dir, settings.chroma_collection, embedder)
        dense = DenseRetriever(embedder, store)
        bm25 = BM25Retriever(load_chunks(settings.chunks_path))
        retriever = HybridRetriever(
            dense,
            bm25,
            alpha=settings.hybrid_alpha,
            top_k_dense=settings.top_k_dense,
            top_k_bm25=settings.top_k_bm25,
            top_k_hybrid=settings.top_k_hybrid,
        )
        reranker = Reranker(settings.reranker_model, device=settings.device)
        generator = AnswerGenerator(create_llm_client(settings)) if with_generator else None
        logger.info(
            "pipeline_ready", extra={"indexed_chunks": len(bm25), "generator": with_generator}
        )
        return cls(settings, retriever, reranker, embedder.count_tokens, generator)

    @property
    def indexed_chunks(self) -> int:
        return len(self.retriever.bm25)

    def llm_health(self) -> LLMHealth:
        if self.generator is None:
            return LLMHealth(reachable=False, model_available=False, detail="generator disabled")
        return self.generator.llm.health()

    def retrieve(self, question: str) -> RetrievalOutcome:
        """Run every retrieval stage (no LLM) and return the full trace."""
        started = time.perf_counter()
        hybrid = self.retriever.search(question)
        retrieval_ms = _ms_since(started)

        started = time.perf_counter()
        reranked = self.reranker.rerank(question, hybrid.hybrid, self.settings.top_k_reranked)
        rerank_ms = _ms_since(started)

        filtered = filter_context(
            reranked,
            min_score=self.settings.min_rerank_score,
            max_tokens=self.settings.max_context_tokens,
            count_tokens=self.count_tokens,
        )
        trace = RetrievalTrace(
            dense=hybrid.dense,
            bm25=hybrid.bm25,
            hybrid=hybrid.hybrid,
            reranked=reranked,
            context=filtered.kept,
            filtered_out=filtered.dropped,
        )
        return RetrievalOutcome(trace=trace, retrieval_ms=retrieval_ms, rerank_ms=rerank_ms)

    def query(self, question: str) -> QueryResponse:
        """Answer from the knowledge base, with citations and the full retrieval trace."""
        if self.generator is None:
            raise RuntimeError("Pipeline was built without a generator; use retrieve() instead.")
        started = time.perf_counter()
        outcome = self.retrieve(question)

        generation_started = time.perf_counter()
        result = self.generator.generate(question, outcome.trace.context)
        generation_ms = _ms_since(generation_started) if result.llm_called else 0.0

        timings = Timings(
            retrieval_ms=outcome.retrieval_ms,
            rerank_ms=outcome.rerank_ms,
            generation_ms=generation_ms,
            total_ms=_ms_since(started),
        )
        logger.info(
            "query_completed",
            extra={
                "question": question[:200],
                "hybrid_candidates": len(outcome.trace.hybrid),
                "context_chunks": len(outcome.trace.context),
                "cited": len(result.citations),
                "insufficient_evidence": result.insufficient_evidence,
                "warnings": len(result.warnings),
                **timings.model_dump(),
            },
        )
        return QueryResponse(
            answer=result.answer,
            insufficient_evidence=result.insufficient_evidence,
            sources=result.citations,
            retrieval=outcome.trace,
            timings=timings,
            warnings=result.warnings,
        )

"""Cross-encoder reranking with a BGE reranker (LangChain `HuggingFaceCrossEncoder`).

Bi-encoders (dense retrieval) embed query and passage independently, which is
fast but coarse. A cross-encoder reads the query and the passage *together*
through full attention and outputs one relevance score, which is far more
accurate but too slow to run over the whole corpus. Hence the classic
two-stage design: cheap hybrid retrieval of N candidates, then rerank and keep K.

For single-label models the cross-encoder applies a sigmoid, so scores lie in
(0, 1) and one ``MIN_RERANK_SCORE`` threshold means the same for every query.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.models.schemas import RetrievalMethod, RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512) -> None:
        started = time.perf_counter()
        self.model_name = model_name
        model_kwargs = {"max_length": max_length}
        if device:
            model_kwargs["device"] = device
        self._model = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
        logger.info(
            "reranker_loaded",
            extra={
                "model": model_name,
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance probabilities in (0, 1) for (query, passage) pairs."""
        if not passages:
            return []
        return [float(x) for x in self._model.score([(query, p) for p in passages])]

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Score all candidates with the cross-encoder and keep the best ``top_k``."""
        started = time.perf_counter()
        scores = self.score(query, [c.content for c in candidates])
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: (-pair[1], pair[0].chunk_id)
        )
        results = [
            RetrievedChunk(
                content=candidate.content,
                metadata=candidate.metadata,
                score=score,
                method=RetrievalMethod.RERANKED,
                rank=rank,
                component_scores={
                    **candidate.component_scores,
                    candidate.method.value: candidate.score,
                    f"{candidate.method.value}_rank": float(candidate.rank),
                },
            )
            for rank, (candidate, score) in enumerate(ranked[:top_k], start=1)
        ]
        logger.info(
            "rerank",
            extra={
                "candidates": len(candidates),
                "kept": len(results),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return results

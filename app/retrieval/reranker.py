"""Cross-encoder reranking with a BGE reranker.

Bi-encoders (dense retrieval) embed query and passage independently, which is
fast but coarse. A cross-encoder reads the query and the passage *together*
through full attention and outputs one relevance logit, which is far more
accurate but too slow to run over the whole corpus. Hence the classic
two-stage design: cheap hybrid retrieval of N candidates, then rerank and keep K.

Scores are passed through a sigmoid, so they lie in (0, 1) and a single
``MIN_RERANK_SCORE`` threshold means the same thing for every query.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from app.models.schemas import RetrievalMethod, RetrievedChunk

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        started = time.perf_counter()
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device, max_length=max_length)
        logger.info(
            "reranker_loaded",
            extra={
                "model": model_name,
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Raw relevance logits for (query, passage) pairs."""
        if not passages:
            return []
        logits = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            activation_fn=_identity,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(x) for x in logits]

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Score all candidates with the cross-encoder and keep the best ``top_k``."""
        started = time.perf_counter()
        logits = self.score(query, [c.content for c in candidates])
        ranked = sorted(
            zip(candidates, logits, strict=True), key=lambda pair: (-pair[1], pair[0].chunk_id)
        )
        results = [
            RetrievedChunk(
                content=candidate.content,
                metadata=candidate.metadata,
                score=sigmoid(logit),
                method=RetrievalMethod.RERANKED,
                rank=rank,
                component_scores={
                    **candidate.component_scores,
                    candidate.method.value: candidate.score,
                    f"{candidate.method.value}_rank": float(candidate.rank),
                    "rerank_logit": logit,
                },
            )
            for rank, (candidate, logit) in enumerate(ranked[:top_k], start=1)
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


def _identity(x):
    """Return logits unchanged (the sigmoid is applied explicitly in `rerank`)."""
    return x

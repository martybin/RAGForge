"""Embedding model wrapper shared by ingestion and query-time retrieval.

Keeping a single class for both sides guarantees documents and queries are
embedded with the same model and the same normalization; mixing these up is a
classic silent failure in RAG systems.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Sentence-Transformers bi-encoder producing L2-normalized embeddings.

    BGE models are trained asymmetrically: short queries are prefixed with an
    instruction, passages are embedded as-is. Because vectors are normalized,
    cosine similarity equals the dot product.
    """

    def __init__(
        self,
        model_name: str,
        query_instruction: str = "",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        started = time.perf_counter()
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(
            "embedding_model_loaded",
            extra={
                "model": model_name,
                "device": str(self._model.device),
                "dim": self.dimension,
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    @property
    def dimension(self) -> int:
        return int(self._model.get_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        return int(self._model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        """Number of model tokens in ``text`` (without [CLS]/[SEP]).

        Uses the Rust "fast" tokenizer directly when available: it is quicker for the
        chunker's many calls and does not warn about texts longer than the model window
        (counting long sections before splitting them is expected).
        """
        backend = getattr(self._model.tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return len(backend.encode(text, add_special_tokens=False).ids)
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(list(texts), show_progress=show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([self.query_instruction + query])[0]

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

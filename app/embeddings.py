"""LangChain embeddings shared by ingestion and query-time retrieval.

`HuggingFaceEmbeddings` (langchain-huggingface) wraps a Sentence-Transformers
bi-encoder. Vectors are L2-normalized, so cosine similarity equals the dot
product. BGE is asymmetric: queries get an instruction prefix, passages do not.
One instance serves both sides so documents and queries can never diverge.
"""

from __future__ import annotations

import logging
import time

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class Embedder(HuggingFaceEmbeddings):
    """`HuggingFaceEmbeddings` plus token counting for token-aware chunking."""

    def __init__(
        self,
        model_name: str,
        query_instruction: str = "",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        started = time.perf_counter()
        super().__init__(
            model_name=model_name,
            model_kwargs={"device": device} if device else {},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
            query_encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": batch_size,
                "prompt": query_instruction,
            }
            if query_instruction
            else {"normalize_embeddings": True},
        )
        logger.info(
            "embedding_model_loaded",
            extra={
                "model": model_name,
                "device": str(self._client.device),
                "dim": self.dimension,
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    @property
    def dimension(self) -> int:
        return int(self._client.get_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        return int(self._client.max_seq_length)

    def count_tokens(self, text: str) -> int:
        """Number of model tokens in ``text`` (without [CLS]/[SEP]).

        Uses the Rust tokenizer directly: faster for the chunker's many calls and
        silent about texts longer than the model window (counting long sections
        before splitting them is expected).
        """
        tokenizer = self._client.tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return len(backend.encode(text, add_special_tokens=False).ids)
        return len(tokenizer.encode(text, add_special_tokens=False))

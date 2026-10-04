"""Lightweight stand-ins for model-backed components."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9_]+")


class HashingEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar unit vectors.

    Implements the subset of `app.embeddings.Embedder` used by ingestion and retrieval.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.model_name = "hashing-test-embedder"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension)
        for word in _WORD_RE.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query)

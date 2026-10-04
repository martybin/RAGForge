"""Lightweight stand-ins for model-backed components."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

_WORD_RE = re.compile(r"[a-z0-9_]+")


class HashingEmbedder(Embeddings):
    """Deterministic bag-of-words LangChain embeddings: shared words give similar unit vectors."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.model_name = "hashing-test-embedder"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension)
        for word in _WORD_RE.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    """Stands in for `ChatLLM`: records the rendered prompt messages, returns a canned reply."""

    provider = "fake"
    model_name = "fake-model"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []  # (system, user) per call

    @property
    def chain(self):
        def respond(prompt_value) -> str:
            system, user = prompt_value.to_messages()
            self.calls.append((system.content, user.content))
            return self.reply

        return RunnableLambda(respond)

    def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply

    def health(self):
        raise NotImplementedError

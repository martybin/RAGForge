"""Evaluation metrics.

Retrieval metrics are computed at *source* level: a retrieved chunk is relevant
if it comes from one of the question's ``relevant_sources``. This is robust to
re-chunking (chunk ids change when chunk size changes; file paths do not).

Generation metrics avoid using a small local LLM as a judge where a cheaper,
deterministic model does the job better:

* Faithfulness  - NLI cross-encoder: is each answer sentence entailed by a context chunk?
* Answer relevance - RAGAS-style: the LLM writes the question the answer responds to,
  scored by embedding cosine similarity with the real question.
* Context relevance - share of final-context chunks coming from a relevant source.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from statistics import fmean

_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*(\"'])")
_INFERENCE_PREFIX_RE = re.compile(r"^\s*inference:\s*", re.IGNORECASE)


# --- Retrieval ------------------------------------------------------------------------


def recall_at_k(retrieved_sources: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant sources that appear among the top-k retrieved chunks."""
    if not relevant:
        raise ValueError("recall is undefined without relevant sources")
    return len(relevant & set(retrieved_sources[:k])) / len(relevant)


def reciprocal_rank(retrieved_sources: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first chunk from a relevant source (0 if none was retrieved)."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def context_relevance(context_sources: Sequence[str], relevant: set[str]) -> float:
    """Precision of the final LLM context: share of chunks from a relevant source."""
    if not context_sources:
        return 0.0
    return sum(source in relevant for source in context_sources) / len(context_sources)


def mean(values: Sequence[float]) -> float | None:
    return round(fmean(values), 4) if values else None


# --- Generation -------------------------------------------------------------------------


def split_statements(answer: str) -> list[str]:
    """Split an answer into checkable statements: sentences, without citations or code blocks."""
    text = re.sub(r"```.*?```", " ", answer, flags=re.DOTALL)
    statements = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line)  # list markers
        for sentence in _SENTENCE_RE.split(line):
            sentence = _INFERENCE_PREFIX_RE.sub("", _CITATION_RE.sub("", sentence)).strip()
            if len(sentence.split()) >= 4:  # skip fragments like "For example:"
                statements.append(sentence)
    return statements


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (embeddings are usually already unit-normalized)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0

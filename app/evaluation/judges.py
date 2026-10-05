"""Model-backed evaluation judges.

* `NLIFaithfulnessJudge` - an NLI cross-encoder checks whether each answer
  statement is *entailed* by at least one context passage. Deterministic,
  local, and much more reliable than asking a small LLM to grade itself.
* `AnswerRelevanceJudge` - RAGAS-style answer relevance: the LLM writes the
  question the answer appears to respond to; its embedding similarity to the
  real question measures whether the answer addresses what was asked.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from app.embeddings import Embedder
from app.evaluation.metrics import cosine, split_statements, strip_citations
from app.generation.llm import ChatLLM

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


class StatementCheck(BaseModel):
    statement: str
    entailment: float
    supported: bool


class FaithfulnessResult(BaseModel):
    score: float | None  # None when the answer has no checkable statement
    statements: list[StatementCheck]


class NLIFaithfulnessJudge:
    def __init__(
        self, model_name: str = DEFAULT_NLI_MODEL, threshold: float = 0.5, device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model = CrossEncoder(model_name, device=device)
        labels = {label.lower(): int(i) for i, label in self._model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError(f"{model_name} has no 'entailment' label: {labels}")
        self._entailment_index = labels["entailment"]

    def entailment(self, premises: Sequence[str], hypothesis: str) -> list[float]:
        """P(premise entails hypothesis) for each premise."""
        probs = self._model.predict(
            [(premise, hypothesis) for premise in premises],
            apply_softmax=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(probs)[:, self._entailment_index].tolist()

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> FaithfulnessResult:
        """Share of answer statements entailed by at least one context passage."""
        statements = split_statements(answer)
        if not statements or not contexts:
            return FaithfulnessResult(score=None, statements=[])
        checks = []
        for statement in statements:
            best = max(self.entailment(contexts, statement))
            checks.append(
                StatementCheck(
                    statement=statement, entailment=round(best, 4), supported=best >= self.threshold
                )
            )
        score = sum(check.supported for check in checks) / len(checks)
        return FaithfulnessResult(score=round(score, 4), statements=checks)


QUESTION_GENERATION_PROMPT = """Here is an answer from a technical documentation assistant:

{answer}

Write the single question that this answer responds to. Output only the question, nothing else."""


class AnswerRelevanceJudge:
    def __init__(self, llm: ChatLLM, embedder: Embedder) -> None:
        self.llm = llm
        self.embedder = embedder

    def score(self, question: str, answer: str) -> tuple[float, str]:
        """Returns (cosine of real vs. reconstructed question, reconstructed question)."""
        generated = self.llm.chat(
            "You write concise questions.",
            QUESTION_GENERATION_PROMPT.format(answer=strip_citations(answer)),
        ).strip()
        original, reconstructed = self.embedder.embed_documents([question, generated])
        return round(cosine(original, reconstructed), 4), generated

    def similarity(self, text_a: str, text_b: str) -> float:
        """Embedding cosine similarity between two texts (e.g. answer vs. expected answer)."""
        a, b = self.embedder.embed_documents([strip_citations(text_a), strip_citations(text_b)])
        return round(cosine(a, b), 4)

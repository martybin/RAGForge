"""Grounded answer generation.

Hallucination protection happens in three layers:

1. **Evidence gate (here):** if context filtering left no passage, the LLM is
   not called at all and the refusal message is returned. This is the cheapest
   and most reliable refusal there is.
2. **Prompt (`prompts.py`):** context-only answering, mandatory ``[n]``
   citations, an exact refusal sentence and explicit "Inference:" labelling.
3. **Citation validation (`citations.py`):** citations to passages that were
   never shown are removed, uncited answers are flagged as unverified.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from app.generation.citations import resolve_citations
from app.generation.llm import ChatLLM, LLMError
from app.generation.prompts import RAG_PROMPT, REFUSAL_MESSAGE, format_context
from app.models.schemas import Citation, RetrievedChunk


class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool
    warnings: list[str]
    llm_called: bool


class AnswerGenerator:
    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm
        # LCEL: prompt template -> chat model -> string output parser
        self.chain = RAG_PROMPT | llm.chain

    def generate(self, question: str, context: Sequence[RetrievedChunk]) -> GenerationResult:
        if not context:
            return GenerationResult(
                answer=REFUSAL_MESSAGE,
                citations=[],
                insufficient_evidence=True,
                warnings=[
                    "No retrieved passage passed the relevance threshold; the LLM was not called."
                ],
                llm_called=False,
            )
        try:
            raw_answer = self.chain.invoke(
                {"context": format_context(context), "question": question.strip()}
            )
        except Exception as exc:  # provider SDKs raise many different error types
            raise LLMError(f"LLM call failed ({self.llm.provider}): {exc}") from exc
        check = resolve_citations(raw_answer, context)
        return GenerationResult(
            answer=check.answer,
            citations=check.citations,
            insufficient_evidence=check.insufficient_evidence,
            warnings=check.warnings,
            llm_called=True,
        )

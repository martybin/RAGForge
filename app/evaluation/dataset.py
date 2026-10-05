"""Evaluation dataset: questions with expected answers and the sources that contain them.

Unanswerable questions (``answerable: false``) have no relevant sources; the
system is expected to refuse them. They measure hallucination protection.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class EvalExample(BaseModel):
    id: str
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    relevant_sources: list[str]
    answerable: bool = True

    @model_validator(mode="after")
    def _sources_match_answerability(self) -> EvalExample:
        if self.answerable and not self.relevant_sources:
            raise ValueError(f"{self.id}: answerable questions need relevant_sources")
        if not self.answerable and self.relevant_sources:
            raise ValueError(f"{self.id}: unanswerable questions must not list relevant_sources")
        return self


def load_dataset(path: Path) -> list[EvalExample]:
    """Load and validate the evaluation set (ids must be unique)."""
    examples = [EvalExample.model_validate(item) for item in json.loads(path.read_text("utf-8"))]
    ids = [example.id for example in examples]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate example ids: {duplicates}")
    return examples

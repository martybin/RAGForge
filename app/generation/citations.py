"""Post-generation citation handling: the last line of hallucination defence.

The prompt asks the model to cite passages as ``[n]``. Here we verify that
mechanically: every cited number must refer to a passage that was actually in
the context. Citations to non-existent passages are removed and reported, and
answers that make claims without any citation are flagged as unverified.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from app.generation.prompts import REFUSAL_MESSAGE
from app.models.schemas import Citation, RetrievedChunk

# [1], [2, 3], [1,2]: one bracket group may hold several numbers.
_CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SPACE_RE = re.compile(r"\s+")


class CitationCheck(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool
    invalid_citations: list[int]
    warnings: list[str]


def extract_citation_numbers(text: str) -> list[int]:
    """Cited passage numbers in order of first appearance, without duplicates."""
    numbers: list[int] = []
    for group in _CITATION_GROUP_RE.findall(text):
        for number in (int(n) for n in group.split(",")):
            if number not in numbers:
                numbers.append(number)
    return numbers


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text.replace("’", "'").lower()).strip()


_REFUSAL_KEY = _normalize(REFUSAL_MESSAGE).rstrip(".")


def is_refusal(answer: str) -> bool:
    """True when the answer is (or starts with) the mandated refusal sentence."""
    return _REFUSAL_KEY in _normalize(answer)


def _strip_invalid(text: str, valid: set[int]) -> str:
    def keep_valid(match: re.Match[str]) -> str:
        kept = [n.strip() for n in match.group(1).split(",") if int(n) in valid]
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _CITATION_GROUP_RE.sub(keep_valid, text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)  # tidy "word [9]." -> "word."


def to_citation(index: int, chunk: RetrievedChunk) -> Citation:
    meta = chunk.metadata
    return Citation(
        index=index,
        source=meta.source,
        document=meta.document,
        section=meta.section,
        chunk_id=meta.chunk_id,
        page=meta.page,
        score=chunk.score,
    )


def resolve_citations(answer: str, context: Sequence[RetrievedChunk]) -> CitationCheck:
    """Validate ``[n]`` citations against the context that was sent to the model."""
    cited = extract_citation_numbers(answer)
    valid = {n for n in cited if 1 <= n <= len(context)}
    invalid = [n for n in cited if n not in valid]
    cleaned = _strip_invalid(answer, valid).strip() if invalid else answer.strip()
    refusal = is_refusal(cleaned)

    warnings = []
    if invalid:
        warnings.append(f"Removed citations to non-existent passages: {invalid}")
    if not refusal and not valid:
        warnings.append("The answer cites no context passage; treat it as unverified.")

    citations = [] if refusal else [to_citation(n, context[n - 1]) for n in sorted(valid)]
    return CitationCheck(
        answer=cleaned,
        citations=citations,
        insufficient_evidence=refusal,
        invalid_citations=invalid,
        warnings=warnings,
    )


def format_references(citations: Sequence[Citation]) -> str:
    """Plain-text reference list, e.g. ``[1] pytorch/data.md (torch.utils.data > ...)``."""
    return "\n".join(f"[{c.index}] {c.source} ({c.section})" for c in citations)

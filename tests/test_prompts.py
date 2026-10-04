from app.generation.prompts import (
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
    format_passage,
)
from app.models.schemas import ChunkMetadata, RetrievalMethod, RetrievedChunk


def chunk(section: str, body: str, source: str = "pytorch/data.md", page: int | None = None):
    metadata = ChunkMetadata(
        source=source,
        document="torch.utils.data",
        section=section,
        chunk_id=f"{source}#0001",
        chunk_index=1,
        token_count=10,
        page=page,
    )
    return RetrievedChunk(
        content=f"{section}\n\n{body}",
        metadata=metadata,
        score=0.9,
        method=RetrievalMethod.RERANKED,
        rank=1,
    )


def test_system_prompt_encodes_grounding_rules():
    assert "ONLY" in SYSTEM_PROMPT
    assert REFUSAL_MESSAGE in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT  # citation format
    assert "Inference:" in SYSTEM_PROMPT  # inference must be labelled
    assert "prior knowledge" in SYSTEM_PROMPT


def test_passages_are_numbered_with_provenance_and_no_duplicated_header():
    passage = format_passage(2, chunk("torch.utils.data > Memory Pinning", "Set pin_memory=True."))
    assert passage == (
        "[2] Source: pytorch/data.md | Section: torch.utils.data > Memory Pinning\n"
        "Set pin_memory=True."
    )


def test_page_is_included_when_available():
    assert "| Page: 3" in format_passage(1, chunk("Doc", "Text.", source="paper.pdf", page=3))


def test_user_prompt_contains_all_passages_in_order_and_the_question():
    prompt = build_user_prompt(
        "  How do I pin memory?  ",
        [chunk("A > First", "first body"), chunk("B > Second", "second body")],
    )
    assert prompt.index("[1] Source") < prompt.index("[2] Source")
    assert "Question: How do I pin memory?\n" in prompt
    assert "first body" in prompt and "second body" in prompt

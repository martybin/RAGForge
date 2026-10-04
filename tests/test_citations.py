from app.generation.citations import (
    extract_citation_numbers,
    format_references,
    is_refusal,
    resolve_citations,
)
from app.generation.prompts import REFUSAL_MESSAGE
from tests.conftest import make_result

CONTEXT = [
    make_result("pytorch/data.md#0004", 0.91),
    make_result("pytorch/notes/cuda.md#0012", 0.75),
]


def test_extracts_single_grouped_and_repeated_citations_in_order():
    text = "A [2]. B [1, 2]. C [1][2]. D [3,1]."
    assert extract_citation_numbers(text) == [2, 1, 3]


def test_valid_citations_map_to_their_context_chunks():
    check = resolve_citations("Use `pin_memory=True` [1]. It speeds up copies [2].", CONTEXT)
    assert [c.index for c in check.citations] == [1, 2]
    assert check.citations[0].chunk_id == "pytorch/data.md#0004"
    assert check.citations[1].source == "pytorch/notes/cuda.md"
    assert check.citations[0].score == 0.91
    assert check.warnings == [] and not check.insufficient_evidence


def test_citations_to_missing_passages_are_removed_and_reported():
    check = resolve_citations("Fact one [1]. Invented fact [7]. Mixed [2, 9].", CONTEXT)
    assert check.answer == "Fact one [1]. Invented fact. Mixed [2]."
    assert check.invalid_citations == [7, 9]
    assert [c.index for c in check.citations] == [1, 2]
    assert "non-existent" in check.warnings[0]


def test_uncited_answer_is_flagged_as_unverified():
    check = resolve_citations("PyTorch is great.", CONTEXT)
    assert check.citations == []
    assert any("unverified" in w for w in check.warnings)


def test_refusal_is_detected_and_carries_no_sources():
    check = resolve_citations(REFUSAL_MESSAGE, CONTEXT)
    assert check.insufficient_evidence
    assert check.citations == []
    assert check.warnings == []


def test_refusal_detection_tolerates_typographic_apostrophe_and_case():
    assert is_refusal(
        "I couldn’t find enough evidence in the knowledge base to answer this reliably"
    )
    assert not is_refusal("The knowledge base says to call zero_grad [1].")


def test_reference_list_formatting():
    check = resolve_citations("Fact [1].", CONTEXT)
    assert format_references(check.citations) == "[1] pytorch/data.md (pytorch/data.md > section)"

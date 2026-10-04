from app.retrieval.context_filter import filter_context, jaccard, shingles
from tests.conftest import make_result


def word_count(text: str) -> int:
    return len(text.split())


def run(chunks, min_score=0.1, max_tokens=1000):
    return filter_context(
        chunks, min_score=min_score, max_tokens=max_tokens, count_tokens=word_count
    )


def test_low_scoring_chunks_are_dropped_with_reason():
    result = run([make_result("a#1", 0.9), make_result("b#1", 0.05)])
    assert [c.chunk_id for c in result.kept] == ["a#1"]
    assert result.dropped[0].chunk_id == "b#1"
    assert "below threshold" in result.dropped[0].reason


def test_exact_and_near_duplicates_keep_the_higher_ranked_copy():
    text = "pin memory speeds up host to device copies when using a DataLoader with CUDA"
    result = run(
        [
            make_result("a#1", 0.9, content=text),
            make_result("a#1", 0.8, content=text),
            make_result("b#7", 0.7, content=text + " tensors"),
            make_result("c#2", 0.6, content="completely different text about optimizers"),
        ]
    )
    assert [c.chunk_id for c in result.kept] == ["a#1", "c#2"]
    assert {d.reason for d in result.dropped} == {"near-duplicate of a#1"}


def test_token_budget_skips_chunks_that_do_not_fit_but_keeps_later_small_ones():
    result = run(
        [
            make_result("a#1", 0.9, content="one two three four"),
            make_result("b#1", 0.8, content="w " * 10),
            make_result("c#1", 0.7, content="five six"),
        ],
        max_tokens=7,
    )
    assert [c.chunk_id for c in result.kept] == ["a#1", "c#1"]
    assert "budget" in result.dropped[0].reason


def test_output_is_deterministic_and_preserves_metadata_and_order():
    chunks = [
        make_result(f"doc{i}.md#{i}", 1 - i / 10, content=f"unique text number {i} " * 3)
        for i in range(5)
    ]
    first, second = run(chunks), run(chunks)
    assert first == second
    assert [c.chunk_id for c in first.kept] == [c.chunk_id for c in chunks]
    assert first.kept[0].metadata == chunks[0].metadata


def test_jaccard_of_shingles():
    a, b = shingles("a b c d"), shingles("a b c e")
    assert jaccard(a, a) == 1.0
    assert 0 < jaccard(a, b) < 1

import json

import pytest
from pydantic import ValidationError

from app.config.settings import PROJECT_ROOT
from app.evaluation.dataset import EvalExample, load_dataset
from app.evaluation.metrics import (
    context_relevance,
    cosine,
    mean,
    recall_at_k,
    reciprocal_rank,
    split_statements,
    strip_citations,
)

RANKED = ["a.md", "b.md", "a.md", "c.md", "d.md"]


def test_recall_at_k_counts_distinct_relevant_sources():
    assert recall_at_k(RANKED, {"c.md"}, 3) == 0.0
    assert recall_at_k(RANKED, {"c.md"}, 4) == 1.0
    assert recall_at_k(RANKED, {"a.md", "d.md"}, 3) == 0.5
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 3)


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank(RANKED, {"b.md", "c.md"}) == 0.5
    assert reciprocal_rank(RANKED, {"z.md"}) == 0.0


def test_context_relevance_is_precision_of_final_context():
    assert context_relevance(["a.md", "a.md", "b.md", "c.md"], {"a.md"}) == 0.5
    assert context_relevance([], {"a.md"}) == 0.0


def test_mean_rounds_and_handles_empty():
    assert mean([1, 0, 0]) == 0.3333
    assert mean([]) is None


def test_statements_drop_citations_code_list_markers_and_fragments():
    answer = (
        "Set `pin_memory=True` in the DataLoader [1]. Inference: this speeds up copies [1][2].\n"
        "- Define a pin_memory method on custom batches [1].\n"
        "For example:\n```python\nloader = DataLoader(ds, pin_memory=True)\n```"
    )
    assert split_statements(answer) == [
        "Set `pin_memory=True` in the DataLoader.",
        "this speeds up copies.",
        "Define a pin_memory method on custom batches.",
    ]


def test_strip_citations_and_cosine():
    assert strip_citations("Use x [1, 2].") == "Use x."
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_unanswerable_examples_must_not_have_sources():
    with pytest.raises(ValidationError):
        EvalExample(
            id="x", question="q", expected_answer="a", relevant_sources=["a.md"], answerable=False
        )
    with pytest.raises(ValidationError):
        EvalExample(id="x", question="q", expected_answer="a", relevant_sources=[])


def test_duplicate_ids_are_rejected(tmp_path):
    item = {"id": "q1", "question": "q", "expected_answer": "a", "relevant_sources": ["a.md"]}
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([item, item]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)


def test_shipped_dataset_is_valid_and_references_known_sources():
    examples = load_dataset(PROJECT_ROOT / "evaluation" / "questions.json")
    from scripts.download_docs import DOC_PATHS

    known = {f"pytorch/{path}" for path in DOC_PATHS}
    assert len(examples) >= 30
    assert any(not e.answerable for e in examples)
    for example in examples:
        assert set(example.relevant_sources) <= known, example.id

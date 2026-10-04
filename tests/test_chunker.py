import re

import pytest

from app.ingestion.chunker import MarkdownChunker, split_sections
from app.models.schemas import Document


def word_count(text: str) -> int:
    """Whitespace 'tokenizer' so chunking tests need no model download."""
    return len(text.split())


MARKDOWN = """# Reproducibility

Intro paragraph about randomness.

## Controlling sources of randomness

### PyTorch random number generator

Use torch.manual_seed to seed the RNG for all devices.

```python
import torch
# Not a heading: a comment inside code
torch.manual_seed(0)
```

## DataLoader

Use worker_init_fn and a generator to preserve reproducibility.
"""


def make_document(content: str = MARKDOWN, **overrides) -> Document:
    fields = {
        "content": content,
        "source": "pytorch/notes/randomness.md",
        "title": "Reproducibility",
    }
    return Document(**(fields | overrides))


def test_sections_follow_heading_hierarchy_and_ignore_code_comments():
    sections = split_sections(MARKDOWN)
    assert [s.path for s in sections] == [
        ("Reproducibility",),
        ("Reproducibility", "Controlling sources of randomness", "PyTorch random number generator"),
        ("Reproducibility", "DataLoader"),
    ]
    assert "# Not a heading" in sections[1].text


def test_heading_only_sections_produce_no_chunks():
    chunks = MarkdownChunker(50, 10, word_count).chunk_document(make_document())
    sections = [c.metadata.section for c in chunks]
    assert "Reproducibility > Controlling sources of randomness" not in sections


def test_metadata_is_preserved_on_every_chunk():
    chunks = MarkdownChunker(50, 10, word_count).chunk_document(make_document())
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        meta = chunk.metadata
        assert meta.source == "pytorch/notes/randomness.md"
        assert meta.document == "Reproducibility"
        assert meta.chunk_id == f"pytorch/notes/randomness.md#{meta.chunk_index:04d}"
        assert meta.page is None
        assert meta.token_count == word_count(chunk.content)


def test_chunks_start_with_their_section_path():
    chunks = MarkdownChunker(50, 10, word_count).chunk_document(make_document())
    dataloader = next(c for c in chunks if c.metadata.section.endswith("DataLoader"))
    assert dataloader.content.startswith("Reproducibility > DataLoader\n\n")


def test_long_sections_are_split_within_budget_with_overlap():
    sentences = " ".join(f"Sentence number {i} explains a detail." for i in range(60))
    document = make_document(f"# Big\n\n{sentences}\n", title="Big")
    chunker = MarkdownChunker(chunk_size=40, chunk_overlap=10, count_tokens=word_count)
    chunks = chunker.chunk_document(document)

    assert len(chunks) > 5
    assert all(c.metadata.token_count <= 40 for c in chunks)
    first_ids = re.findall(r"number (\d+)", chunks[0].content)
    second_ids = re.findall(r"number (\d+)", chunks[1].content)
    assert first_ids[-1] in second_ids  # the last sentence is repeated as overlap


def test_code_indentation_survives_chunking():
    chunks = MarkdownChunker(200, 20, word_count).chunk_document(make_document())
    code_chunk = next(c for c in chunks if "torch.manual_seed(0)" in c.content)
    assert "```python\nimport torch\n" in code_chunk.content


def test_document_without_headings_uses_title_as_section():
    chunks = MarkdownChunker(50, 10, word_count).chunk_document(
        make_document("Just text.", source="notes.txt", title="Notes")
    )
    assert chunks[0].metadata.section == "Notes"


def test_paginated_documents_get_page_in_metadata_and_unique_ids():
    chunker = MarkdownChunker(50, 10, word_count)
    page_1 = chunker.chunk_document(make_document("Page one text.", source="a.pdf", page=1))
    page_2 = chunker.chunk_document(make_document("Page two text.", source="a.pdf", page=2))
    assert page_1[0].metadata.page == 1
    assert page_1[0].chunk_id != page_2[0].chunk_id


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        MarkdownChunker(chunk_size=10, chunk_overlap=10, count_tokens=word_count)

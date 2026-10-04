import pytest

from app.ingestion.loaders import load_documents


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "data_loading.md").write_text(
        "```python\n# not the title\n```\n\n"
        "# Data Loading\n\nUse {class}`~torch.utils.data.DataLoader`.\n",
        encoding="utf-8",
    )
    (tmp_path / "release_notes.txt").write_text("Line one.   \r\nLine two.\r\n", encoding="utf-8")
    (tmp_path / "diagram.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("# Hidden\n", encoding="utf-8")
    return tmp_path


def test_loads_supported_files_with_relative_posix_sources(raw_dir):
    documents = load_documents(raw_dir)
    assert [doc.source for doc in documents] == ["guides/data_loading.md", "release_notes.txt"]


def test_markdown_title_comes_from_first_h1_outside_code(raw_dir):
    markdown = load_documents(raw_dir)[0]
    assert markdown.title == "Data Loading"


def test_markdown_is_cleaned_at_load_time(raw_dir):
    markdown = load_documents(raw_dir)[0]
    assert "`torch.utils.data.DataLoader`" in markdown.content
    assert "{class}" not in markdown.content


def test_text_files_are_normalized_and_titled_from_filename(raw_dir):
    text = load_documents(raw_dir)[1]
    assert text.content == "Line one.\nLine two.\n"
    assert text.title == "Release notes"
    assert text.page is None


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")

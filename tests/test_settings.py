import logging

import pytest
from pydantic import ValidationError

from app.config.settings import PROJECT_ROOT, Settings


def make_settings(**overrides) -> Settings:
    """Build settings without reading the developer's local .env file."""
    return Settings(_env_file=None, **overrides)


def test_defaults_are_consistent():
    settings = make_settings()
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.top_k_reranked <= settings.top_k_hybrid
    assert 0.0 <= settings.hybrid_alpha <= 1.0


def test_relative_paths_resolve_against_project_root():
    settings = make_settings(raw_data_dir="data/raw")
    assert settings.raw_data_dir == PROJECT_ROOT / "data" / "raw"
    assert settings.chroma_dir == PROJECT_ROOT / "data" / "processed" / "chroma"


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("HYBRID_ALPHA", "0.3")
    monkeypatch.setenv("TOP_K_RERANKED", "3")
    settings = make_settings()
    assert settings.hybrid_alpha == 0.3
    assert settings.top_k_reranked == 3


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_reranked_k_cannot_exceed_hybrid_k():
    with pytest.raises(ValidationError, match="TOP_K_RERANKED"):
        make_settings(top_k_hybrid=5, top_k_reranked=10)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_must_be_a_weight(alpha):
    with pytest.raises(ValidationError):
        make_settings(hybrid_alpha=alpha)


def test_openai_compatible_provider_requires_base_url():
    with pytest.raises(ValidationError, match="OPENAI_BASE_URL"):
        make_settings(llm_provider="openai_compatible")


def test_api_key_is_not_exposed_in_repr():
    settings = make_settings(
        llm_provider="openai_compatible", openai_base_url="http://x/v1", openai_api_key="sk-secret"
    )
    assert "sk-secret" not in repr(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-secret"


def test_oversized_chunks_trigger_truncation_warning(caplog):
    with caplog.at_level(logging.WARNING):
        make_settings(chunk_size=700, chunk_overlap=100)
    assert "truncated" in caplog.text

"""Application configuration loaded from environment variables (and `.env`).

All tunable parameters of the pipeline live here so that every experiment
(chunk size, hybrid alpha, top-k values, ...) is reproducible from config alone.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Both BGE models used by default are BERT-style encoders with a 512-token window.
# Chunks longer than this are silently truncated at embedding / reranking time.
MODEL_MAX_TOKENS = 512

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Typed, validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM -----------------------------------------------------------------
    llm_provider: Literal["ollama", "openai_compatible"] = "ollama"
    llm_model: str | None = Field(
        default=None, description="Model name, e.g. 'qwen2.5:3b'. Required for generation."
    )
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str | None = None
    openai_api_key: SecretStr | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, gt=0)
    llm_context_window: int = Field(
        default=8192,
        gt=0,
        description="Ollama num_ctx. Its small default would silently truncate long RAG prompts.",
    )
    llm_timeout_seconds: float = Field(default=300.0, gt=0)

    # --- Embedding / reranking models ---------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_instruction: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="Prefix added to queries (not passages); model-specific, BGE's by default.",
    )
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_batch_size: int = Field(default=32, gt=0)
    device: str | None = Field(
        default=None, description="'cpu', 'cuda', 'mps' or unset for auto-detection."
    )

    # --- Storage ---------------------------------------------------------------
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    chroma_collection: str = "ragforge"

    # --- Chunking (sizes are in tokens of the embedding model's tokenizer) ----
    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)

    # --- Retrieval ---------------------------------------------------------------
    top_k_dense: int = Field(default=10, gt=0)
    top_k_bm25: int = Field(default=10, gt=0)
    top_k_hybrid: int = Field(default=20, gt=0)
    top_k_reranked: int = Field(default=5, gt=0)
    hybrid_alpha: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Weight of the dense score in hybrid fusion."
    )

    # --- Context filtering ------------------------------------------------------
    min_rerank_score: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Chunks whose (sigmoid) reranker score is below this are dropped.",
    )
    max_context_tokens: int = Field(default=2000, gt=0)

    # --- API / UI ------------------------------------------------------------------
    api_url: str = "http://localhost:8000"

    # --- Logging ---------------------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    @field_validator("raw_data_dir", "processed_data_dir")
    @classmethod
    def _resolve_relative_to_project(cls, path: Path) -> Path:
        """Make relative paths independent of the current working directory."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k_reranked > self.top_k_hybrid:
            raise ValueError("TOP_K_RERANKED cannot exceed TOP_K_HYBRID")
        if self.llm_provider == "openai_compatible" and not self.openai_base_url:
            raise ValueError("OPENAI_BASE_URL is required when LLM_PROVIDER=openai_compatible")
        if self.chunk_size > MODEL_MAX_TOKENS:
            logger.warning(
                "CHUNK_SIZE=%d exceeds the %d-token window of the default BGE models; "
                "chunk tails will be truncated during embedding and reranking.",
                self.chunk_size,
                MODEL_MAX_TOKENS,
            )
        return self

    @property
    def chroma_dir(self) -> Path:
        return self.processed_data_dir / "chroma"

    @property
    def chunks_path(self) -> Path:
        """JSONL snapshot of all chunks; the BM25 index is rebuilt from it."""
        return self.processed_data_dir / "chunks.jsonl"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings()

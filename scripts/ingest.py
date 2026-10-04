"""Build the retrieval index from the documents in RAW_DATA_DIR.

Pipeline: load -> clean -> chunk (token-aware) -> embed -> Chroma + BM25 snapshot.

Usage:
    python scripts/ingest.py
"""

from __future__ import annotations

from app.config.logging_config import configure_logging
from app.config.settings import get_settings
from app.ingestion.indexer import run_ingestion


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, "text")
    report = run_ingestion(settings)

    print("\nIngestion complete")
    print(f"  documents      : {report.documents}")
    print(f"  chunks         : {report.chunks}")
    tokens = f"mean {report.mean_tokens}, min {report.min_tokens}, max {report.max_tokens}"
    print(f"  tokens / chunk : {tokens}")
    print(f"  chunk config   : size={settings.chunk_size} overlap={settings.chunk_overlap}")
    print(f"  embedding model: {settings.embedding_model}")
    print(f"  elapsed        : {report.elapsed_s}s")


if __name__ == "__main__":
    main()

"""Download the demo knowledge base: a curated subset of the PyTorch documentation.

The files are fetched from the official pytorch/pytorch repository at a pinned
release tag, so the evaluation set (which references these files) stays
reproducible. They are BSD-licensed but are not committed to this repository;
run this script instead.

Usage:
    python scripts/download_docs.py [--tag v2.14.1] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

from app.config.settings import get_settings

PYTORCH_TAG = "v2.14.1"
RAW_URL = "https://raw.githubusercontent.com/pytorch/pytorch/{tag}/docs/source/{path}"

# Prose-heavy pages: conceptual notes and API guides whose text lives in the
# Markdown itself (pages that are mostly autodoc stubs are excluded).
DOC_PATHS = [
    "notes/amp_examples.md",
    "notes/autograd.md",
    "notes/broadcasting.md",
    "notes/cuda.md",
    "notes/ddp.md",
    "notes/extending.md",
    "notes/faq.md",
    "notes/gradcheck.md",
    "notes/large_scale_deployments.md",
    "notes/modules.md",
    "notes/multiprocessing.md",
    "notes/numerical_accuracy.md",
    "notes/randomness.md",
    "notes/serialization.md",
    "amp.md",
    "checkpoint.md",
    "complex_numbers.md",
    "data.md",
    "hub.md",
    "optim.md",
    "tensor_attributes.md",
    "tensor_view.md",
    "torch_cuda_memory.md",
]


def download(tag: str, target_dir: Path, force: bool) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for doc_path in DOC_PATHS:
            destination = target_dir / doc_path
            if destination.exists() and not force:
                print(f"skip     {doc_path} (exists)")
                continue
            try:
                response = client.get(RAW_URL.format(tag=tag, path=doc_path))
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"FAILED   {doc_path}: {exc}", file=sys.stderr)
                failures += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(response.text, encoding="utf-8")
            print(f"saved    {doc_path} ({len(response.content) / 1024:.1f} KiB)")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=PYTORCH_TAG, help="pytorch/pytorch git tag")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    target = get_settings().raw_data_dir / "pytorch"
    failures = download(args.tag, target, args.force)
    print(
        f"\nPyTorch {args.tag} docs in {target} ({len(DOC_PATHS) - failures}/{len(DOC_PATHS)} ok)"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

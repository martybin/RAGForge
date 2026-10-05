"""Evaluate RAGForge on evaluation/questions.json.

Usage:
    python scripts/evaluate.py                    # retrieval + generation (needs the LLM)
    python scripts/evaluate.py --retrieval-only   # no LLM needed
    python scripts/evaluate.py --limit 5          # quick smoke run

Writes every per-question record plus the summary to a JSON report, so all
reported numbers can be traced back to individual answers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from api.main import pipeline_config
from app.config.logging_config import configure_logging
from app.config.settings import PROJECT_ROOT, Settings, get_settings
from app.embeddings import Embedder
from app.evaluation.dataset import load_dataset
from app.evaluation.judges import DEFAULT_NLI_MODEL, AnswerRelevanceJudge, NLIFaithfulnessJudge
from app.evaluation.runner import (
    K_VALUES,
    run_generation,
    run_retrieval,
    summarize_generation,
    summarize_retrieval,
)
from app.pipeline.rag_pipeline import RAGPipeline

STAGE_LABELS = {
    "dense": "Dense (BGE)",
    "bm25": "BM25",
    "hybrid": "Hybrid",
    "reranked": "Hybrid + Rerank",
}


def fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.2f}"


def print_retrieval(summary: dict, settings: Settings) -> None:
    print(f"\nRetrieval Evaluation ({summary['answerable_questions']} answerable questions)")
    header = "".join(f"{f'Recall@{k}':>10}" for k in K_VALUES) + f"{'MRR@5':>8}"
    print(f"  {'Stage':<20}{header}")
    for stage, metrics in summary["stages"].items():
        label = STAGE_LABELS[stage] + (f" (a={settings.hybrid_alpha})" if stage == "hybrid" else "")
        values = "".join(f"{fmt(metrics[f'recall@{k}']):>10}" for k in K_VALUES)
        print(f"  {label:<20}{values}{fmt(metrics['mrr@5']):>8}")
    gate = summary["evidence_gate"]
    print(f"  Context relevance (final context precision): {fmt(summary['context_relevance'])}")
    print(
        f"  Evidence gate: unanswerable blocked before LLM {fmt(gate['unanswerable_blocked_before_llm'])}, "
        f"answerable wrongly blocked {fmt(gate['answerable_blocked_before_llm'])}"
    )
    latency = summary["latency_ms"]
    print(
        f"  Mean latency: retrieval {latency['retrieval_mean']:.0f} ms, rerank {latency['rerank_mean']:.0f} ms"
    )


def print_generation(summary: dict, settings: Settings, nli_model: str) -> None:
    print(f"\nGeneration Evaluation (LLM: {settings.llm_model} via {settings.llm_provider})")
    print(f"  Faithfulness            : {fmt(summary['faithfulness'])}   (NLI judge: {nli_model})")
    print(
        f"  Answer Relevance        : {fmt(summary['answer_relevance'])}   (judge LLM: {settings.llm_model})"
    )
    print(
        f"  Answer Similarity       : {fmt(summary['answer_similarity'])}   (vs. expected answer)"
    )
    print(f"  Answers with citations  : {fmt(summary['answers_with_citations'])}")
    print(f"  Invalid citations       : {summary['answers_with_invalid_citations']} answers")
    print(
        f"  Correct refusal rate    : {fmt(summary['correct_refusal_rate'])}   (unanswerable questions)"
    )
    print(
        f"  False refusal rate      : {fmt(summary['false_refusal_rate'])}   (answerable questions)"
    )
    print(
        f"  Mean generation time    : {fmt(summary['generation_s_mean'])} s over {summary['llm_calls']} LLM calls"
    )


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAGForge retrieval and generation.")
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "questions.json"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "latest.json"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="checkpoint per question under evaluation/results/checkpoints and resume from it",
    )
    parser.add_argument("--retrieval-only", action="store_true", help="skip LLM generation metrics")
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N questions"
    )
    parser.add_argument(
        "--nli-model", default=DEFAULT_NLI_MODEL, help="NLI cross-encoder for faithfulness"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, "text")
    examples = load_dataset(args.dataset)[: args.limit]

    def ckpt(name: str) -> Path | None:
        if not args.resume:
            return None
        return PROJECT_ROOT / "evaluation" / "results" / "checkpoints" / f"{name}.jsonl"

    started = time.perf_counter()

    pipeline = RAGPipeline.from_settings(settings, with_generator=not args.retrieval_only)
    retrieval_records, outcomes = run_retrieval(pipeline, examples, ckpt("retrieval"))
    retrieval_summary = summarize_retrieval(retrieval_records)
    report: dict = {
        "meta": {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "dataset": str(args.dataset.relative_to(PROJECT_ROOT)),
            "questions": len(examples),
            "config": pipeline_config(settings).model_dump(),
        },
        "retrieval": {
            "summary": retrieval_summary,
            "records": [r.model_dump() for r in retrieval_records],
        },
    }

    generation_summary = None
    if not args.retrieval_only:
        health = pipeline.llm_health()
        if not (health.reachable and health.model_available):
            sys.exit(f"LLM not ready: {health.detail}. Start it or use --retrieval-only.")
        faithfulness_judge = NLIFaithfulnessJudge(args.nli_model, device=settings.device)
        relevance_judge = AnswerRelevanceJudge(
            pipeline.generator.llm,
            Embedder(settings.embedding_model, device=settings.device),
        )
        generation_records = run_generation(
            pipeline, examples, outcomes, faithfulness_judge, relevance_judge, ckpt("generation")
        )
        generation_summary = summarize_generation(generation_records)
        report["meta"]["judges"] = {
            "faithfulness": args.nli_model,
            "answer_relevance": settings.llm_model,
        }
        report["generation"] = {
            "summary": generation_summary,
            "records": [r.model_dump() for r in generation_records],
        }

    report["meta"]["duration_s"] = round(time.perf_counter() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_retrieval(retrieval_summary, settings)
    if generation_summary:
        print_generation(generation_summary, settings, args.nli_model)
    print(f"\nFull report: {args.output} ({report['meta']['duration_s']} s)")


if __name__ == "__main__":
    main()

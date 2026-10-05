"""Evaluation runner: retrieval ablation + generation quality + refusal behaviour.

Retrieval: each question is retrieved once; the trace already contains the
dense, BM25, hybrid and reranked lists, so all four are scored from one pass.
Generation reuses that exact context, so both evaluations see the same retrieval.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from app.evaluation.dataset import EvalExample
from app.evaluation.judges import AnswerRelevanceJudge, NLIFaithfulnessJudge, StatementCheck
from app.evaluation.metrics import context_relevance, mean, recall_at_k, reciprocal_rank
from app.pipeline.rag_pipeline import RAGPipeline, RetrievalOutcome

logger = logging.getLogger(__name__)

RETRIEVAL_STAGES = ("dense", "bm25", "hybrid", "reranked")
K_VALUES = (1, 3, 5)


class RetrievalRecord(BaseModel):
    id: str
    answerable: bool
    relevant_sources: list[str]
    ranked_sources: dict[str, list[str]]
    context_sources: list[str]
    top_rerank_score: float | None
    retrieval_ms: float
    rerank_ms: float


class GenerationRecord(BaseModel):
    id: str
    answerable: bool
    answer: str
    refused: bool
    llm_called: bool
    cited: list[int]
    warnings: list[str]
    faithfulness: float | None = None
    statements: list[StatementCheck] = []
    answer_relevance: float | None = None
    reconstructed_question: str | None = None
    answer_similarity: float | None = None
    generation_ms: float


def _load_checkpoint(path: Path | None, model: type) -> dict:
    """Records saved by an earlier (interrupted) run, keyed by example id."""
    if path is None or not path.exists():
        return {}
    rows = (
        model.model_validate_json(line) for line in path.read_text("utf-8").splitlines() if line
    )
    return {row.id: row for row in rows}


def _append_checkpoint(path: Path | None, row: BaseModel) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(row.model_dump_json() + "\n")


class CachedOutcome(RetrievalOutcome):
    id: str


def run_retrieval(
    pipeline: RAGPipeline, examples: Sequence[EvalExample], checkpoint: Path | None = None
) -> tuple[list[RetrievalRecord], dict[str, RetrievalOutcome]]:
    """Retrieve for every example; with ``checkpoint`` an interrupted run resumes where it stopped."""
    records, outcomes = [], {}
    cached = _load_checkpoint(checkpoint, CachedOutcome)
    for i, example in enumerate(examples, start=1):
        if example.id in cached:
            outcome = cached[example.id]
        else:
            outcome = pipeline.retrieve(example.question)
            _append_checkpoint(checkpoint, CachedOutcome(id=example.id, **outcome.model_dump()))
        trace = outcome.trace
        records.append(
            RetrievalRecord(
                id=example.id,
                answerable=example.answerable,
                relevant_sources=example.relevant_sources,
                ranked_sources={s: [r.source for r in getattr(trace, s)] for s in RETRIEVAL_STAGES},
                context_sources=[r.source for r in trace.context],
                top_rerank_score=trace.reranked[0].score if trace.reranked else None,
                retrieval_ms=outcome.retrieval_ms,
                rerank_ms=outcome.rerank_ms,
            )
        )
        outcomes[example.id] = outcome
        logger.info("eval_retrieval", extra={"n": i, "of": len(examples), "id": example.id})
    return records, outcomes


def summarize_retrieval(records: Sequence[RetrievalRecord]) -> dict:
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]
    stages = {}
    for stage in RETRIEVAL_STAGES:
        metrics = {
            f"recall@{k}": mean(
                [
                    recall_at_k(r.ranked_sources[stage], set(r.relevant_sources), k)
                    for r in answerable
                ]
            )
            for k in K_VALUES
        }
        # MRR over the top 5 for every stage, so lists of different length compare fairly.
        metrics["mrr@5"] = mean(
            [
                reciprocal_rank(r.ranked_sources[stage][:5], set(r.relevant_sources))
                for r in answerable
            ]
        )
        stages[stage] = metrics
    return {
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "stages": stages,
        "context_relevance": mean(
            [context_relevance(r.context_sources, set(r.relevant_sources)) for r in answerable]
        ),
        "evidence_gate": {
            "unanswerable_blocked_before_llm": mean(
                [float(not r.context_sources) for r in unanswerable]
            ),
            "answerable_blocked_before_llm": mean(
                [float(not r.context_sources) for r in answerable]
            ),
        },
        "latency_ms": {
            "retrieval_mean": mean([r.retrieval_ms for r in records]),
            "rerank_mean": mean([r.rerank_ms for r in records]),
        },
    }


def run_generation(
    pipeline: RAGPipeline,
    examples: Sequence[EvalExample],
    outcomes: dict[str, RetrievalOutcome],
    faithfulness_judge: NLIFaithfulnessJudge,
    relevance_judge: AnswerRelevanceJudge,
    checkpoint: Path | None = None,
) -> list[GenerationRecord]:
    if pipeline.generator is None:
        raise RuntimeError("Generation evaluation needs a pipeline with a generator")
    records = []
    cached = _load_checkpoint(checkpoint, GenerationRecord)
    for i, example in enumerate(examples, start=1):
        if example.id in cached:
            records.append(cached[example.id])
            continue
        context = outcomes[example.id].trace.context
        started = time.perf_counter()
        result = pipeline.generator.generate(example.question, context)
        generation_ms = round((time.perf_counter() - started) * 1000, 1)

        record = GenerationRecord(
            id=example.id,
            answerable=example.answerable,
            answer=result.answer,
            refused=result.insufficient_evidence,
            llm_called=result.llm_called,
            cited=[c.index for c in result.citations],
            warnings=result.warnings,
            generation_ms=generation_ms,
        )
        if not result.insufficient_evidence:
            faithfulness = faithfulness_judge.faithfulness(
                result.answer, [c.content for c in context]
            )
            record.faithfulness = faithfulness.score
            record.statements = faithfulness.statements
            if example.answerable:
                record.answer_relevance, record.reconstructed_question = relevance_judge.score(
                    example.question, result.answer
                )
                record.answer_similarity = relevance_judge.similarity(
                    result.answer, example.expected_answer
                )
        records.append(record)
        _append_checkpoint(checkpoint, record)
        logger.info(
            "eval_generation",
            extra={"n": i, "of": len(examples), "id": example.id, "refused": record.refused},
        )
    return records


def summarize_generation(records: Sequence[GenerationRecord]) -> dict:
    answered = [r for r in records if not r.refused]
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]
    return {
        "faithfulness": mean([r.faithfulness for r in answered if r.faithfulness is not None]),
        "answer_relevance": mean(
            [r.answer_relevance for r in answered if r.answer_relevance is not None]
        ),
        "answer_similarity": mean(
            [r.answer_similarity for r in answered if r.answer_similarity is not None]
        ),
        "answers_with_citations": mean([float(bool(r.cited)) for r in answered]),
        "answers_with_invalid_citations": sum(
            any("non-existent" in w for w in r.warnings) for r in answered
        ),
        "correct_refusal_rate": mean([float(r.refused) for r in unanswerable]),
        "false_refusal_rate": mean([float(r.refused) for r in answerable]),
        "answered": len(answered),
        "llm_calls": sum(r.llm_called for r in records),
        "generation_s_mean": mean([r.generation_ms / 1000 for r in records if r.llm_called]),
    }

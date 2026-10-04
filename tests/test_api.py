"""API contract tests with a stub pipeline: validation, response shape, error mapping."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from app.config.settings import Settings
from app.generation.llm import LLMError, LLMHealth
from app.models.schemas import Citation, QueryResponse, RetrievalTrace, Timings
from tests.conftest import make_result


class StubPipeline:
    indexed_chunks = 42

    def __init__(self, error: Exception | None = None, llm_ok: bool = True):
        self.error = error
        self.llm_ok = llm_ok
        self.questions: list[str] = []

    def query(self, question: str) -> QueryResponse:
        self.questions.append(question)
        if self.error:
            raise self.error
        chunk = make_result("pytorch/data.md#0004", 0.93)
        return QueryResponse(
            answer="Set `pin_memory=True` [1].",
            insufficient_evidence=False,
            sources=[
                Citation(
                    index=1,
                    source="pytorch/data.md",
                    document="data.md",
                    section="pytorch/data.md > section",
                    chunk_id="pytorch/data.md#0004",
                    score=0.93,
                )
            ],
            retrieval=RetrievalTrace(
                dense=[chunk], bm25=[chunk], hybrid=[chunk], reranked=[chunk], context=[chunk]
            ),
            timings=Timings(retrieval_ms=10, rerank_ms=20, generation_ms=1000, total_ms=1030),
        )

    def llm_health(self) -> LLMHealth:
        if self.llm_ok:
            return LLMHealth(reachable=True, model_available=True)
        return LLMHealth(reachable=False, model_available=False, detail="connection refused")


SETTINGS = Settings(_env_file=None, llm_model="qwen2.5:3b")


def client_for(pipeline) -> TestClient:
    return TestClient(create_app(pipeline=pipeline, settings=SETTINGS))


def test_query_returns_answer_sources_and_retrieval_trace():
    pipeline = StubPipeline()
    response = client_for(pipeline).post("/query", json={"question": "  How do I pin memory?  "})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[1].")
    assert body["sources"][0]["source"] == "pytorch/data.md"
    assert set(body["retrieval"]) >= {"dense", "bm25", "hybrid", "reranked", "context"}
    assert body["retrieval"]["reranked"][0]["chunk_id"] == "pytorch/data.md#0004"
    assert pipeline.questions == ["How do I pin memory?"]  # validated and stripped


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 1001}, {"question": 123}],
)
def test_invalid_requests_are_rejected_with_422(payload):
    pipeline = StubPipeline()
    response = client_for(pipeline).post("/query", json=payload)
    assert response.status_code == 422
    assert pipeline.questions == []


def test_llm_failure_maps_to_502():
    response = client_for(StubPipeline(error=LLMError("Cannot reach LLM"))).post(
        "/query", json={"question": "q?"}
    )
    assert response.status_code == 502
    assert "Cannot reach LLM" in response.json()["detail"]


def test_missing_pipeline_maps_to_503():
    response = client_for(None).post("/query", json={"question": "q?"})
    assert response.status_code == 503


def test_health_reports_status_index_size_and_config():
    body = client_for(StubPipeline()).get("/health").json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == 42
    assert body["config"]["llm_model"] == "qwen2.5:3b"
    assert body["config"]["hybrid_alpha"] == SETTINGS.hybrid_alpha


def test_health_is_degraded_when_llm_is_down():
    body = client_for(StubPipeline(llm_ok=False)).get("/health").json()
    assert body["status"] == "degraded"
    assert body["detail"] == "connection refused"


def test_health_is_unavailable_without_pipeline():
    assert client_for(None).get("/health").json()["status"] == "unavailable"

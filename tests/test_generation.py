"""LangChain generation chain, chat-model factory and health checks: no model server needed."""

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config.settings import Settings
from app.generation.generator import AnswerGenerator
from app.generation.llm import ChatLLM, LLMError, create_chat_model, create_llm_client
from app.generation.prompts import REFUSAL_MESSAGE, SYSTEM_PROMPT
from tests.conftest import make_result
from tests.fakes import FakeLLM

# --- Generator ------------------------------------------------------------------------


def test_empty_context_refuses_without_calling_the_llm():
    llm = FakeLLM("should never be used")
    result = AnswerGenerator(llm).generate("What is X?", [])
    assert result.answer == REFUSAL_MESSAGE
    assert result.insufficient_evidence and not result.llm_called
    assert llm.calls == []


def test_answer_is_generated_from_context_and_citations_resolved():
    llm = FakeLLM("Set `pin_memory=True` [1].")
    context = [make_result("pytorch/data.md#0004", 0.9, "Memory pinning text")]
    result = AnswerGenerator(llm).generate("How do I pin memory?", context)

    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "Memory pinning text" in user and "How do I pin memory?" in user
    assert "[1] Source: pytorch/data.md" in user
    assert result.llm_called and not result.insufficient_evidence
    assert [c.chunk_id for c in result.citations] == ["pytorch/data.md#0004"]


def test_model_refusal_is_propagated():
    result = AnswerGenerator(FakeLLM(REFUSAL_MESSAGE)).generate("Q?", [make_result("a#0", 0.9)])
    assert result.insufficient_evidence and result.citations == []


def test_chain_runs_with_a_real_langchain_chat_model():
    llm = ChatLLM(FakeListChatModel(responses=["Answer [1]."]), "fake", "fake-list", None)
    result = AnswerGenerator(llm).generate("Q?", [make_result("a#0", 0.9)])
    assert result.answer == "Answer [1]." and [c.index for c in result.citations] == [1]


def test_provider_errors_become_llm_errors():
    class Boom(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise ConnectionError("connection refused")

    generator = AnswerGenerator(ChatLLM(Boom(responses=["x"]), "fake", "m", None))
    with pytest.raises(LLMError, match="connection refused"):
        generator.generate("Q?", [make_result("a#0", 0.9)])


# --- Factory --------------------------------------------------------------------------


def test_factory_requires_model_name():
    with pytest.raises(LLMError, match="LLM_MODEL"):
        create_llm_client(Settings(_env_file=None, llm_model=None))


def test_factory_builds_langchain_chat_models_per_provider():
    ollama = create_chat_model(
        Settings(_env_file=None, llm_model="qwen3:8b", llm_think=False, llm_context_window=4096)
    )
    assert isinstance(ollama, ChatOllama)
    assert ollama.model == "qwen3:8b" and ollama.num_ctx == 4096 and ollama.reasoning is False
    other = create_chat_model(
        Settings(
            _env_file=None,
            llm_provider="openai_compatible",
            openai_base_url="http://vllm:8000/v1",
            openai_api_key="sk-test",
            llm_model="m",
        )
    )
    assert isinstance(other, ChatOpenAI) and other.model_name == "m"


# --- Health checks --------------------------------------------------------------------


def patch_get(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: httpx.Client(transport=transport).get(url, **kw)
    )


def test_ollama_health_reports_missing_model(monkeypatch):
    patch_get(monkeypatch, lambda r: httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]}))
    health = create_llm_client(Settings(_env_file=None, llm_model="llama3.2")).health()
    assert health.reachable and not health.model_available
    assert "ollama pull llama3.2" in health.detail


def test_ollama_health_ok_when_model_is_pulled(monkeypatch):
    patch_get(monkeypatch, lambda r: httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]}))
    health = create_llm_client(Settings(_env_file=None, llm_model="qwen2.5:3b")).health()
    assert health.reachable and health.model_available


def test_unreachable_server_is_reported_not_raised(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    patch_get(monkeypatch, refuse)
    health = create_llm_client(Settings(_env_file=None, llm_model="m")).health()
    assert health.reachable is False and "refused" in health.detail

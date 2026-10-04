"""LLM providers as LangChain chat models.

`create_chat_model` returns a LangChain `BaseChatModel` (`ChatOllama` or
`ChatOpenAI`). Adding a provider = one branch in that factory. `ChatLLM` wraps a
chat model with the two things the app needs beyond LangChain: a health check
and a plain ``chat(system, user)`` used by the evaluation judges.
"""

from __future__ import annotations

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config.settings import Settings


class LLMError(RuntimeError):
    """The LLM backend is unreachable, misconfigured or returned an error."""


class LLMHealth(BaseModel):
    reachable: bool
    model_available: bool
    detail: str | None = None


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build the configured LangChain chat model. ``LLM_MODEL`` must be set."""
    if not settings.llm_model:
        raise LLMError("LLM_MODEL is not set. Configure it in .env (see .env.example).")
    if settings.llm_provider == "ollama":
        extra = {} if settings.llm_think is None else {"reasoning": settings.llm_think}
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            num_ctx=settings.llm_context_window,
            client_kwargs={"timeout": settings.llm_timeout_seconds},
            **extra,
        )
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else "none"
    return ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_completion_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )


class ChatLLM:
    """A LangChain chat model plus provider name, health check and ``chat()``."""

    def __init__(
        self,
        model: BaseChatModel,
        provider: str,
        model_name: str,
        health_url: str | None,
        api_key: str | None = None,
    ):
        self.model = model
        self.provider = provider
        self.model_name = model_name
        self._health_url = health_url
        self._api_key = api_key

    @property
    def chain(self):
        """LCEL runnable: messages -> text."""
        return self.model | StrOutputParser()

    def chat(self, system: str, user: str) -> str:
        try:
            return self.chain.invoke([SystemMessage(system), HumanMessage(user)])
        except Exception as exc:  # provider SDKs raise many different error types
            raise LLMError(f"LLM call failed ({self.provider}): {exc}") from exc

    def health(self) -> LLMHealth:
        if not self._health_url:
            return LLMHealth(reachable=True, model_available=True, detail="health check skipped")
        try:
            response = httpx.get(self._health_url, timeout=5, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return LLMHealth(reachable=False, model_available=False, detail=str(exc))
        payload = response.json()
        if self.provider == "ollama":
            names = {m.get("name") for m in payload.get("models", [])}
            wanted = self.model_name if ":" in self.model_name else f"{self.model_name}:latest"
            available = wanted in names
            detail = None if available else f"Model not pulled; run `ollama pull {self.model_name}`"
        else:
            available = self.model_name in {m.get("id") for m in payload.get("data", [])}
            detail = None if available else f"Model '{self.model_name}' not listed by the server"
        return LLMHealth(reachable=True, model_available=available, detail=detail)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}


def create_llm_client(settings: Settings) -> ChatLLM:
    """Chat model + health check for the configured provider."""
    model = create_chat_model(settings)
    if settings.llm_provider == "ollama":
        health_url = settings.ollama_base_url.rstrip("/") + "/api/tags"
        return ChatLLM(model, "ollama", settings.llm_model, health_url)
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    health_url = (settings.openai_base_url or "").rstrip("/") + "/models"
    return ChatLLM(model, "openai_compatible", settings.llm_model, health_url, key)

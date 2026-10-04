"""FastAPI service exposing the RAG pipeline.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /query   answer a question (answer, cited sources, full retrieval trace)
    GET  /health  readiness of index + LLM, and the active configuration
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.config.logging_config import configure_logging
from app.config.settings import Settings, get_settings
from app.generation.llm import LLMError
from app.models.schemas import HealthResponse, PipelineConfig, QueryRequest, QueryResponse
from app.pipeline.rag_pipeline import RAGPipeline
from app.storage import IndexNotFoundError

logger = logging.getLogger(__name__)


def pipeline_config(settings: Settings) -> PipelineConfig:
    return PipelineConfig.model_validate(
        settings.model_dump(include=set(PipelineConfig.model_fields))
    )


def create_app(pipeline: RAGPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Passing ``pipeline`` skips model loading (used by tests)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_format)
        app.state.pipeline = pipeline
        app.state.startup_error = None
        if pipeline is None:
            try:
                app.state.pipeline = RAGPipeline.from_settings(settings)
            except (IndexNotFoundError, LLMError) as exc:
                # Stay up so /health can explain what is wrong instead of crash-looping.
                app.state.startup_error = str(exc)
                logger.error("pipeline_startup_failed", extra={"error": str(exc)})
        yield

    app = FastAPI(
        title="RAGForge",
        description="Technical knowledge assistant with hybrid retrieval, reranking and cited answers.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline
    app.state.startup_error = None

    def get_pipeline(request: Request) -> RAGPipeline:
        current = request.app.state.pipeline
        if current is None:
            detail = request.app.state.startup_error or "Pipeline is not loaded."
            raise HTTPException(status_code=503, detail=detail)
        return current

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, rag: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
        try:
            return rag.query(body.question)
        except LLMError as exc:
            logger.error("llm_error", extra={"error": str(exc)})
            raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}") from exc

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        config = pipeline_config(settings)
        current: RAGPipeline | None = request.app.state.pipeline
        if current is None:
            return HealthResponse(
                status="unavailable", detail=request.app.state.startup_error, config=config
            )
        llm = current.llm_health()
        ready = llm.reachable and llm.model_available
        return HealthResponse(
            status="ok" if ready else "degraded",
            detail=llm.detail,
            indexed_chunks=current.indexed_chunks,
            llm_reachable=llm.reachable,
            llm_model_available=llm.model_available,
            config=config,
        )

    return app


app = create_app()

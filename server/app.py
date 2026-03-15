"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from server.config import APIConfig
from server.middleware import ErrorHandlerMiddleware, RequestLoggingMiddleware
from server.routes.analytics import router as analytics_router
from server.routes.collections import router as collections_router
from server.routes.documents import router as documents_router
from server.routes.evaluations import router as evaluations_router
from server.routes.query import router as query_router
from server.routes.status import router as status_router
from vectorforge.chunking.html import HTMLChunker
from vectorforge.chunking.markdown import MarkdownChunker
from vectorforge.chunking.recursive import RecursiveChunker
from vectorforge.chunking.registry import ChunkerRegistry
from vectorforge.chunking.token import TokenChunker
from vectorforge.chunking.xml import XMLChunker
from vectorforge.config.settings import load_config
from vectorforge.db.engine import AsyncDatabaseEngine
from vectorforge.embedding.registry import EmbeddingProviderRegistry
from vectorforge.ingestion.loaders.base import DocumentLoaderRegistry
from vectorforge.ingestion.loaders.html_loader import HTMLLoader
from vectorforge.ingestion.loaders.markdown_loader import MarkdownLoader
from vectorforge.ingestion.loaders.pdf_loader import PDFLoader
from vectorforge.ingestion.loaders.text_loader import TextLoader
from vectorforge.ingestion.loaders.xml_loader import XMLLoader
from vectorforge.llm.registry import LLMProviderRegistry
from vectorforge.monitoring.health import (
    HealthChecker,
    database_health_probe,
    pgvector_health_probe,
)
from vectorforge.storage.router import StorageRouter
from vectorforge.vectorstore.pgvector import PgVectorStore

logger = logging.getLogger(__name__)


def _build_lifespan(
    api_config: APIConfig,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the lifespan context manager for the FastAPI app.

    Creates the database engine, provider registries, and health checker
    on startup; disposes on shutdown.

    Args:
        api_config: The API configuration.

    Returns:
        An async context manager suitable for FastAPI's lifespan parameter.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        vf_config = load_config()

        # Database
        db_engine = AsyncDatabaseEngine(vf_config.database)
        db_engine.create_engine()
        app.state.db_engine = db_engine
        app.state.vf_config = vf_config
        app.state.api_config = api_config

        # Embedding registry
        embedding_registry = EmbeddingProviderRegistry()
        embedding_registry.auto_discover(vf_config.embedding.default_provider)
        app.state.embedding_registry = embedding_registry

        # LLM registry
        llm_registry = LLMProviderRegistry()
        llm_registry.auto_discover(vf_config.llm.default_provider)
        app.state.llm_registry = llm_registry

        # Health checker
        health_checker = HealthChecker()
        health_checker.register(
            "database",
            lambda: database_health_probe(db_engine),
        )
        health_checker.register(
            "pgvector",
            lambda: pgvector_health_probe(db_engine),
        )
        app.state.health_checker = health_checker

        # --- Shared service-level singletons ---
        vector_store = PgVectorStore(session_factory=db_engine.session_factory)
        app.state.vector_store = vector_store

        loader_registry = DocumentLoaderRegistry(loaders=[
            TextLoader(),
            MarkdownLoader(),
            HTMLLoader(),
            PDFLoader(),
            XMLLoader(),
        ])
        app.state.loader_registry = loader_registry

        chunker_registry = ChunkerRegistry()
        chunker_registry.register(RecursiveChunker())
        chunker_registry.register(TokenChunker())
        chunker_registry.register(MarkdownChunker())
        chunker_registry.register(HTMLChunker())
        chunker_registry.register(XMLChunker())
        app.state.chunker_registry = chunker_registry

        storage_router = StorageRouter(vf_config.storage)
        app.state.storage_router = storage_router

        logger.info(
            "VectorForge API started on %s:%s",
            api_config.host,
            api_config.port,
        )
        yield

        await db_engine.dispose()
        logger.info("VectorForge API shut down")

    return lifespan


def create_app(api_config: APIConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        api_config: Optional API configuration override.

    Returns:
        A fully configured FastAPI application instance.
    """
    cfg = api_config or APIConfig()

    app = FastAPI(
        title="VectorForge API",
        version="0.1.0",
        description="High-performance RAG engine REST API",
        lifespan=_build_lifespan(cfg),
    )

    # --- Middleware (last added = first executed) ---
    app.add_middleware(ErrorHandlerMiddleware)

    if cfg.log_requests:
        app.add_middleware(RequestLoggingMiddleware)

    is_wildcard = cfg.cors_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=not is_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    app.include_router(collections_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")
    app.include_router(query_router, prefix="/api")
    app.include_router(analytics_router, prefix="/api")
    app.include_router(evaluations_router, prefix="/api")
    app.include_router(status_router, prefix="/api")

    return app

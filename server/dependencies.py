"""FastAPI dependency injection — engine, session, auth, registries, services."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from server.config import APIConfig
from vectorforge.chunking.registry import ChunkerRegistry
from vectorforge.db.engine import AsyncDatabaseEngine
from vectorforge.db.repositories.collection_repo import CollectionRepository
from vectorforge.db.repositories.query_log_repo import QueryLogRepository
from vectorforge.embedding.registry import EmbeddingProviderRegistry
from vectorforge.ingestion.loaders.base import DocumentLoaderRegistry
from vectorforge.ingestion.service import IngestionService
from vectorforge.llm.registry import LLMProviderRegistry
from vectorforge.monitoring.health import HealthChecker
from vectorforge.pipeline.context import ContextBuilder
from vectorforge.pipeline.rag import QueryService
from vectorforge.retriever.dense import DenseRetriever
from vectorforge.storage.router import StorageRouter
from vectorforge.vectorstore.base import BaseVectorStore

# ---------------------------------------------------------------------------
# Core dependencies
# ---------------------------------------------------------------------------


def get_db_engine(request: Request) -> AsyncDatabaseEngine:
    """Retrieve the database engine from app state.

    Args:
        request: The incoming HTTP request.

    Returns:
        The AsyncDatabaseEngine attached at startup.
    """
    engine: AsyncDatabaseEngine = request.app.state.db_engine
    return engine


async def get_session(
    engine: Annotated[AsyncDatabaseEngine, Depends(get_db_engine)],
) -> AsyncIterator[AsyncSession]:
    """Yield a managed async session that commits on success.

    Args:
        engine: The injected database engine.

    Yields:
        An AsyncSession scoped to the request.
    """
    async with engine.get_session() as session:
        yield session


def get_api_config(request: Request) -> APIConfig:
    """Retrieve the API configuration from app state.

    Args:
        request: The incoming HTTP request.

    Returns:
        The APIConfig instance.
    """
    config: APIConfig = request.app.state.api_config
    return config


def get_embedding_registry(request: Request) -> EmbeddingProviderRegistry:
    """Retrieve the embedding provider registry from app state.

    Args:
        request: The incoming HTTP request.

    Returns:
        The EmbeddingProviderRegistry.
    """
    registry: EmbeddingProviderRegistry = request.app.state.embedding_registry
    return registry


def get_llm_registry(request: Request) -> LLMProviderRegistry:
    """Retrieve the LLM provider registry from app state.

    Args:
        request: The incoming HTTP request.

    Returns:
        The LLMProviderRegistry.
    """
    registry: LLMProviderRegistry = request.app.state.llm_registry
    return registry


def get_health_checker(request: Request) -> HealthChecker:
    """Retrieve the health checker from app state.

    Args:
        request: The incoming HTTP request.

    Returns:
        The HealthChecker instance.
    """
    checker: HealthChecker = request.app.state.health_checker
    return checker


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def verify_api_key(
    config: Annotated[APIConfig, Depends(get_api_config)],
    x_api_key: str | None = Header(default=None),
) -> str | None:
    """Verify the API key when auth is enabled.

    If ``auth_required`` is ``False`` in config, this is a no-op.

    Args:
        config: The API configuration.
        x_api_key: The API key from the ``X-Api-Key`` request header.

    Returns:
        The validated API key, or ``None`` when auth is disabled.

    Raises:
        HTTPException: 401 if auth is required and the key is missing or invalid.
    """
    if not config.auth_required:
        return None
    if not x_api_key or not hmac.compare_digest(x_api_key, config.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Service-level dependencies
# ---------------------------------------------------------------------------


def get_vector_store(request: Request) -> BaseVectorStore:
    """Retrieve the vector store from app state."""
    store: BaseVectorStore = request.app.state.vector_store
    return store


def get_loader_registry(request: Request) -> DocumentLoaderRegistry:
    """Retrieve the document loader registry from app state."""
    registry: DocumentLoaderRegistry = request.app.state.loader_registry
    return registry


def get_chunker_registry(request: Request) -> ChunkerRegistry:
    """Retrieve the chunker registry from app state."""
    registry: ChunkerRegistry = request.app.state.chunker_registry
    return registry


def get_storage_router(request: Request) -> StorageRouter:
    """Retrieve the storage router from app state."""
    router: StorageRouter = request.app.state.storage_router
    return router


def get_ingestion_service(
    request: Request,
    embedding_registry: Annotated[
        EmbeddingProviderRegistry, Depends(get_embedding_registry)
    ],
    loader_registry: Annotated[DocumentLoaderRegistry, Depends(get_loader_registry)],
    chunker_registry: Annotated[ChunkerRegistry, Depends(get_chunker_registry)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    storage_router: Annotated[StorageRouter, Depends(get_storage_router)],
) -> IngestionService:
    """Build an IngestionService from app-state singletons."""
    vf_config = request.app.state.vf_config
    return IngestionService(
        loader_registry=loader_registry,
        chunker_registry=chunker_registry,
        embedding_provider=embedding_registry.get_default(),
        vector_store=vector_store,
        storage_router=storage_router,
        chunking_config=vf_config.chunking,
    )


def get_query_service(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    embedding_registry: Annotated[
        EmbeddingProviderRegistry, Depends(get_embedding_registry)
    ],
    llm_registry: Annotated[LLMProviderRegistry, Depends(get_llm_registry)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
) -> QueryService:
    """Build a QueryService from app-state singletons + per-request session."""
    collection_repo = CollectionRepository(session)
    query_log_repo = QueryLogRepository(session)
    retriever = DenseRetriever(
        embedding_registry=embedding_registry,
        vector_store=vector_store,
        collection_repo=collection_repo,
    )
    context_builder = ContextBuilder()
    return QueryService(
        retriever=retriever,
        context_builder=context_builder,
        llm_registry=llm_registry,
        query_log_repo=query_log_repo,
    )


# ---------------------------------------------------------------------------
# Convenience type aliases
# ---------------------------------------------------------------------------

DbEngine = Annotated[AsyncDatabaseEngine, Depends(get_db_engine)]
DbSession = Annotated[AsyncSession, Depends(get_session)]
ApiKey = Annotated[str | None, Depends(verify_api_key)]
EmbeddingReg = Annotated[EmbeddingProviderRegistry, Depends(get_embedding_registry)]
LLMReg = Annotated[LLMProviderRegistry, Depends(get_llm_registry)]
HealthCheck = Annotated[HealthChecker, Depends(get_health_checker)]
VectorStoreDep = Annotated[BaseVectorStore, Depends(get_vector_store)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]
QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]

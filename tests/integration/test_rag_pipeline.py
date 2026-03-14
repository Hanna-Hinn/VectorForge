"""Integration tests for the RAG pipeline.

These tests wire real (but mocked-at-boundary) components together
to verify the full query flow without hitting external services.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from vectorforge.llm.base import BaseLLMProvider
from vectorforge.llm.registry import LLMProviderRegistry
from vectorforge.llm.types import LLMRequestConfig, LLMResponse
from vectorforge.models.domain import Chunk, Collection, RetrievedChunk
from vectorforge.pipeline.context import ContextBuilder
from vectorforge.pipeline.rag import QueryService
from vectorforge.pipeline.types import QueryResult
from vectorforge.retriever.dense import DenseRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_collection() -> Collection:
    return Collection(
        id=uuid.uuid4(),
        name="int-test",
        embedding_config={"default_provider": "voyage", "metric": "cosine"},
        created_at=datetime.now(UTC),
    )


def _make_retrieved(text: str, score: float, index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text=text,
            index=index,
            start_char=0,
            end_char=len(text),
            metadata={},
            created_at=datetime.now(UTC),
        ),
        score=score,
        document_source=f"doc_{index}.txt",
    )


class _FakeLLMProvider(BaseLLMProvider):
    """A fake LLM provider for integration testing."""

    def provider_name(self) -> str:
        return "fake"

    def default_model(self) -> str:
        return "fake-model"

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> LLMResponse:
        return LLMResponse(
            content="Integration test answer.",
            model="fake-model",
            provider="fake",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        )

    async def _call_api_stream(  # type: ignore[override]
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ):  # type: ignore[no-untyped-def]
        for token in ["Integration ", "test ", "answer."]:
            yield token

    async def validate_credentials(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRagPipelineIntegration:
    """End-to-end tests wiring real components with mocked boundaries."""

    async def test_full_query_flow(self) -> None:
        """Full pipeline: retrieve → context → generate → result."""
        collection = _make_collection()
        chunks = [
            _make_retrieved("Python is a programming language.", 0.95, 0),
            _make_retrieved("RAG stands for Retrieval-Augmented Generation.", 0.88, 1),
        ]

        # Mock retriever dependencies
        collection_repo = AsyncMock()
        collection_repo.find_by_id = AsyncMock(return_value=collection)

        embedding_provider = AsyncMock()
        embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 1024)

        embedding_registry = MagicMock()
        embedding_registry.get = MagicMock(return_value=embedding_provider)

        vector_store = AsyncMock()
        vector_store.search = AsyncMock(return_value=chunks)

        retriever = DenseRetriever(
            embedding_registry=embedding_registry,
            vector_store=vector_store,
            collection_repo=collection_repo,
        )

        # Real context builder
        context_builder = ContextBuilder()

        # Fake LLM
        llm_registry = LLMProviderRegistry()
        llm_registry.register(_FakeLLMProvider())
        llm_registry.set_default("fake")

        service = QueryService(
            retriever=retriever,
            context_builder=context_builder,
            llm_registry=llm_registry,
        )

        result = await service.query("What is RAG?", collection.id)

        assert isinstance(result, QueryResult)
        assert result.answer == "Integration test answer."
        assert len(result.retrieved_chunks) == 2
        assert result.retrieval_latency_ms > 0
        assert result.generation_latency_ms > 0
        assert result.total_latency_ms > 0
        assert len(result.sources) == 2

    async def test_full_streaming_flow(self) -> None:
        """Streaming pipeline returns tokens."""
        collection = _make_collection()
        chunks = [_make_retrieved("Streaming test data.", 0.9, 0)]

        collection_repo = AsyncMock()
        collection_repo.find_by_id = AsyncMock(return_value=collection)

        embedding_provider = AsyncMock()
        embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 1024)

        embedding_registry = MagicMock()
        embedding_registry.get = MagicMock(return_value=embedding_provider)

        vector_store = AsyncMock()
        vector_store.search = AsyncMock(return_value=chunks)

        retriever = DenseRetriever(
            embedding_registry=embedding_registry,
            vector_store=vector_store,
            collection_repo=collection_repo,
        )

        llm_registry = LLMProviderRegistry()
        llm_registry.register(_FakeLLMProvider())
        llm_registry.set_default("fake")

        service = QueryService(
            retriever=retriever,
            context_builder=ContextBuilder(),
            llm_registry=llm_registry,
        )

        tokens: list[str] = []
        async for token in service.query_stream("q", collection.id):
            tokens.append(token)

        assert tokens == ["Integration ", "test ", "answer."]

    async def test_empty_retrieval(self) -> None:
        """Pipeline handles no results gracefully."""
        collection = _make_collection()

        collection_repo = AsyncMock()
        collection_repo.find_by_id = AsyncMock(return_value=collection)

        embedding_provider = AsyncMock()
        embedding_provider.embed_query = AsyncMock(return_value=[0.1] * 1024)

        embedding_registry = MagicMock()
        embedding_registry.get = MagicMock(return_value=embedding_provider)

        vector_store = AsyncMock()
        vector_store.search = AsyncMock(return_value=[])

        retriever = DenseRetriever(
            embedding_registry=embedding_registry,
            vector_store=vector_store,
            collection_repo=collection_repo,
        )

        llm_registry = LLMProviderRegistry()
        llm_registry.register(_FakeLLMProvider())
        llm_registry.set_default("fake")

        service = QueryService(
            retriever=retriever,
            context_builder=ContextBuilder(),
            llm_registry=llm_registry,
        )

        result = await service.query("Nonexistent topic", collection.id)

        assert "No relevant documents" in result.answer
        assert result.llm_response is None
        assert result.retrieved_chunks == []

"""Unit tests for the RAG pipeline orchestrator."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from vectorforge.llm.types import LLMResponse
from vectorforge.models.domain import Chunk, RetrievedChunk
from vectorforge.pipeline.context import ContextBuilder
from vectorforge.pipeline.rag import _NO_RESULTS_ANSWER, QueryService
from vectorforge.pipeline.types import QueryConfig, QueryResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "chunk text", index: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
        metadata={},
        created_at=datetime.now(UTC),
    )


def _make_retrieved(
    text: str = "chunk text",
    score: float = 0.9,
    index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(text, index),
        score=score,
        document_source="doc.txt",
    )


def _make_llm_response(content: str = "The answer is 42.") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="gpt-4o",
        provider="openai",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )


def _build_query_service(
    chunks: list[RetrievedChunk] | None = None,
    llm_response: LLMResponse | None = None,
) -> tuple[QueryService, AsyncMock, MagicMock, AsyncMock]:
    """Build a QueryService with mocked dependencies."""
    retriever = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=chunks or [])

    context_builder = ContextBuilder()

    llm_provider = MagicMock()
    llm_provider.provider_name.return_value = "openai"
    llm_provider.default_model.return_value = "gpt-4o"
    llm_provider.generate = AsyncMock(
        return_value=llm_response or _make_llm_response(),
    )

    async def _stream(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        yield "The "
        yield "answer."

    llm_provider.generate_stream = _stream

    llm_registry = MagicMock()
    llm_registry.get_default.return_value = llm_provider
    llm_registry.get.return_value = llm_provider

    query_log_repo = AsyncMock()
    query_log_repo.create = AsyncMock()

    service = QueryService(
        retriever=retriever,
        context_builder=context_builder,
        llm_registry=llm_registry,
        query_log_repo=query_log_repo,
    )
    return service, retriever, llm_registry, query_log_repo


# ---------------------------------------------------------------------------
# QueryConfig tests
# ---------------------------------------------------------------------------


class TestQueryConfig:
    """Tests for QueryConfig defaults."""

    def test_defaults(self) -> None:
        config = QueryConfig()
        assert config.top_k == 10
        assert config.min_score == 0.0
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.include_sources is True

    def test_custom(self) -> None:
        config = QueryConfig(top_k=5, min_score=0.5, temperature=0.0)
        assert config.top_k == 5
        assert config.min_score == 0.5


# ---------------------------------------------------------------------------
# QueryResult tests
# ---------------------------------------------------------------------------


class TestQueryResult:
    """Tests for QueryResult."""

    def test_creation(self) -> None:
        result = QueryResult(
            query="test",
            answer="answer",
            retrieval_latency_ms=10.0,
            generation_latency_ms=50.0,
            total_latency_ms=60.0,
        )
        assert result.query == "test"
        assert result.answer == "answer"


# ---------------------------------------------------------------------------
# QueryService tests
# ---------------------------------------------------------------------------


class TestQueryService:
    """Tests for the RAG query service."""

    async def test_query_returns_answer(self) -> None:
        chunks = [_make_retrieved("text A", 0.9, 0), _make_retrieved("text B", 0.8, 1)]
        service, _, _, _ = _build_query_service(chunks=chunks)
        coll_id = uuid.uuid4()

        result = await service.query("What is RAG?", coll_id)

        assert isinstance(result, QueryResult)
        assert result.answer == "The answer is 42."
        assert result.query == "What is RAG?"
        assert len(result.retrieved_chunks) == 2
        assert result.retrieval_latency_ms > 0
        assert result.total_latency_ms > 0

    async def test_query_no_results(self) -> None:
        service, _, _, _ = _build_query_service(chunks=[])
        coll_id = uuid.uuid4()

        result = await service.query("Unknown topic", coll_id)

        assert result.answer == _NO_RESULTS_ANSWER
        assert result.retrieved_chunks == []
        assert result.llm_response is None
        assert result.generation_latency_ms == 0.0

    async def test_query_passes_config(self) -> None:
        chunks = [_make_retrieved()]
        service, retriever, _, _ = _build_query_service(chunks=chunks)
        coll_id = uuid.uuid4()
        config = QueryConfig(top_k=3, min_score=0.5, temperature=0.0)

        await service.query("q", coll_id, config)

        call_kwargs = retriever.retrieve.call_args.kwargs
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["min_score"] == 0.5

    async def test_query_uses_custom_llm_provider(self) -> None:
        chunks = [_make_retrieved()]
        service, _, llm_registry, _ = _build_query_service(chunks=chunks)
        coll_id = uuid.uuid4()
        config = QueryConfig(llm_provider="anthropic")

        await service.query("q", coll_id, config)

        llm_registry.get.assert_called_with("anthropic")

    async def test_query_uses_default_llm_provider(self) -> None:
        chunks = [_make_retrieved()]
        service, _, llm_registry, _ = _build_query_service(chunks=chunks)
        coll_id = uuid.uuid4()

        await service.query("q", coll_id)

        llm_registry.get_default.assert_called_once()

    async def test_query_includes_sources(self) -> None:
        chunks = [_make_retrieved("text", 0.95, 0)]
        service, _, _, _ = _build_query_service(chunks=chunks)
        coll_id = uuid.uuid4()

        result = await service.query("q", coll_id)

        assert len(result.sources) >= 1
        assert result.sources[0].document_source == "doc.txt"

    async def test_query_logs_query(self) -> None:
        chunks = [_make_retrieved()]
        service, _, _, query_log_repo = _build_query_service(chunks=chunks)
        coll_id = uuid.uuid4()

        await service.query("q", coll_id)

        # Give the background task a chance to execute
        import asyncio
        await asyncio.sleep(0.05)

        query_log_repo.create.assert_awaited_once()

    async def test_query_log_failure_swallowed(self) -> None:
        chunks = [_make_retrieved()]
        service, _, _, query_log_repo = _build_query_service(chunks=chunks)
        query_log_repo.create = AsyncMock(side_effect=RuntimeError("DB down"))
        coll_id = uuid.uuid4()

        # Should not raise
        result = await service.query("q", coll_id)
        assert result.answer == "The answer is 42."

        import asyncio
        await asyncio.sleep(0.05)

    async def test_query_stream_yields_tokens(self) -> None:
        chunks = [_make_retrieved()]
        service, _, _, _ = _build_query_service(chunks=chunks)
        coll_id = uuid.uuid4()

        tokens: list[str] = []
        async for token in service.query_stream("q", coll_id):
            tokens.append(token)

        assert len(tokens) >= 1

    async def test_query_stream_no_results(self) -> None:
        service, _, _, _ = _build_query_service(chunks=[])
        coll_id = uuid.uuid4()

        tokens: list[str] = []
        async for token in service.query_stream("q", coll_id):
            tokens.append(token)

        assert tokens == [_NO_RESULTS_ANSWER]

    async def test_query_without_log_repo(self) -> None:
        chunks = [_make_retrieved()]
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=chunks)

        llm_provider = MagicMock()
        llm_provider.provider_name.return_value = "openai"
        llm_provider.default_model.return_value = "gpt-4o"
        llm_provider.generate = AsyncMock(return_value=_make_llm_response())

        llm_registry = MagicMock()
        llm_registry.get_default.return_value = llm_provider

        service = QueryService(
            retriever=retriever,
            context_builder=ContextBuilder(),
            llm_registry=llm_registry,
            query_log_repo=None,
        )

        result = await service.query("q", uuid.uuid4())
        assert result.answer == "The answer is 42."

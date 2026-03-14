"""Unit tests for the retriever module."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from vectorforge.exceptions import NotFoundError
from vectorforge.models.domain import (
    Chunk,
    Collection,
    DistanceMetric,
    RetrievedChunk,
)
from vectorforge.pipeline.query import MAX_QUERY_LENGTH, preprocess_query
from vectorforge.retriever.dense import DenseRetriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_collection(
    embedding_config: dict[str, object] | None = None,
) -> Collection:
    return Collection(
        id=uuid.uuid4(),
        name="test-collection",
        description="",
        embedding_config=embedding_config or {
            "default_provider": "voyage",
            "metric": "cosine",
        },
        created_at=datetime.now(UTC),
    )


def _make_chunk(index: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=f"Chunk text {index}",
        index=index,
        start_char=0,
        end_char=20,
        metadata={},
        created_at=datetime.now(UTC),
    )


def _make_retrieved(score: float = 0.9, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(index),
        score=score,
        document_source="test.txt",
    )


def _build_retriever(
    collection: Collection | None = None,
    search_results: list[RetrievedChunk] | None = None,
    embed_vector: list[float] | None = None,
) -> tuple[DenseRetriever, AsyncMock, AsyncMock, AsyncMock]:
    """Build a DenseRetriever with mocked dependencies."""
    coll = collection or _make_collection()

    collection_repo = AsyncMock()
    collection_repo.find_by_id = AsyncMock(return_value=coll)

    provider = AsyncMock()
    provider.embed_query = AsyncMock(return_value=embed_vector or [0.1] * 1024)

    registry = MagicMock()
    registry.get = MagicMock(return_value=provider)

    vector_store = AsyncMock()
    vector_store.search = AsyncMock(return_value=search_results or [])

    retriever = DenseRetriever(
        embedding_registry=registry,
        vector_store=vector_store,
        collection_repo=collection_repo,
    )
    return retriever, collection_repo, registry, vector_store


# ---------------------------------------------------------------------------
# QueryPreprocessor tests
# ---------------------------------------------------------------------------


class TestPreprocessQuery:
    """Tests for the preprocess_query function."""

    def test_strips_whitespace(self) -> None:
        assert preprocess_query("  hello world  ") == "hello world"

    def test_collapses_multiple_spaces(self) -> None:
        assert preprocess_query("hello   world") == "hello world"

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="Query cannot be empty"):
            preprocess_query("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Query cannot be empty"):
            preprocess_query("   \t\n  ")

    def test_truncates_long_query(self) -> None:
        long_query = "a" * (MAX_QUERY_LENGTH + 500)
        result = preprocess_query(long_query)
        assert len(result) == MAX_QUERY_LENGTH

    def test_normal_query_unchanged(self) -> None:
        assert preprocess_query("What is RAG?") == "What is RAG?"


# ---------------------------------------------------------------------------
# DenseRetriever tests
# ---------------------------------------------------------------------------


class TestDenseRetriever:
    """Tests for the DenseRetriever."""

    async def test_retrieve_returns_chunks(self) -> None:
        chunks = [_make_retrieved(0.9, 0), _make_retrieved(0.8, 1)]
        retriever, _, _, _ = _build_retriever(search_results=chunks)
        coll_id = (await retriever._collection_repo.find_by_id(uuid.uuid4())).id

        results = await retriever.retrieve("test query", coll_id, top_k=5)

        assert len(results) == 2
        assert results[0].score == 0.9

    async def test_retrieve_calls_embed_query(self) -> None:
        retriever, _, registry, _ = _build_retriever(
            search_results=[_make_retrieved()],
        )
        coll_id = (await retriever._collection_repo.find_by_id(uuid.uuid4())).id

        await retriever.retrieve("test query", coll_id)

        provider = registry.get.return_value
        provider.embed_query.assert_awaited_once_with("test query")

    async def test_retrieve_calls_vector_store_search(self) -> None:
        retriever, _, _, vector_store = _build_retriever(
            search_results=[_make_retrieved()],
        )
        coll_id = (await retriever._collection_repo.find_by_id(uuid.uuid4())).id

        await retriever.retrieve("test query", coll_id, top_k=5)

        vector_store.search.assert_awaited_once()
        call_kwargs = vector_store.search.call_args.kwargs
        assert call_kwargs["top_k"] == 5

    async def test_retrieve_filters_by_min_score(self) -> None:
        chunks = [
            _make_retrieved(0.9, 0),
            _make_retrieved(0.3, 1),
            _make_retrieved(0.1, 2),
        ]
        retriever, _, _, _ = _build_retriever(search_results=chunks)
        coll_id = (await retriever._collection_repo.find_by_id(uuid.uuid4())).id

        results = await retriever.retrieve(
            "test query", coll_id, min_score=0.5,
        )

        assert len(results) == 1
        assert results[0].score == 0.9

    async def test_retrieve_not_found_collection(self) -> None:
        retriever, collection_repo, _, _ = _build_retriever()
        collection_repo.find_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError, match="not found"):
            await retriever.retrieve("test", uuid.uuid4())

    async def test_retrieve_empty_query_raises(self) -> None:
        retriever, _, _, _ = _build_retriever()

        with pytest.raises(ValueError, match="empty"):
            await retriever.retrieve("", uuid.uuid4())

    async def test_retrieve_uses_override_provider(self) -> None:
        retriever, _, registry, _ = _build_retriever(
            search_results=[_make_retrieved()],
        )
        coll_id = (await retriever._collection_repo.find_by_id(uuid.uuid4())).id

        await retriever.retrieve(
            "test", coll_id, embedding_provider="cohere",
        )

        registry.get.assert_called_with("cohere")

    async def test_retrieve_with_scores_delegates(self) -> None:
        chunks = [_make_retrieved(0.9), _make_retrieved(0.1)]
        retriever, _, _, _ = _build_retriever(search_results=chunks)
        coll_id = (await retriever._collection_repo.find_by_id(uuid.uuid4())).id

        results = await retriever.retrieve_with_scores("test", coll_id)

        assert len(results) == 2

    async def test_resolve_metric_defaults_to_cosine(self) -> None:
        coll = _make_collection(embedding_config={})
        metric = DenseRetriever._resolve_metric(coll)
        assert metric == DistanceMetric.COSINE

    async def test_resolve_metric_from_config(self) -> None:
        coll = _make_collection(embedding_config={"metric": "l2"})
        metric = DenseRetriever._resolve_metric(coll)
        assert metric == DistanceMetric.L2

    def test_resolve_provider_name_override(self) -> None:
        coll = _make_collection()
        assert DenseRetriever._resolve_provider_name("cohere", coll) == "cohere"

    def test_resolve_provider_name_from_config(self) -> None:
        coll = _make_collection(
            embedding_config={"default_provider": "openai"},
        )
        assert DenseRetriever._resolve_provider_name(None, coll) == "openai"

    def test_resolve_provider_name_fallback(self) -> None:
        coll = _make_collection(embedding_config={})
        assert DenseRetriever._resolve_provider_name(None, coll) == "voyage"

"""Unit tests for the hybrid search module (keyword, fusion, hybrid retriever)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from vectorforge.models.domain import Chunk, RetrievedChunk
from vectorforge.retriever.fusion import RRFScoreFusion
from vectorforge.retriever.hybrid import HybridRetriever
from vectorforge.retriever.keyword import KeywordSearcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(index: int = 0, chunk_id: uuid.UUID | None = None) -> Chunk:
    return Chunk(
        id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=f"Chunk text {index}",
        index=index,
        start_char=0,
        end_char=20,
        metadata={},
        created_at=datetime.now(UTC),
    )


def _make_retrieved(
    score: float = 0.9,
    index: int = 0,
    chunk_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(index, chunk_id=chunk_id),
        score=score,
        document_source="test.txt",
    )


# ---------------------------------------------------------------------------
# RRFScoreFusion tests
# ---------------------------------------------------------------------------


class TestRRFScoreFusion:
    """Tests for the RRF fusion algorithm."""

    def test_fuse_empty_lists(self) -> None:
        fusion = RRFScoreFusion()
        result = fusion.fuse([], [], top_k=5)
        assert result == []

    def test_fuse_dense_only(self) -> None:
        fusion = RRFScoreFusion(k=60)
        dense = [_make_retrieved(0.9, 0), _make_retrieved(0.8, 1)]
        result = fusion.fuse(dense, [], top_k=5)
        assert len(result) == 2
        assert result[0].score > result[1].score

    def test_fuse_keyword_only(self) -> None:
        fusion = RRFScoreFusion(k=60)
        keyword = [_make_retrieved(0.5, 0), _make_retrieved(0.3, 1)]
        result = fusion.fuse([], keyword, top_k=5)
        assert len(result) == 2
        assert result[0].score > result[1].score

    def test_fuse_merges_both_lists(self) -> None:
        fusion = RRFScoreFusion(k=60)
        cid1 = uuid.uuid4()
        cid2 = uuid.uuid4()
        cid3 = uuid.uuid4()

        dense = [_make_retrieved(0.9, 0, cid1), _make_retrieved(0.8, 1, cid2)]
        keyword = [_make_retrieved(0.5, 2, cid3), _make_retrieved(0.3, 0, cid1)]

        result = fusion.fuse(dense, keyword, top_k=10)

        chunk_ids = {str(r.chunk.id) for r in result}
        assert str(cid1) in chunk_ids
        assert str(cid2) in chunk_ids
        assert str(cid3) in chunk_ids
        assert len(result) == 3

    def test_fuse_shared_chunk_gets_boosted(self) -> None:
        """A chunk appearing in both lists should score higher than one in only one."""
        fusion = RRFScoreFusion(k=60)
        shared_id = uuid.uuid4()
        unique_id = uuid.uuid4()

        dense = [_make_retrieved(0.9, 0, shared_id)]
        keyword = [_make_retrieved(0.5, 0, shared_id), _make_retrieved(0.3, 1, unique_id)]

        result = fusion.fuse(dense, keyword, top_k=10)

        scores = {str(r.chunk.id): r.score for r in result}
        assert scores[str(shared_id)] > scores[str(unique_id)]

    def test_fuse_respects_top_k(self) -> None:
        fusion = RRFScoreFusion(k=60)
        dense = [_make_retrieved(0.9, i) for i in range(10)]
        keyword = [_make_retrieved(0.5, i + 10) for i in range(10)]

        result = fusion.fuse(dense, keyword, top_k=5)
        assert len(result) == 5

    def test_fuse_weights_shift_ranking(self) -> None:
        """Higher dense_weight should favor dense results."""
        fusion = RRFScoreFusion(k=1)
        dense_id = uuid.uuid4()
        keyword_id = uuid.uuid4()

        dense = [_make_retrieved(0.9, 0, dense_id)]
        keyword = [_make_retrieved(0.5, 0, keyword_id)]

        # Heavy dense weight
        result_dense = fusion.fuse(
            dense, keyword, dense_weight=0.9, keyword_weight=0.1, top_k=5,
        )
        dense_score = next(
            r.score for r in result_dense if str(r.chunk.id) == str(dense_id)
        )
        keyword_score = next(
            r.score for r in result_dense if str(r.chunk.id) == str(keyword_id)
        )
        assert dense_score > keyword_score

    def test_fuse_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            RRFScoreFusion(k=0)

    def test_fuse_scores_are_positive(self) -> None:
        fusion = RRFScoreFusion(k=60)
        dense = [_make_retrieved(0.9, i) for i in range(5)]
        keyword = [_make_retrieved(0.5, i + 5) for i in range(5)]

        result = fusion.fuse(dense, keyword, top_k=10)
        for r in result:
            assert r.score > 0.0


# ---------------------------------------------------------------------------
# KeywordSearcher tests
# ---------------------------------------------------------------------------


class TestKeywordSearcher:
    """Tests for the KeywordSearcher."""

    async def test_search_empty_query_returns_empty(self) -> None:
        session = AsyncMock()
        searcher = KeywordSearcher(session)
        result = await searcher.search("", uuid.uuid4())
        assert result == []

    async def test_search_whitespace_query_returns_empty(self) -> None:
        session = AsyncMock()
        searcher = KeywordSearcher(session)
        result = await searcher.search("   ", uuid.uuid4())
        assert result == []

    async def test_search_executes_query(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        searcher = KeywordSearcher(session)

        result = await searcher.search("test query", uuid.uuid4())

        session.execute.assert_awaited_once()
        assert result == []

    async def test_search_converts_results(self) -> None:
        chunk_model = MagicMock()
        chunk_model.id = uuid.uuid4()
        chunk_model.document_id = uuid.uuid4()
        chunk_model.content = "sample chunk text"
        chunk_model.chunk_index = 0
        chunk_model.start_char = 0
        chunk_model.end_char = 17
        chunk_model.chunk_metadata = {"key": "value"}
        chunk_model.created_at = datetime.now(UTC)

        row = (chunk_model, 0.75, "doc.txt")
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=MagicMock(all=MagicMock(return_value=[row])),
        )

        searcher = KeywordSearcher(session)
        results = await searcher.search("test", uuid.uuid4())

        assert len(results) == 1
        assert results[0].score == 0.75
        assert results[0].document_source == "doc.txt"
        assert results[0].chunk.text == "sample chunk text"
        assert results[0].chunk.metadata == {"key": "value"}

    async def test_search_handles_none_metadata(self) -> None:
        chunk_model = MagicMock()
        chunk_model.id = uuid.uuid4()
        chunk_model.document_id = uuid.uuid4()
        chunk_model.content = "text"
        chunk_model.chunk_index = 0
        chunk_model.start_char = 0
        chunk_model.end_char = 4
        chunk_model.chunk_metadata = None
        chunk_model.created_at = datetime.now(UTC)

        row = (chunk_model, 0.5, None)
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=MagicMock(all=MagicMock(return_value=[row])),
        )

        searcher = KeywordSearcher(session)
        results = await searcher.search("test", uuid.uuid4())

        assert results[0].chunk.metadata == {}
        assert results[0].document_source == ""


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """Tests for the HybridRetriever."""

    def _build_hybrid(
        self,
        dense_results: list[RetrievedChunk] | None = None,
        keyword_results: list[RetrievedChunk] | None = None,
    ) -> tuple[HybridRetriever, AsyncMock, AsyncMock]:
        dense = AsyncMock()
        dense.retrieve = AsyncMock(return_value=dense_results or [])

        keyword = AsyncMock()
        keyword.search = AsyncMock(return_value=keyword_results or [])

        retriever = HybridRetriever(
            dense_retriever=dense,
            keyword_searcher=keyword,
        )
        return retriever, dense, keyword

    async def test_retrieve_calls_both_sources(self) -> None:
        retriever, dense, keyword = self._build_hybrid()
        coll_id = uuid.uuid4()

        await retriever.retrieve("test query", coll_id, top_k=5)

        dense.retrieve.assert_awaited_once()
        keyword.search.assert_awaited_once()

    async def test_retrieve_passes_collection_id(self) -> None:
        retriever, dense, keyword = self._build_hybrid()
        coll_id = uuid.uuid4()

        await retriever.retrieve("test query", coll_id)

        dense_kwargs = dense.retrieve.call_args.kwargs
        assert dense_kwargs["collection_id"] == coll_id
        keyword_kwargs = keyword.search.call_args.kwargs
        assert keyword_kwargs["collection_id"] == coll_id

    async def test_retrieve_requests_extra_candidates(self) -> None:
        retriever, dense, _keyword = self._build_hybrid()
        coll_id = uuid.uuid4()

        await retriever.retrieve("test query", coll_id, top_k=5)

        dense_kwargs = dense.retrieve.call_args.kwargs
        assert dense_kwargs["top_k"] == 15  # top_k * 3

    async def test_retrieve_returns_fused_results(self) -> None:
        d1 = _make_retrieved(0.9, 0)
        d2 = _make_retrieved(0.8, 1)
        k1 = _make_retrieved(0.5, 2)

        retriever, _, _ = self._build_hybrid(
            dense_results=[d1, d2],
            keyword_results=[k1],
        )
        coll_id = uuid.uuid4()

        results = await retriever.retrieve("test query", coll_id, top_k=10)

        assert len(results) == 3

    async def test_retrieve_respects_top_k(self) -> None:
        dense_results = [_make_retrieved(0.9, i) for i in range(10)]
        keyword_results = [_make_retrieved(0.5, i + 10) for i in range(10)]

        retriever, _, _ = self._build_hybrid(
            dense_results=dense_results,
            keyword_results=keyword_results,
        )
        coll_id = uuid.uuid4()

        results = await retriever.retrieve("test query", coll_id, top_k=5)

        assert len(results) == 5

    async def test_retrieve_filters_by_min_score(self) -> None:
        # With RRF k=60, scores are small: weight / (k + rank + 1)
        # 0.6 / 61 ≈ 0.00984
        d1 = _make_retrieved(0.9, 0)
        retriever, _, _ = self._build_hybrid(dense_results=[d1])
        coll_id = uuid.uuid4()

        results = await retriever.retrieve(
            "test query", coll_id, min_score=1.0,
        )

        assert len(results) == 0

    async def test_retrieve_empty_both_returns_empty(self) -> None:
        retriever, _, _ = self._build_hybrid()
        coll_id = uuid.uuid4()

        results = await retriever.retrieve("test query", coll_id)

        assert results == []

    async def test_retrieve_with_scores_delegates(self) -> None:
        d1 = _make_retrieved(0.9, 0)
        retriever, _, _ = self._build_hybrid(dense_results=[d1])
        coll_id = uuid.uuid4()

        results = await retriever.retrieve_with_scores("test", coll_id)

        assert len(results) == 1

    async def test_custom_weights(self) -> None:
        dense_mock = AsyncMock()
        keyword_mock = AsyncMock()
        dense_mock.retrieve = AsyncMock(return_value=[_make_retrieved(0.9, 0)])
        keyword_mock.search = AsyncMock(return_value=[_make_retrieved(0.5, 1)])

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            keyword_searcher=keyword_mock,
            dense_weight=0.8,
            keyword_weight=0.2,
        )

        results = await retriever.retrieve("test", uuid.uuid4(), top_k=10)
        assert len(results) == 2

    async def test_custom_fusion_is_used(self) -> None:
        fusion = MagicMock(spec=RRFScoreFusion)
        fusion.fuse = MagicMock(return_value=[_make_retrieved(0.5, 0)])

        dense_mock = AsyncMock()
        keyword_mock = AsyncMock()
        dense_mock.retrieve = AsyncMock(return_value=[_make_retrieved(0.9, 0)])
        keyword_mock.search = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            dense_retriever=dense_mock,
            keyword_searcher=keyword_mock,
            fusion=fusion,
        )

        results = await retriever.retrieve("test", uuid.uuid4())

        fusion.fuse.assert_called_once()
        assert len(results) == 1

    async def test_passes_filters_to_dense_only(self) -> None:
        retriever, dense, keyword = self._build_hybrid()
        filters = {"status": "published"}

        await retriever.retrieve("test", uuid.uuid4(), filters=filters)

        dense_kwargs = dense.retrieve.call_args.kwargs
        assert dense_kwargs["filters"] == filters
        # keyword search doesn't receive filters
        keyword_kwargs = keyword.search.call_args.kwargs
        assert "filters" not in keyword_kwargs

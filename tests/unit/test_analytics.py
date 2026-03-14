"""Unit tests for the analytics module (types + service)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from vectorforge.analytics.service import QueryAnalyticsService
from vectorforge.analytics.types import (
    AnalyticsSummary,
    LatencyStats,
    QueryFrequency,
    VolumeDataPoint,
)

# ---------------------------------------------------------------------------
# Type model tests
# ---------------------------------------------------------------------------


class TestAnalyticsTypes:
    """Tests for the analytics Pydantic models."""

    def test_query_frequency(self) -> None:
        qf = QueryFrequency(query_text="hello", count=5)
        assert qf.query_text == "hello"
        assert qf.count == 5

    def test_latency_stats(self) -> None:
        ls = LatencyStats(
            avg_ms=100.0, min_ms=10.0, max_ms=500.0,
            p50_ms=80.0, p95_ms=400.0, sample_count=50,
        )
        assert ls.avg_ms == 100.0
        assert ls.sample_count == 50

    def test_volume_data_point(self) -> None:
        dt = datetime.now(UTC)
        vdp = VolumeDataPoint(date=dt, count=42)
        assert vdp.count == 42

    def test_analytics_summary(self) -> None:
        summary = AnalyticsSummary(
            total_queries=100,
            unique_queries=30,
            latency=None,
            top_queries=[QueryFrequency(query_text="test", count=10)],
            volume=[],
        )
        assert summary.total_queries == 100
        assert summary.unique_queries == 30
        assert summary.latency is None
        assert len(summary.top_queries) == 1

    def test_analytics_summary_with_latency(self) -> None:
        latency = LatencyStats(
            avg_ms=50.0, min_ms=5.0, max_ms=200.0,
            p50_ms=40.0, p95_ms=180.0, sample_count=20,
        )
        summary = AnalyticsSummary(
            total_queries=20,
            unique_queries=10,
            latency=latency,
            top_queries=[],
            volume=[],
        )
        assert summary.latency is not None
        assert summary.latency.p50_ms == 40.0


# ---------------------------------------------------------------------------
# QueryAnalyticsService tests
# ---------------------------------------------------------------------------


def _mock_execute_sequence(*return_values: object) -> AsyncMock:
    """Create a session mock that yields values in sequence from execute()."""
    session = AsyncMock()
    results = []
    for val in return_values:
        mock_result = MagicMock()
        if isinstance(val, list):
            mock_result.all = MagicMock(return_value=val)
        else:
            mock_result.one = MagicMock(return_value=val)
            mock_result.scalar_one = MagicMock(return_value=val)
        results.append(mock_result)
    session.execute = AsyncMock(side_effect=results)
    return session


class TestQueryAnalyticsService:
    """Tests for the QueryAnalyticsService."""

    async def test_get_top_queries(self) -> None:
        row1 = MagicMock(query_text="what is RAG", cnt=15)
        row2 = MagicMock(query_text="explain embeddings", cnt=8)

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all = MagicMock(return_value=[row1, row2])
        session.execute = AsyncMock(return_value=mock_result)

        service = QueryAnalyticsService(session)
        coll_id = uuid.uuid4()

        top = await service.get_top_queries(coll_id, top_n=5)

        assert len(top) == 2
        assert top[0].query_text == "what is RAG"
        assert top[0].count == 15
        assert top[1].count == 8
        session.execute.assert_awaited_once()

    async def test_get_top_queries_with_since(self) -> None:
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all = MagicMock(return_value=[])
        session.execute = AsyncMock(return_value=mock_result)

        service = QueryAnalyticsService(session)
        since = datetime(2024, 1, 1, tzinfo=UTC)

        result = await service.get_top_queries(uuid.uuid4(), since=since)
        assert result == []

    async def test_get_latency_stats_no_data(self) -> None:
        agg_row = MagicMock(cnt=0, avg_ms=None, min_ms=None, max_ms=None)

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.one = MagicMock(return_value=agg_row)
        session.execute = AsyncMock(return_value=mock_result)

        service = QueryAnalyticsService(session)
        stats = await service.get_latency_stats(uuid.uuid4())

        assert stats is None

    async def test_get_latency_stats_with_data(self) -> None:
        row = MagicMock(
            cnt=10, avg_ms=100.0, min_ms=10.0, max_ms=500.0,
            p50_ms=80.0, p95_ms=400.0,
        )
        mock_result = MagicMock()
        mock_result.one = MagicMock(return_value=row)

        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)

        service = QueryAnalyticsService(session)
        stats = await service.get_latency_stats(uuid.uuid4())

        assert stats is not None
        assert stats.avg_ms == 100.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 500.0
        assert stats.p50_ms == 80.0
        assert stats.p95_ms == 400.0
        assert stats.sample_count == 10
        session.execute.assert_awaited_once()

    async def test_get_query_volume(self) -> None:
        day1 = datetime(2024, 6, 1, tzinfo=UTC)
        day2 = datetime(2024, 6, 2, tzinfo=UTC)
        row1 = MagicMock(day=day1, cnt=5)
        row2 = MagicMock(day=day2, cnt=12)

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all = MagicMock(return_value=[row1, row2])
        session.execute = AsyncMock(return_value=mock_result)

        service = QueryAnalyticsService(session)
        volume = await service.get_query_volume(uuid.uuid4())

        assert len(volume) == 2
        assert volume[0].date == day1
        assert volume[0].count == 5
        assert volume[1].count == 12

    async def test_get_query_volume_empty(self) -> None:
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all = MagicMock(return_value=[])
        session.execute = AsyncMock(return_value=mock_result)

        service = QueryAnalyticsService(session)
        volume = await service.get_query_volume(uuid.uuid4())

        assert volume == []

    async def test_get_summary(self) -> None:
        """get_summary aggregates counts, latency, top queries, and volume."""
        coll_id = uuid.uuid4()

        # Mock _get_counts → (total, unique)
        counts_row = MagicMock(total=50, unique=20)
        counts_result = MagicMock()
        counts_result.one = MagicMock(return_value=counts_row)

        # Mock get_latency_stats (single combined query)
        latency_row = MagicMock(
            cnt=50, avg_ms=75.0, min_ms=5.0, max_ms=300.0,
            p50_ms=60.0, p95_ms=250.0,
        )
        latency_result = MagicMock()
        latency_result.one = MagicMock(return_value=latency_row)

        # Mock get_top_queries
        top_row = MagicMock(query_text="explain RAG", cnt=15)
        top_result = MagicMock()
        top_result.all = MagicMock(return_value=[top_row])

        # Mock get_query_volume
        vol_row = MagicMock(day=datetime(2024, 6, 1, tzinfo=UTC), cnt=50)
        vol_result = MagicMock()
        vol_result.all = MagicMock(return_value=[vol_row])

        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=[
                counts_result,    # _get_counts
                latency_result,   # get_latency_stats (combined)
                top_result,       # get_top_queries
                vol_result,       # get_query_volume
            ]
        )

        service = QueryAnalyticsService(session)
        summary = await service.get_summary(coll_id, top_n=5)

        assert summary.total_queries == 50
        assert summary.unique_queries == 20
        assert summary.latency is not None
        assert summary.latency.avg_ms == 75.0
        assert len(summary.top_queries) == 1
        assert summary.top_queries[0].query_text == "explain RAG"
        assert len(summary.volume) == 1

    async def test_get_summary_no_latency(self) -> None:
        """get_summary with no latency data returns latency=None."""
        coll_id = uuid.uuid4()

        counts_row = MagicMock(total=5, unique=3)
        counts_result = MagicMock()
        counts_result.one = MagicMock(return_value=counts_row)

        # Latency agg with cnt=0
        agg_row = MagicMock(cnt=0, avg_ms=None, min_ms=None, max_ms=None)
        agg_result = MagicMock()
        agg_result.one = MagicMock(return_value=agg_row)

        top_result = MagicMock()
        top_result.all = MagicMock(return_value=[])

        vol_result = MagicMock()
        vol_result.all = MagicMock(return_value=[])

        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=[counts_result, agg_result, top_result, vol_result]
        )

        service = QueryAnalyticsService(session)
        summary = await service.get_summary(coll_id)

        assert summary.total_queries == 5
        assert summary.latency is None
        assert summary.top_queries == []
        assert summary.volume == []

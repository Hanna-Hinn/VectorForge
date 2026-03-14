"""Query analytics service.

Provides aggregation queries over the ``query_logs`` table to surface
usage patterns, latency metrics, and query volume trends.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vectorforge.analytics.types import (
    AnalyticsSummary,
    LatencyStats,
    QueryFrequency,
    VolumeDataPoint,
)
from vectorforge.models.db import QueryLogModel

logger = logging.getLogger(__name__)


class QueryAnalyticsService:
    """Computes analytics over query log records.

    All methods operate within the provided session and do not
    commit or close it — the caller manages the session lifecycle.

    Args:
        session: An active async database session.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_summary(
        self,
        collection_id: uuid.UUID,
        since: datetime | None = None,
        top_n: int = 10,
    ) -> AnalyticsSummary:
        """Build a high-level analytics summary for a collection.

        Args:
            collection_id: The collection to analyse.
            since: Optional lower bound on ``created_at``.
            top_n: Number of top queries to include.

        Returns:
            An AnalyticsSummary with counts, latency, top queries, and volume.
        """
        total, unique = await self._get_counts(collection_id, since)
        latency = await self.get_latency_stats(collection_id, since)
        top_queries = await self.get_top_queries(collection_id, top_n, since)
        volume = await self.get_query_volume(collection_id, since)

        return AnalyticsSummary(
            total_queries=total,
            unique_queries=unique,
            latency=latency,
            top_queries=top_queries,
            volume=volume,
        )

    async def get_top_queries(
        self,
        collection_id: uuid.UUID,
        top_n: int = 10,
        since: datetime | None = None,
    ) -> list[QueryFrequency]:
        """Return the most frequent queries for a collection.

        Args:
            collection_id: The collection to analyse.
            top_n: Maximum number of entries.
            since: Optional lower bound on ``created_at``.

        Returns:
            List of QueryFrequency ordered by count descending.
        """
        stmt = (
            select(
                QueryLogModel.query_text,
                func.count().label("cnt"),
            )
            .where(QueryLogModel.collection_id == collection_id)
            .group_by(QueryLogModel.query_text)
            .order_by(func.count().desc())
            .limit(top_n)
        )
        stmt = self._apply_since(stmt, since)

        result = await self._session.execute(stmt)
        return [
            QueryFrequency(query_text=row.query_text, count=row.cnt)
            for row in result.all()
        ]

    async def get_latency_stats(
        self,
        collection_id: uuid.UUID,
        since: datetime | None = None,
    ) -> LatencyStats | None:
        """Compute latency statistics for a collection.

        Only rows where ``latency_ms`` is not null are included.

        Args:
            collection_id: The collection to analyse.
            since: Optional lower bound on ``created_at``.

        Returns:
            LatencyStats or None if no data is available.
        """
        stmt = select(
            func.count().label("cnt"),
            func.avg(QueryLogModel.latency_ms).label("avg_ms"),
            func.min(QueryLogModel.latency_ms).label("min_ms"),
            func.max(QueryLogModel.latency_ms).label("max_ms"),
            func.percentile_cont(0.5)
            .within_group(QueryLogModel.latency_ms)
            .label("p50_ms"),
            func.percentile_cont(0.95)
            .within_group(QueryLogModel.latency_ms)
            .label("p95_ms"),
        ).where(
            QueryLogModel.collection_id == collection_id,
            QueryLogModel.latency_ms.isnot(None),
        )
        stmt = self._apply_since(stmt, since)

        result = await self._session.execute(stmt)
        row = result.one()

        if row.cnt == 0:
            return None

        return LatencyStats(
            avg_ms=float(row.avg_ms),
            min_ms=float(row.min_ms),
            max_ms=float(row.max_ms),
            p50_ms=float(row.p50_ms),
            p95_ms=float(row.p95_ms),
            sample_count=int(row.cnt),
        )

    async def get_query_volume(
        self,
        collection_id: uuid.UUID,
        since: datetime | None = None,
    ) -> list[VolumeDataPoint]:
        """Return daily query counts for a collection.

        Args:
            collection_id: The collection to analyse.
            since: Optional lower bound on ``created_at``.

        Returns:
            List of VolumeDataPoint ordered by date ascending.
        """
        date_trunc = func.date_trunc("day", QueryLogModel.created_at).label("day")
        stmt = (
            select(date_trunc, func.count().label("cnt"))
            .where(QueryLogModel.collection_id == collection_id)
            .group_by(date_trunc)
            .order_by(date_trunc)
        )
        stmt = self._apply_since(stmt, since)

        result = await self._session.execute(stmt)
        return [
            VolumeDataPoint(date=row.day, count=row.cnt)
            for row in result.all()
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_counts(
        self,
        collection_id: uuid.UUID,
        since: datetime | None,
    ) -> tuple[int, int]:
        """Return total and unique query counts.

        Args:
            collection_id: The collection to analyse.
            since: Optional lower bound on ``created_at``.

        Returns:
            Tuple of (total_queries, unique_queries).
        """
        stmt = select(
            func.count().label("total"),
            func.count(distinct(QueryLogModel.query_text)).label("unique"),
        ).where(QueryLogModel.collection_id == collection_id)
        stmt = self._apply_since(stmt, since)

        result = await self._session.execute(stmt)
        row = result.one()
        return int(row.total), int(row.unique)

    @staticmethod
    def _apply_since(stmt: Any, since: datetime | None) -> Any:
        """Apply an optional ``created_at >= since`` filter.

        Args:
            stmt: A SQLAlchemy select statement.
            since: Lower bound timestamp or None.

        Returns:
            The potentially filtered statement.
        """
        if since is not None:
            stmt = stmt.where(QueryLogModel.created_at >= since)
        return stmt

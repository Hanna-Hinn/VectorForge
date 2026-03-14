"""Analytics endpoints for query log analysis."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query

from server.dependencies import ApiKey, DbSession
from server.schemas import (
    AnalyticsSummaryResponse,
    LatencyStatsResponse,
    QueryFrequencyResponse,
    TopQueriesResponse,
    VolumeDataPointResponse,
)
from vectorforge.analytics.service import QueryAnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

SinceParam = Annotated[datetime | None, Query(alias="from")]


@router.get(
    "/{collection_id}/summary",
    response_model=AnalyticsSummaryResponse,
)
async def get_summary(
    collection_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
    since: SinceParam = None,
    top_n: int = Query(default=10, ge=1, le=100),
) -> AnalyticsSummaryResponse:
    """Get analytics summary for a collection."""
    logger.info("Fetching analytics summary for collection %s", collection_id)
    svc = QueryAnalyticsService(session)
    summary = await svc.get_summary(collection_id, since=since, top_n=top_n)

    latency_resp: LatencyStatsResponse | None = None
    if summary.latency is not None:
        latency_resp = LatencyStatsResponse(
            avg_ms=summary.latency.avg_ms,
            min_ms=summary.latency.min_ms,
            max_ms=summary.latency.max_ms,
            p50_ms=summary.latency.p50_ms,
            p95_ms=summary.latency.p95_ms,
            sample_count=summary.latency.sample_count,
        )

    return AnalyticsSummaryResponse(
        total_queries=summary.total_queries,
        unique_queries=summary.unique_queries,
        latency=latency_resp,
        top_queries=[
            QueryFrequencyResponse(query_text=q.query_text, count=q.count)
            for q in summary.top_queries
        ],
        volume=[
            VolumeDataPointResponse(date=v.date, count=v.count)
            for v in summary.volume
        ],
    )


@router.get(
    "/{collection_id}/top-queries",
    response_model=TopQueriesResponse,
)
async def get_top_queries(
    collection_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
    limit: int = Query(default=20, ge=1, le=100),
    since: SinceParam = None,
) -> TopQueriesResponse:
    """Get the most frequent queries for a collection."""
    logger.info("Fetching top %d queries for collection %s", limit, collection_id)
    svc = QueryAnalyticsService(session)
    queries = await svc.get_top_queries(collection_id, top_n=limit, since=since)
    return TopQueriesResponse(
        queries=[
            QueryFrequencyResponse(query_text=q.query_text, count=q.count)
            for q in queries
        ],
    )


@router.get(
    "/{collection_id}/latency",
    response_model=LatencyStatsResponse | None,
)
async def get_latency_stats(
    collection_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
    since: SinceParam = None,
) -> LatencyStatsResponse | None:
    """Get latency statistics for a collection."""
    svc = QueryAnalyticsService(session)
    stats = await svc.get_latency_stats(collection_id, since=since)
    if stats is None:
        return None
    return LatencyStatsResponse(
        avg_ms=stats.avg_ms,
        min_ms=stats.min_ms,
        max_ms=stats.max_ms,
        p50_ms=stats.p50_ms,
        p95_ms=stats.p95_ms,
        sample_count=stats.sample_count,
    )

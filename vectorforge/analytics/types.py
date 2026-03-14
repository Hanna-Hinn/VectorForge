"""Pydantic models for query analytics results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class QueryFrequency(BaseModel):
    """A query text and how often it was asked."""

    query_text: str
    count: int


class LatencyStats(BaseModel):
    """Aggregated latency statistics for a collection's queries."""

    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    sample_count: int


class VolumeDataPoint(BaseModel):
    """Query volume for a specific date."""

    date: datetime
    count: int


class AnalyticsSummary(BaseModel):
    """High-level analytics summary for a collection."""

    total_queries: int
    unique_queries: int
    latency: LatencyStats | None = None
    top_queries: list[QueryFrequency]
    volume: list[VolumeDataPoint]

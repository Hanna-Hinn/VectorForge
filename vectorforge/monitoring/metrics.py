"""In-process metrics collector for VectorForge.

Tracks counters (monotonic), gauges (point-in-time), and histograms
(distributions). Thread-safe via a lock. All metrics are keyed by
a dotted name and optional string tags.
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict
from datetime import UTC, datetime

from pydantic import BaseModel


class HistogramSummary(BaseModel):
    """Statistical summary of a histogram metric."""

    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float


class MetricsSnapshot(BaseModel):
    """Point-in-time snapshot of all collected metrics."""

    timestamp: datetime
    counters: dict[str, dict[str, float]]
    gauges: dict[str, dict[str, float]]
    histograms: dict[str, dict[str, HistogramSummary]]


def _tags_key(tags: dict[str, str] | None) -> str:
    """Convert a tags dict to a deterministic string key.

    Args:
        tags: Optional tag key-value pairs.

    Returns:
        A sorted, comma-separated string representation.
    """
    if not tags:
        return ""
    return ",".join(f"{k}={v}" for k, v in sorted(tags.items()))


def _percentile(data: list[float], pct: float) -> float:
    """Compute a percentile value from a sorted list.

    Args:
        data: Sorted list of float values.
        pct: Percentile to compute (0-100).

    Returns:
        The percentile value.
    """
    if not data:
        return 0.0
    k = (len(data) - 1) * (pct / 100.0)
    floor = int(k)
    ceil = floor + 1
    if ceil >= len(data):
        return data[floor]
    d = k - floor
    return data[floor] + d * (data[ceil] - data[floor])


class MetricsCollector:
    """Thread-safe in-process metrics store.

    Supports counters, gauges, and histograms with optional tags.
    Use ``get_metrics_collector()`` to obtain the singleton instance.
    """

    def __init__(self) -> None:
        self._counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: dict[str, dict[str, float]] = defaultdict(dict)
        self._histograms: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._lock = threading.Lock()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Dotted metric name.
            value: Amount to increment by.
            tags: Optional key-value tags.
        """
        key = _tags_key(tags)
        with self._lock:
            self._counters[name][key] += value

    def gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric to a specific value.

        Args:
            name: Dotted metric name.
            value: The gauge value.
            tags: Optional key-value tags.
        """
        key = _tags_key(tags)
        with self._lock:
            self._gauges[name][key] = value

    def observe(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record an observation in a histogram metric.

        Args:
            name: Dotted metric name.
            value: The observed value.
            tags: Optional key-value tags.
        """
        key = _tags_key(tags)
        with self._lock:
            self._histograms[name][key].append(value)

    def snapshot(self) -> MetricsSnapshot:
        """Create a point-in-time snapshot of all metrics.

        Returns:
            A MetricsSnapshot with deep-copied metric data.
        """
        with self._lock:
            counters_copy: dict[str, dict[str, float]] = {
                name: dict(tags_map) for name, tags_map in self._counters.items()
            }
            gauges_copy: dict[str, dict[str, float]] = {
                name: dict(tags_map) for name, tags_map in self._gauges.items()
            }
            histograms_copy: dict[str, dict[str, HistogramSummary]] = {}
            for name, tags_map in self._histograms.items():
                histograms_copy[name] = {}
                for tag_key, values in tags_map.items():
                    sorted_vals = sorted(values)
                    total = sum(sorted_vals)
                    count = len(sorted_vals)
                    histograms_copy[name][tag_key] = HistogramSummary(
                        count=count,
                        sum=total,
                        min=sorted_vals[0] if sorted_vals else 0.0,
                        max=sorted_vals[-1] if sorted_vals else 0.0,
                        avg=statistics.mean(sorted_vals) if sorted_vals else 0.0,
                        p50=_percentile(sorted_vals, 50),
                        p95=_percentile(sorted_vals, 95),
                        p99=_percentile(sorted_vals, 99),
                    )

        return MetricsSnapshot(
            timestamp=datetime.now(UTC),
            counters=counters_copy,
            gauges=gauges_copy,
            histograms=histograms_copy,
        )

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Module-level singleton
_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Return the singleton MetricsCollector instance.

    Returns:
        The global MetricsCollector.
    """
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector


def reset_metrics_collector() -> None:
    """Reset the singleton collector. Primarily for testing."""
    global _collector
    with _collector_lock:
        _collector = None

"""Monitoring module for VectorForge.

Provides structured logging, in-process metrics collection,
instrumentation decorators, and health check infrastructure.
"""

from vectorforge.monitoring.decorators import instrument
from vectorforge.monitoring.health import (
    ComponentHealth,
    HealthChecker,
    SystemHealth,
    database_health_probe,
    pgvector_health_probe,
)
from vectorforge.monitoring.logging import JSONFormatter, configure_logging
from vectorforge.monitoring.metrics import (
    HistogramSummary,
    MetricsCollector,
    MetricsSnapshot,
    get_metrics_collector,
    reset_metrics_collector,
)

__all__ = [
    "ComponentHealth",
    "HealthChecker",
    "HistogramSummary",
    "JSONFormatter",
    "MetricsCollector",
    "MetricsSnapshot",
    "SystemHealth",
    "configure_logging",
    "database_health_probe",
    "get_metrics_collector",
    "instrument",
    "pgvector_health_probe",
    "reset_metrics_collector",
]

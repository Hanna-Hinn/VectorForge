"""Unit tests for VectorForge monitoring infrastructure."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vectorforge.config.settings import MonitoringConfig
from vectorforge.monitoring.decorators import instrument
from vectorforge.monitoring.health import ComponentHealth, HealthChecker
from vectorforge.monitoring.logging import JSONFormatter, configure_logging
from vectorforge.monitoring.metrics import MetricsCollector, get_metrics_collector

# ---------------------------------------------------------------------------
# Logging Tests
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    """Tests for the JSON log formatter."""

    def test_json_format_output(self) -> None:
        """JSON formatter produces valid JSON with required keys."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "timestamp" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Hello world"
        assert parsed["line"] == 42

    def test_json_format_with_extra(self) -> None:
        """JSON formatter merges extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.extra = {"request_id": "abc123"}  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "abc123"


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_json_logging_config(self) -> None:
        """Logging is configured with JSON format."""
        config = MonitoringConfig(log_level="DEBUG", log_format="json")
        configure_logging(config)

        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) >= 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_text_logging_config(self) -> None:
        """Logging is configured with text format."""
        config = MonitoringConfig(log_level="WARNING", log_format="text")
        configure_logging(config)

        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_noisy_loggers_suppressed(self) -> None:
        """Third-party loggers are set to WARNING."""
        config = MonitoringConfig()
        configure_logging(config)

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    """Tests for the MetricsCollector."""

    def test_counter_increment(self) -> None:
        """Counter increments accumulate correctly."""
        mc = MetricsCollector()
        mc.increment("test.counter")
        mc.increment("test.counter")
        mc.increment("test.counter")

        snap = mc.snapshot()
        assert snap.counters["test.counter"][""] == 3.0

    def test_counter_with_tags(self) -> None:
        """Counters with different tags are tracked separately."""
        mc = MetricsCollector()
        mc.increment("api.calls", tags={"provider": "voyage"})
        mc.increment("api.calls", tags={"provider": "cohere"})
        mc.increment("api.calls", tags={"provider": "voyage"})

        snap = mc.snapshot()
        assert snap.counters["api.calls"]["provider=voyage"] == 2.0
        assert snap.counters["api.calls"]["provider=cohere"] == 1.0

    def test_gauge_overwrites(self) -> None:
        """Gauge values are overwritten, not accumulated."""
        mc = MetricsCollector()
        mc.gauge("active_connections", 5)
        mc.gauge("active_connections", 3)

        snap = mc.snapshot()
        assert snap.gauges["active_connections"][""] == 3.0

    def test_histogram_percentiles(self) -> None:
        """Histogram computes correct statistical summaries."""
        mc = MetricsCollector()
        values = [10, 20, 30, 40, 50, 100, 200, 500, 1000]
        for v in values:
            mc.observe("request.duration_ms", float(v))

        snap = mc.snapshot()
        hist = snap.histograms["request.duration_ms"][""]
        assert hist.count == 9
        assert hist.min == 10.0
        assert hist.max == 1000.0
        assert 40 <= hist.p50 <= 60  # median around 50
        assert hist.p99 >= 500

    def test_reset_clears_all(self) -> None:
        """reset() clears all metrics."""
        mc = MetricsCollector()
        mc.increment("test")
        mc.gauge("test", 1)
        mc.observe("test", 1)
        mc.reset()

        snap = mc.snapshot()
        assert snap.counters == {}
        assert snap.gauges == {}
        assert snap.histograms == {}

    def test_singleton_access(self) -> None:
        """get_metrics_collector returns the same instance."""
        mc1 = get_metrics_collector()
        mc2 = get_metrics_collector()
        assert mc1 is mc2


# ---------------------------------------------------------------------------
# Instrument Decorator Tests
# ---------------------------------------------------------------------------


class TestInstrumentDecorator:
    """Tests for the @instrument decorator."""

    async def test_async_success(self) -> None:
        """Decorator records calls and duration on success."""

        @instrument(name="test.func")
        async def sample_func() -> str:
            return "ok"

        result = await sample_func()
        assert result == "ok"

        mc = get_metrics_collector()
        snap = mc.snapshot()
        assert snap.counters["test.func.calls"][""] == 1.0
        assert "test.func.duration_ms" in snap.histograms
        assert "test.func.errors" not in snap.counters

    async def test_async_error(self) -> None:
        """Decorator records errors and re-raises exceptions."""

        @instrument(name="test.failing")
        async def failing_func() -> None:
            msg = "boom"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="boom"):
            await failing_func()

        mc = get_metrics_collector()
        snap = mc.snapshot()
        assert snap.counters["test.failing.errors"]["error_type=ValueError"] == 1.0

    def test_sync_success(self) -> None:
        """Decorator works with sync functions."""

        @instrument(name="test.sync")
        def sync_func() -> int:
            return 42

        result = sync_func()
        assert result == 42

        mc = get_metrics_collector()
        snap = mc.snapshot()
        assert snap.counters["test.sync.calls"][""] == 1.0

    def test_sync_error(self) -> None:
        """Decorator records errors on sync functions."""

        @instrument(name="test.sync_fail")
        def sync_fail() -> None:
            msg = "sync boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="sync boom"):
            sync_fail()

        mc = get_metrics_collector()
        snap = mc.snapshot()
        assert snap.counters["test.sync_fail.errors"]["error_type=RuntimeError"] == 1.0


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


class TestHealthChecker:
    """Tests for the HealthChecker."""

    async def test_all_healthy(self) -> None:
        """check_all returns healthy when all probes pass."""
        checker = HealthChecker()

        async def healthy_db() -> ComponentHealth:
            return ComponentHealth(
                name="database",
                status="healthy",
                last_checked=datetime.now(UTC),
            )

        async def healthy_cache() -> ComponentHealth:
            return ComponentHealth(
                name="cache",
                status="healthy",
                last_checked=datetime.now(UTC),
            )

        checker.register("database", healthy_db)
        checker.register("cache", healthy_cache)

        result = await checker.check_all()
        assert result.status == "healthy"
        assert len(result.components) == 2

    async def test_degraded_status(self) -> None:
        """check_all returns degraded when any probe is degraded."""
        checker = HealthChecker()

        async def healthy_probe() -> ComponentHealth:
            return ComponentHealth(
                name="db",
                status="healthy",
                last_checked=datetime.now(UTC),
            )

        async def degraded_probe() -> ComponentHealth:
            return ComponentHealth(
                name="cache",
                status="degraded",
                message="High latency",
                last_checked=datetime.now(UTC),
            )

        checker.register("db", healthy_probe)
        checker.register("cache", degraded_probe)

        result = await checker.check_all()
        assert result.status == "degraded"

    async def test_unhealthy_on_timeout(self) -> None:
        """check_all marks timed-out probes as unhealthy."""
        checker = HealthChecker()

        async def slow_probe() -> ComponentHealth:
            await asyncio.sleep(10)
            return ComponentHealth(
                name="slow",
                status="healthy",
                last_checked=datetime.now(UTC),
            )

        checker.register("slow_service", slow_probe)

        result = await checker.check_all(timeout=0.1)
        assert result.status == "unhealthy"
        slow_component = result.components[0]
        assert slow_component.status == "unhealthy"
        assert slow_component.message is not None
        assert "timed out" in slow_component.message

    async def test_unhealthy_on_exception(self) -> None:
        """check_all marks probes that raise as unhealthy."""
        checker = HealthChecker()

        async def broken_probe() -> ComponentHealth:
            msg = "connection refused"
            raise ConnectionError(msg)

        checker.register("broken", broken_probe)

        result = await checker.check_all()
        assert result.status == "unhealthy"
        assert "connection refused" in (result.components[0].message or "")

    async def test_check_one(self) -> None:
        """check_one runs a single probe by name."""
        checker = HealthChecker()

        async def ok_probe() -> ComponentHealth:
            return ComponentHealth(
                name="service",
                status="healthy",
                last_checked=datetime.now(UTC),
            )

        checker.register("service", ok_probe)

        result = await checker.check_one("service")
        assert result.status == "healthy"

    async def test_check_one_missing_raises(self) -> None:
        """check_one raises KeyError for unregistered probes."""
        checker = HealthChecker()
        with pytest.raises(KeyError):
            await checker.check_one("nonexistent")

    async def test_unregister(self) -> None:
        """unregister removes a probe."""
        checker = HealthChecker()

        async def probe() -> ComponentHealth:
            return ComponentHealth(
                name="tmp",
                status="healthy",
                last_checked=datetime.now(UTC),
            )

        checker.register("tmp", probe)
        checker.unregister("tmp")

        result = await checker.check_all()
        assert len(result.components) == 0


class TestMonitoringConfigDefaults:
    """Tests for MonitoringConfig validation."""

    def test_defaults(self) -> None:
        """MonitoringConfig loads with correct defaults."""
        config = MonitoringConfig()
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.metrics_enabled is True
        assert config.health_check_timeout_seconds == 5

    def test_invalid_log_level_raises(self) -> None:
        """Invalid log_level raises ValidationError."""
        with pytest.raises(ValidationError):
            MonitoringConfig(log_level="INVALID")  # type: ignore[arg-type]

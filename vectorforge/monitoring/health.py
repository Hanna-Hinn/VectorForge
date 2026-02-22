"""Health check system for VectorForge.

Components register async probes that report their status.
The HealthChecker aggregates all probes into a system-wide health report.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type alias for health probe callables
HealthProbe = Callable[[], Awaitable["ComponentHealth"]]


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str
    status: Literal["healthy", "degraded", "unhealthy"]
    latency_ms: float | None = None
    message: str | None = None
    last_checked: datetime


class SystemHealth(BaseModel):
    """Aggregated health status of all registered components."""

    status: Literal["healthy", "degraded", "unhealthy"]
    components: list[ComponentHealth]
    checked_at: datetime


class HealthChecker:
    """Aggregates health probes from all VectorForge components.

    Components register probe callables during initialization.
    ``check_all()`` runs all probes and returns a system-wide status.
    """

    def __init__(self) -> None:
        self._probes: dict[str, HealthProbe] = {}

    def register(self, name: str, probe: HealthProbe) -> None:
        """Register a health probe for a component.

        Args:
            name: Component identifier (e.g., "database", "pgvector").
            probe: Async callable returning a ComponentHealth.
        """
        self._probes[name] = probe
        logger.debug("Health probe registered: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a registered health probe.

        Args:
            name: The component name to unregister.
        """
        self._probes.pop(name, None)

    async def check_all(self, timeout: float = 5.0) -> SystemHealth:
        """Run all registered health probes and aggregate results.

        Args:
            timeout: Maximum seconds to wait for each probe.

        Returns:
            SystemHealth with per-component results and overall status.
        """
        results: list[ComponentHealth] = []
        now = datetime.now(UTC)

        for name, probe in self._probes.items():
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(probe(), timeout=timeout)
                result.latency_ms = (time.perf_counter() - start) * 1000
                result.last_checked = now
                results.append(result)
            except TimeoutError:
                results.append(
                    ComponentHealth(
                        name=name,
                        status="unhealthy",
                        message="Health probe timed out",
                        latency_ms=timeout * 1000,
                        last_checked=now,
                    )
                )
            except Exception as exc:
                results.append(
                    ComponentHealth(
                        name=name,
                        status="unhealthy",
                        message=str(exc),
                        latency_ms=(time.perf_counter() - start) * 1000,
                        last_checked=now,
                    )
                )

        overall: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        if any(c.status == "unhealthy" for c in results):
            overall = "unhealthy"
        elif any(c.status == "degraded" for c in results):
            overall = "degraded"

        return SystemHealth(status=overall, components=results, checked_at=now)

    async def check_one(self, name: str, timeout: float = 5.0) -> ComponentHealth:
        """Run a single health probe by name.

        Args:
            name: The component name to check.
            timeout: Maximum seconds to wait.

        Returns:
            ComponentHealth for the specified component.

        Raises:
            KeyError: If no probe is registered with the given name.
        """
        probe = self._probes[name]
        now = datetime.now(UTC)
        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(probe(), timeout=timeout)
            result.latency_ms = (time.perf_counter() - start) * 1000
            result.last_checked = now
            return result
        except TimeoutError:
            return ComponentHealth(
                name=name,
                status="unhealthy",
                message="Health probe timed out",
                latency_ms=timeout * 1000,
                last_checked=now,
            )
        except Exception as exc:
            return ComponentHealth(
                name=name,
                status="unhealthy",
                message=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
                last_checked=now,
            )


# ---------------------------------------------------------------------------
# Built-in health probes
# ---------------------------------------------------------------------------


async def database_health_probe(
    engine: object,
) -> ComponentHealth:
    """Health probe for the PostgreSQL database.

    Args:
        engine: An AsyncDatabaseEngine instance.

    Returns:
        ComponentHealth indicating database reachability.
    """
    from vectorforge.db.engine import AsyncDatabaseEngine

    if not isinstance(engine, AsyncDatabaseEngine):
        msg = (
            f"Expected AsyncDatabaseEngine, got {type(engine).__name__}"
        )
        raise TypeError(msg)
    now = datetime.now(UTC)
    try:
        healthy = await engine.health_check()
        return ComponentHealth(
            name="database",
            status="healthy" if healthy else "unhealthy",
            message=None if healthy else "Health check query failed",
            last_checked=now,
        )
    except Exception as exc:
        return ComponentHealth(
            name="database",
            status="unhealthy",
            message=str(exc),
            last_checked=now,
        )


async def pgvector_health_probe(
    engine: object,
) -> ComponentHealth:
    """Health probe for the pgvector extension.

    Args:
        engine: An AsyncDatabaseEngine instance.

    Returns:
        ComponentHealth indicating pgvector availability and version.
    """
    from sqlalchemy import text

    from vectorforge.db.engine import AsyncDatabaseEngine

    if not isinstance(engine, AsyncDatabaseEngine):
        msg = (
            f"Expected AsyncDatabaseEngine, got {type(engine).__name__}"
        )
        raise TypeError(msg)
    now = datetime.now(UTC)
    try:
        async with engine.get_session() as session:
            result = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            row = result.scalar_one_or_none()
            if row:
                return ComponentHealth(
                    name="pgvector",
                    status="healthy",
                    message=f"v{row}",
                    last_checked=now,
                )
            return ComponentHealth(
                name="pgvector",
                status="unhealthy",
                message="extension not found",
                last_checked=now,
            )
    except Exception as exc:
        return ComponentHealth(
            name="pgvector",
            status="unhealthy",
            message=str(exc),
            last_checked=now,
        )

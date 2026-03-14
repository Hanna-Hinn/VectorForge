"""Background scheduler for periodic evaluation runs.

Uses a plain ``asyncio.Task`` loop — no external scheduler dependency.
Start the scheduler during application lifespan and stop it on shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from vectorforge.evaluation.config import EvaluationConfig
from vectorforge.evaluation.types import EvaluationRun

logger = logging.getLogger(__name__)


class BackgroundScheduler:
    """Runs periodic evaluations on a configurable interval.

    The scheduler owns an asyncio task that sleeps between runs.
    It must be started **after** the event loop is running (e.g. in a
    FastAPI lifespan handler) and stopped on shutdown.

    Args:
        service_factory: Async callable that returns a fully-wired
            ``EvaluationService`` with its own session.  A new service
            is created for each scheduled run so that sessions are
            not held open across long sleep intervals.
        config: Evaluation configuration (controls interval, enabled flag).
    """

    def __init__(
        self,
        service_factory: Any,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._service_factory = service_factory
        self._config = config or EvaluationConfig()
        self._task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the scheduler loop is currently active."""
        return self._running

    def start(self) -> None:
        """Start the background evaluation loop.

        Does nothing if evaluation is disabled in config or the
        scheduler is already running.
        """
        if not self._config.enabled:
            logger.info("Evaluation scheduler disabled (config.enabled=False)")
            return

        if self._running:
            logger.warning("Evaluation scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Evaluation scheduler started (every %dh)",
            self._config.schedule_interval_hours,
        )

    async def stop(self) -> None:
        """Stop the background evaluation loop gracefully."""
        if not self._running:
            return

        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Evaluation scheduler stopped")

    async def trigger_now(self) -> EvaluationRun:
        """Trigger an immediate evaluation run.

        Returns:
            The completed EvaluationRun.
        """
        service = await self._service_factory()
        return await service.run_evaluation(self._config)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Core loop: run evaluation then sleep."""
        interval_seconds = self._config.schedule_interval_hours * 3600

        while self._running:
            try:
                service = await self._service_factory()
                run = await service.run_evaluation(self._config)
                logger.info(
                    "Scheduled evaluation run %s finished (status=%s)",
                    run.id,
                    run.status,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduled evaluation failed")

            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break

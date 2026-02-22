"""Instrumentation decorator for automatic metrics and logging.

The ``@instrument`` decorator wraps sync and async functions to:
- Log entry/exit with structured context
- Record execution duration as a histogram metric
- Increment error counter on exceptions
- Re-raise all exceptions (never swallows)
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from vectorforge.monitoring.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def instrument(
    name: str | None = None,
    log_args: bool = False,
    tags: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """Decorator that instruments a function with metrics and logging.

    Args:
        name: Override metric name. Defaults to ``module.qualname``.
        log_args: Whether to log function arguments (be careful with PII).
        tags: Additional tags to attach to all metrics.

    Returns:
        A decorator that wraps the target function.
    """
    merged_tags = dict(tags) if tags else {}

    def decorator(func: F) -> F:
        metric_name = name or f"{func.__module__}.{func.__qualname__}"

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                metrics = get_metrics_collector()
                extra: dict[str, Any] = {"metric": metric_name}
                if log_args:
                    extra["args"] = str(args)
                    extra["kwargs"] = str(kwargs)

                logger.debug("→ %s", metric_name, extra={"extra": extra})
                start = time.perf_counter()

                try:
                    result = await func(*args, **kwargs)
                    duration = (time.perf_counter() - start) * 1000
                    logger.debug(
                        "← %s",
                        metric_name,
                        extra={"extra": {**extra, "duration_ms": duration}},
                    )
                    metrics.observe(f"{metric_name}.duration_ms", duration, merged_tags)
                    metrics.increment(f"{metric_name}.calls", tags=merged_tags)
                    return result
                except Exception as exc:
                    duration = (time.perf_counter() - start) * 1000
                    logger.error(
                        "✗ %s",
                        metric_name,
                        extra={
                            "extra": {
                                **extra,
                                "error": str(exc),
                                "duration_ms": duration,
                            }
                        },
                    )
                    error_tags = {**merged_tags, "error_type": type(exc).__name__}
                    metrics.increment(f"{metric_name}.errors", tags=error_tags)
                    raise

            return cast(F, async_wrapper)

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                metrics = get_metrics_collector()
                extra: dict[str, Any] = {"metric": metric_name}
                if log_args:
                    extra["args"] = str(args)
                    extra["kwargs"] = str(kwargs)

                logger.debug("→ %s", metric_name, extra={"extra": extra})
                start = time.perf_counter()

                try:
                    result = func(*args, **kwargs)
                    duration = (time.perf_counter() - start) * 1000
                    logger.debug(
                        "← %s",
                        metric_name,
                        extra={"extra": {**extra, "duration_ms": duration}},
                    )
                    metrics.observe(f"{metric_name}.duration_ms", duration, merged_tags)
                    metrics.increment(f"{metric_name}.calls", tags=merged_tags)
                    return result
                except Exception as exc:
                    duration = (time.perf_counter() - start) * 1000
                    logger.error(
                        "✗ %s",
                        metric_name,
                        extra={
                            "extra": {
                                **extra,
                                "error": str(exc),
                                "duration_ms": duration,
                            }
                        },
                    )
                    error_tags = {**merged_tags, "error_type": type(exc).__name__}
                    metrics.increment(f"{metric_name}.errors", tags=error_tags)
                    raise

            return cast(F, sync_wrapper)

    return decorator

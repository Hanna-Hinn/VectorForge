"""Shared CLI helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vectorforge.db.engine import AsyncDatabaseEngine


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine synchronously.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's return value.
    """
    return asyncio.run(coro)


@asynccontextmanager
async def managed_engine() -> AsyncIterator[AsyncDatabaseEngine]:
    """Create a managed async database engine.

    Loads config, initialises the engine, and ensures it is
    disposed on exit — centralising the lifecycle boilerplate.

    Yields:
        An initialized AsyncDatabaseEngine.
    """
    from vectorforge.config.settings import load_config
    from vectorforge.db.engine import AsyncDatabaseEngine

    config = load_config()
    db = AsyncDatabaseEngine(config.database)
    db.create_engine()
    try:
        yield db
    finally:
        await db.dispose()

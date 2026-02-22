"""Async database engine and session management for VectorForge.

Provides a singleton-style engine wrapper that manages the SQLAlchemy
async engine and session factory. All database access flows through this module.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from vectorforge.config.settings import DatabaseConfig
from vectorforge.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class AsyncDatabaseEngine:
    """Manages the async SQLAlchemy engine and session factory.

    Args:
        config: Database configuration with connection parameters.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    def create_engine(self) -> AsyncEngine:
        """Create and configure the async engine.

        Returns:
            The configured AsyncEngine instance.
        """
        url = self._config.database_url
        self._engine = create_async_engine(
            url,
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
            echo=self._config.echo_sql,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info(
            "Database engine created",
            extra={
                "host": self._config.host,
                "port": self._config.port,
                "database": self._config.database,
                "pool_size": self._config.pool_size,
            },
        )
        return self._engine

    @property
    def engine(self) -> AsyncEngine:
        """Return the underlying engine, raising if not initialized."""
        if self._engine is None:
            msg = "Engine not initialized. Call create_engine() first."
            raise DatabaseError(msg)
        return self._engine

    @asynccontextmanager
    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Yield an async session within a managed context.

        The session is committed on success and rolled back on exception.

        Yields:
            An AsyncSession bound to the engine.

        Raises:
            DatabaseError: If the session factory is not initialized.
        """
        if self._session_factory is None:
            msg = "Session factory not initialized. Call create_engine() first."
            raise DatabaseError(msg)

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def dispose(self) -> None:
        """Dispose of the engine and release all pooled connections."""
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Database engine disposed")
            self._engine = None
            self._session_factory = None

    async def health_check(self) -> bool:
        """Execute a simple query to verify database connectivity.

        Returns:
            True if the database is reachable, False otherwise.
        """
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error("Database health check failed", extra={"error": str(exc)})
            return False

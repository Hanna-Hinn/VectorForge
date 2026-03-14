"""Storage router — directs content to the appropriate backend.

Routes documents to PostgreSQL or S3 based on content size,
with fallback to PostgreSQL if S3 is not configured.
"""

from __future__ import annotations

import logging

from vectorforge.config.settings import StorageConfig
from vectorforge.exceptions import StorageError
from vectorforge.storage.base import BaseStorageBackend
from vectorforge.storage.postgres import PostgresStorageBackend
from vectorforge.storage.s3 import S3StorageBackend

logger = logging.getLogger(__name__)


class StorageRouter:
    """Route document storage to the appropriate backend.

    By default, documents smaller than ``threshold_mb`` are stored
    in PostgreSQL and larger ones in S3. If S3 is not configured,
    all documents go to PostgreSQL.

    Args:
        config: StorageConfig with backend preferences and S3 settings.
    """

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._threshold_bytes = config.threshold_mb * 1024 * 1024
        self._pg = PostgresStorageBackend()
        self._s3: S3StorageBackend | None = None

        if config.is_s3_configured():
            self._s3 = S3StorageBackend(
                bucket=config.s3_bucket,
                region=config.s3_region,
                access_key=config.s3_access_key,
                secret_key=config.s3_secret_key,
                endpoint_url=config.s3_endpoint_url,
            )
            logger.info(
                "Storage router initialised: pg + s3 (threshold=%dMB)",
                config.threshold_mb,
            )
        else:
            logger.info("Storage router initialised: pg only (S3 not configured)")

    def select_backend(self, content_size: int) -> tuple[BaseStorageBackend, str]:
        """Select the appropriate backend for the given content size.

        Args:
            content_size: The content size in bytes.

        Returns:
            Tuple of (backend instance, backend name).
        """
        if self._s3 is not None and content_size >= self._threshold_bytes:
            return self._s3, "s3"
        return self._pg, "pg"

    async def store(self, key: str, content: bytes) -> tuple[str, str]:
        """Store content in the appropriate backend.

        Args:
            key: The storage key.
            content: The raw content bytes.

        Returns:
            Tuple of (storage key, backend name).
        """
        backend, backend_name = self.select_backend(len(content))
        storage_key = await backend.store(key, content)
        return storage_key, backend_name

    async def retrieve(self, key: str, backend_name: str) -> bytes:
        """Retrieve content from the specified backend.

        Args:
            key: The storage key.
            backend_name: The backend to retrieve from ('pg' or 's3').

        Returns:
            The raw content bytes.

        Raises:
            StorageError: If the backend is unavailable or retrieval fails.
        """
        backend = self._get_backend(backend_name)
        return await backend.retrieve(key)

    async def delete(self, key: str, backend_name: str) -> None:
        """Delete content from the specified backend.

        Args:
            key: The storage key.
            backend_name: The backend to delete from.
        """
        backend = self._get_backend(backend_name)
        await backend.delete(key)

    def _get_backend(self, backend_name: str) -> BaseStorageBackend:
        """Resolve a backend by name.

        Args:
            backend_name: 'pg' or 's3'.

        Returns:
            The corresponding backend instance.

        Raises:
            StorageError: If the backend name is unknown or S3 is not configured.
        """
        if backend_name == "pg":
            return self._pg
        if backend_name == "s3":
            if self._s3 is None:
                msg = "S3 backend requested but not configured"
                raise StorageError(msg)
            return self._s3
        msg = f"Unknown storage backend: {backend_name}"
        raise StorageError(msg)

    @property
    def pg_backend(self) -> PostgresStorageBackend:
        """Access the PostgreSQL backend directly."""
        return self._pg

    @property
    def s3_backend(self) -> S3StorageBackend | None:
        """Access the S3 backend (None if not configured)."""
        return self._s3

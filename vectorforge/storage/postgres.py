"""PostgreSQL storage backend.

Stores document content directly in the documents table
(raw_content column) via the DocumentRepository.
"""

from __future__ import annotations

import logging

from vectorforge.exceptions import StorageError
from vectorforge.storage.base import BaseStorageBackend

logger = logging.getLogger(__name__)


class PostgresStorageBackend(BaseStorageBackend):
    """Store document content in PostgreSQL.

    Content is stored as text in the documents table.
    The key is the document ID; actual persistence is handled
    by the DocumentRepository at the service level.

    This backend acts as a thin wrapper that holds content in
    memory until the caller persists it via the repository.
    """

    def __init__(self) -> None:
        self._cache: dict[str, bytes] = {}

    async def store(self, key: str, content: bytes) -> str:
        """Cache content for later persistence by the repository.

        Args:
            key: The document identifier (usually document ID as string).
            content: The raw content bytes.

        Returns:
            The same key for retrieval.
        """
        self._cache[key] = content
        logger.debug("Stored %d bytes in pg backend (key=%s)", len(content), key)
        return key

    async def retrieve(self, key: str) -> bytes:
        """Retrieve cached content.

        Args:
            key: The storage key.

        Returns:
            The raw content bytes.

        Raises:
            StorageError: If the key is not found in the cache.
        """
        if key not in self._cache:
            msg = f"Content not found in pg backend cache for key={key}"
            raise StorageError(msg)
        return self._cache[key]

    async def delete(self, key: str) -> None:
        """Remove content from the cache.

        Args:
            key: The storage key to delete.
        """
        self._cache.pop(key, None)
        logger.debug("Deleted content from pg backend (key=%s)", key)

    def evict(self, key: str) -> None:
        """Remove a key from the in-memory cache after DB persistence.

        Call this after the content has been persisted via the repository
        to prevent unbounded memory growth.

        Args:
            key: The storage key to evict.
        """
        self._cache.pop(key, None)

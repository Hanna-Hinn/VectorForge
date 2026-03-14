"""Base storage backend ABC.

Defines the interface for document content storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStorageBackend(ABC):
    """Abstract base class for document storage backends."""

    @abstractmethod
    async def store(self, key: str, content: bytes) -> str:
        """Store content and return the storage key.

        Args:
            key: The storage key / path.
            content: The raw content bytes.

        Returns:
            The storage key for retrieval.
        """

    @abstractmethod
    async def retrieve(self, key: str) -> bytes:
        """Retrieve content by key.

        Args:
            key: The storage key.

        Returns:
            The raw content bytes.

        Raises:
            StorageError: If the content cannot be retrieved.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete content by key.

        Args:
            key: The storage key to delete.
        """

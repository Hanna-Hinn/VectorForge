"""Base vector store ABC and distance metric enum.

Defines the interface for all vector store backends.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from vectorforge.models.domain import DistanceMetric, RetrievedChunk


class BaseVectorStore(ABC):
    """Abstract base class for vector store backends.

    Provides upsert, search, delete, and count operations
    for managing embedding vectors.
    """

    @abstractmethod
    async def upsert(
        self,
        chunk_ids: list[uuid.UUID],
        embeddings: list[list[float]],
        model_name: str = "",
        session: object | None = None,
    ) -> None:
        """Insert or update embedding vectors.

        Args:
            chunk_ids: List of chunk UUIDs.
            embeddings: List of embedding vectors.
            model_name: Name of the embedding model used.
            session: Optional database session for transactional use.
                When provided, the store operates within the caller's
                transaction instead of creating its own. Non-SQL
                backends may ignore this parameter.
        """

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        collection_id: uuid.UUID | None = None,
    ) -> list[RetrievedChunk]:
        """Search for similar vectors.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results.
            filters: Optional metadata filters.
            metric: Distance metric to use.
            collection_id: Optional collection scope.

        Returns:
            List of RetrievedChunk results ordered by similarity.
        """

    @abstractmethod
    async def delete_by_ids(self, chunk_ids: list[uuid.UUID]) -> None:
        """Delete embeddings by chunk IDs.

        Args:
            chunk_ids: List of chunk UUIDs to delete.
        """

    @abstractmethod
    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        """Delete all embeddings for a document.

        Args:
            document_id: The document UUID whose embeddings to delete.
        """

    @abstractmethod
    async def count(self) -> int:
        """Count total stored embeddings.

        Returns:
            Number of embedding records.
        """

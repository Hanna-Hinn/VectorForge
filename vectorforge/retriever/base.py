"""Base retriever ABC.

Defines the interface for all retrieval strategies.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from vectorforge.models.domain import RetrievedChunk


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies.

    Subclasses implement ``retrieve`` for their specific search approach
    (dense vector search, sparse BM25, hybrid, etc.).
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        collection_id: uuid.UUID,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The user query text.
            collection_id: The collection to search within.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.
            min_score: Minimum similarity score threshold.

        Returns:
            List of RetrievedChunk results ordered by relevance.
        """

    @abstractmethod
    async def retrieve_with_scores(
        self,
        query: str,
        collection_id: uuid.UUID,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks with similarity scores (no filtering).

        A convenience method that always includes scores and
        applies no minimum score threshold.

        Args:
            query: The user query text.
            collection_id: The collection to search within.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedChunk results with scores.
        """

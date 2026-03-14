"""Base reranker ABC.

Defines the interface for all re-ranking strategies that refine
retrieval results after initial search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vectorforge.models.domain import RetrievedChunk


class BaseReranker(ABC):
    """Abstract base class for re-ranking strategies.

    Re-rankers take an initial set of retrieved chunks and re-order
    them using a more expensive (but more accurate) scoring model.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Re-rank retrieved chunks by relevance to the query.

        Args:
            query: The user query text.
            chunks: Initial set of retrieved chunks to re-rank.
            top_k: Maximum number of results after re-ranking.

        Returns:
            Re-ranked list of RetrievedChunk sorted by relevance.
        """

    @property
    @abstractmethod
    def reranker_name(self) -> str:
        """Return a human-readable name for this reranker."""

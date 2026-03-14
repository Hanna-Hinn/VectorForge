"""Dense vector retriever using embedding search.

Embeds the query via an embedding provider and searches pgvector
for the most similar chunks.
"""

from __future__ import annotations

import logging
import uuid

from vectorforge.db.repositories.collection_repo import CollectionRepository
from vectorforge.embedding.registry import EmbeddingProviderRegistry
from vectorforge.exceptions import NotFoundError
from vectorforge.models.domain import Collection, DistanceMetric, RetrievedChunk
from vectorforge.monitoring.metrics import get_metrics_collector
from vectorforge.pipeline.query import preprocess_query
from vectorforge.retriever.base import BaseRetriever
from vectorforge.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Retriever that performs dense vector similarity search.

    Uses an embedding provider to vectorise the query, then
    searches the vector store for the closest chunks.

    Args:
        embedding_registry: Registry of embedding providers.
        vector_store: The vector store backend.
        collection_repo: Repository for collection lookups.
    """

    def __init__(
        self,
        embedding_registry: EmbeddingProviderRegistry,
        vector_store: BaseVectorStore,
        collection_repo: CollectionRepository,
    ) -> None:
        self._embedding_registry = embedding_registry
        self._vector_store = vector_store
        self._collection_repo = collection_repo

    async def retrieve(
        self,
        query: str,
        collection_id: uuid.UUID,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        min_score: float = 0.0,
        embedding_provider: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks via dense vector search.

        Args:
            query: The user query text.
            collection_id: The collection to search within.
            top_k: Maximum number of results.
            filters: Optional metadata filters.
            min_score: Minimum similarity score threshold.
            embedding_provider: Override embedding provider name.

        Returns:
            List of RetrievedChunk results ordered by relevance.

        Raises:
            NotFoundError: If the collection does not exist.
            ValueError: If the query is empty.
        """
        metrics = get_metrics_collector()
        tags = {"retriever_type": "dense"}

        query = preprocess_query(query)

        collection = await self._collection_repo.find_by_id(collection_id)
        if collection is None:
            msg = f"Collection {collection_id} not found"
            raise NotFoundError(msg)

        provider_name = self._resolve_provider_name(
            embedding_provider, collection
        )
        provider = self._embedding_registry.get(provider_name)

        query_vector = await provider.embed_query(query)

        metric = self._resolve_metric(collection)

        results = await self._vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            metric=metric,
            collection_id=collection_id,
        )

        if min_score > 0.0:
            results = [r for r in results if r.score >= min_score]

        metrics.observe(
            "retriever.results_returned", float(len(results)), tags=tags,
        )
        if not results:
            metrics.increment("retriever.empty_results", tags=tags)

        logger.info(
            "Retrieved %d chunks for query (top_k=%d, metric=%s)",
            len(results),
            top_k,
            metric.value,
        )
        return results

    async def retrieve_with_scores(
        self,
        query: str,
        collection_id: uuid.UUID,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks with scores and no minimum threshold.

        Args:
            query: The user query text.
            collection_id: The collection to search within.
            top_k: Maximum number of results.

        Returns:
            List of RetrievedChunk results with scores.
        """
        return await self.retrieve(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            min_score=0.0,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_provider_name(
        override: str | None,
        collection: Collection,
    ) -> str:
        """Determine which embedding provider to use.

        Args:
            override: Explicit provider name from the caller.
            collection: The Collection domain model.

        Returns:
            The provider name string.
        """
        if override:
            return override
        cfg = collection.embedding_config or {}
        return str(cfg.get("default_provider", "")) or "voyage"

    @staticmethod
    def _resolve_metric(collection: Collection) -> DistanceMetric:
        """Determine the distance metric from collection config.

        Args:
            collection: The Collection domain model.

        Returns:
            The DistanceMetric to use.
        """
        cfg = collection.embedding_config or {}
        raw = cfg.get("metric", "cosine")
        try:
            return DistanceMetric(str(raw))
        except ValueError:
            return DistanceMetric.COSINE

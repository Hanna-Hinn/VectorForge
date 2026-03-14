"""Hybrid retriever combining dense vector and keyword search.

Runs a DenseRetriever and a KeywordSearcher in parallel, then
merges scores via Reciprocal Rank Fusion to leverage the strengths
of both approaches.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from vectorforge.models.domain import RetrievedChunk
from vectorforge.monitoring.metrics import get_metrics_collector
from vectorforge.retriever.base import BaseRetriever
from vectorforge.retriever.fusion import RRFScoreFusion
from vectorforge.retriever.keyword import BaseKeywordSearcher

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Retriever that fuses dense and keyword search results.

    Both searches are dispatched concurrently via ``asyncio.gather``.
    Scores are fused using Reciprocal Rank Fusion (RRF).

    Args:
        dense_retriever: A retriever implementing BaseRetriever.
        keyword_searcher: A searcher implementing BaseKeywordSearcher.
        fusion: An RRFScoreFusion instance for merging results.
        dense_weight: Weight applied to dense results in fusion.
        keyword_weight: Weight applied to keyword results in fusion.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        keyword_searcher: BaseKeywordSearcher,
        fusion: RRFScoreFusion | None = None,
        dense_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> None:
        self._dense = dense_retriever
        self._keyword = keyword_searcher
        self._fusion = fusion or RRFScoreFusion()
        self._dense_weight = dense_weight
        self._keyword_weight = keyword_weight

    async def retrieve(
        self,
        query: str,
        collection_id: uuid.UUID,
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks via hybrid dense + keyword search.

        Args:
            query: The user query text.
            collection_id: The collection to search within.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (applied to dense only).
            min_score: Minimum fused score threshold.

        Returns:
            List of RetrievedChunk results ordered by fused score.
        """
        metrics = get_metrics_collector()
        tags = {"retriever_type": "hybrid"}

        # Fetch more candidates to give fusion room to work
        candidate_k = top_k * 3

        dense_task = self._dense.retrieve(
            query=query,
            collection_id=collection_id,
            top_k=candidate_k,
            filters=filters,
            min_score=0.0,
        )
        keyword_task = self._keyword.search(
            query=query,
            collection_id=collection_id,
            top_k=candidate_k,
        )

        dense_results, keyword_results = await asyncio.gather(
            dense_task, keyword_task,
        )

        fused = self._fusion.fuse(
            dense_results=dense_results,
            keyword_results=keyword_results,
            dense_weight=self._dense_weight,
            keyword_weight=self._keyword_weight,
            top_k=top_k,
        )

        if min_score > 0.0:
            fused = [r for r in fused if r.score >= min_score]

        metrics.observe(
            "retriever.results_returned", float(len(fused)), tags=tags,
        )
        if not fused:
            metrics.increment("retriever.empty_results", tags=tags)

        logger.info(
            "Hybrid retrieved %d chunks (dense=%d, keyword=%d, fused=%d)",
            len(fused),
            len(dense_results),
            len(keyword_results),
            len(fused),
        )
        return fused

    async def retrieve_with_scores(
        self,
        query: str,
        collection_id: uuid.UUID,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks with fused scores, no minimum threshold.

        Args:
            query: The user query text.
            collection_id: The collection to search within.
            top_k: Maximum number of results.

        Returns:
            List of RetrievedChunk results with fused scores.
        """
        return await self.retrieve(
            query=query,
            collection_id=collection_id,
            top_k=top_k,
            min_score=0.0,
        )

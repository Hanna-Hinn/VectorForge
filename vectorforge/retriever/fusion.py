
"""Reciprocal Rank Fusion for combining retrieval results.

Merges results from multiple retrieval strategies (dense + keyword)
using the RRF algorithm, which is robust to score-scale differences
between heterogeneous result lists.
"""

from __future__ import annotations

import logging

from vectorforge.models.domain import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


class RRFScoreFusion:
    """Reciprocal Rank Fusion combiner.

    Merges two ranked result lists by converting positional ranks
    into a unified score: ``score = weight / (k + rank)``.

    Args:
        k: The RRF constant (default 60).
    """

    def __init__(self, k: int = DEFAULT_RRF_K) -> None:
        if k < 1:
            msg = f"RRF k must be >= 1, got {k}"
            raise ValueError(msg)
        self._k = k

    def fuse(
        self,
        dense_results: list[RetrievedChunk],
        keyword_results: list[RetrievedChunk],
        dense_weight: float = 0.6,
        keyword_weight: float = 0.4,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Fuse dense and keyword results via RRF.

        Args:
            dense_results: Ranked results from the dense retriever.
            keyword_results: Ranked results from the keyword searcher.
            dense_weight: Weight applied to dense RRF scores.
            keyword_weight: Weight applied to keyword RRF scores.
            top_k: Maximum number of results to return.

        Returns:
            A merged list of RetrievedChunk sorted by fused score.
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        self._accumulate(dense_results, dense_weight, scores, chunk_map)
        self._accumulate(keyword_results, keyword_weight, scores, chunk_map)

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        fused: list[RetrievedChunk] = []
        for chunk_id in sorted_ids[:top_k]:
            original = chunk_map[chunk_id]
            fused.append(
                RetrievedChunk(
                    chunk=original.chunk,
                    score=scores[chunk_id],
                    document_source=original.document_source,
                )
            )

        logger.debug(
            "RRF fused %d dense + %d keyword → %d results",
            len(dense_results),
            len(keyword_results),
            len(fused),
        )
        return fused

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _accumulate(
        self,
        results: list[RetrievedChunk],
        weight: float,
        scores: dict[str, float],
        chunk_map: dict[str, RetrievedChunk],
    ) -> None:
        """Add RRF scores from one result list.

        Args:
            results: Ranked result list.
            weight: Weighting factor for this source.
            scores: Accumulated chunk_id → score mapping (mutated).
            chunk_map: Chunk id → RetrievedChunk lookup (mutated).
        """
        for rank, result in enumerate(results):
            chunk_id = str(result.chunk.id)
            rrf_score = weight / (self._k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

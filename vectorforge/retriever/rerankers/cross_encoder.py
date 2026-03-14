"""Cross-encoder re-ranker using sentence-transformers.

Loads a cross-encoder model lazily on first use and scores each
(query, chunk) pair independently for high-accuracy re-ranking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from vectorforge.exceptions import RetrievalError
from vectorforge.models.domain import RetrievedChunk
from vectorforge.retriever.reranker import BaseReranker

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Re-ranker backed by a sentence-transformers CrossEncoder model.

    The model is loaded lazily on the first ``rerank()`` call to
    avoid import overhead when not in use.

    Args:
        model_name: The HuggingFace cross-encoder model id.
        device: Torch device string (``"cpu"``, ``"cuda"``, etc.).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    @property
    def reranker_name(self) -> str:
        """Return the reranker identifier."""
        return f"cross-encoder:{self._model_name}"

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Re-rank chunks using the cross-encoder model.

        Args:
            query: The user query text.
            chunks: Initial set of retrieved chunks.
            top_k: Maximum number of results after re-ranking.

        Returns:
            Re-ranked RetrievedChunk list sorted by cross-encoder score.

        Raises:
            RetrievalError: If the cross-encoder model fails to load or predict.
        """
        if not chunks:
            return []

        model = self._load_model()

        pairs = [[query, c.chunk.text] for c in chunks]

        try:
            scores_array = await asyncio.to_thread(model.predict, pairs)
            scores: list[float] = scores_array.tolist()
        except Exception as exc:
            msg = f"Cross-encoder prediction failed: {exc}"
            raise RetrievalError(msg) from exc

        scored = list(zip(chunks, scores, strict=True))
        scored.sort(key=lambda t: t[1], reverse=True)

        reranked: list[RetrievedChunk] = []
        for original, score in scored[:top_k]:
            reranked.append(
                RetrievedChunk(
                    chunk=original.chunk,
                    score=float(score),
                    document_source=original.document_source,
                )
            )

        logger.info(
            "Cross-encoder reranked %d → %d chunks (model=%s)",
            len(chunks),
            len(reranked),
            self._model_name,
        )
        return reranked

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Lazily load the cross-encoder model.

        Returns:
            The loaded CrossEncoder model instance.

        Raises:
            RetrievalError: If sentence-transformers is not installed.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            )
            raise RetrievalError(msg) from exc

        try:
            self._model = CrossEncoder(self._model_name, device=self._device)
        except Exception as exc:
            msg = f"Failed to load cross-encoder model '{self._model_name}': {exc}"
            raise RetrievalError(msg) from exc

        logger.info(
            "Loaded cross-encoder model '%s' on device '%s'",
            self._model_name,
            self._device,
        )
        return self._model

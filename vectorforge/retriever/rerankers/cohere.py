"""Cohere re-ranker using the Cohere Rerank API via httpx.

Sends query + documents to the ``/v2/rerank`` endpoint and
returns re-scored chunks.
"""

from __future__ import annotations

import logging

import httpx

from vectorforge.exceptions import RetrievalError
from vectorforge.models.domain import RetrievedChunk
from vectorforge.retriever.reranker import BaseReranker

logger = logging.getLogger(__name__)

_COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
_DEFAULT_MODEL = "rerank-english-v3.0"
_REQUEST_TIMEOUT = 30.0


class CohereReranker(BaseReranker):
    """Re-ranker backed by the Cohere Rerank API.

    Uses ``httpx`` to call the Cohere ``/v2/rerank`` endpoint.
    No Cohere SDK dependency required.

    Args:
        api_key: Cohere API key.
        model: The Cohere rerank model to use.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        if not api_key:
            msg = "Cohere API key is required"
            raise ValueError(msg)
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    @property
    def reranker_name(self) -> str:
        """Return the reranker identifier."""
        return f"cohere:{self._model}"

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Re-rank chunks using the Cohere Rerank API.

        Args:
            query: The user query text.
            chunks: Initial set of retrieved chunks.
            top_k: Maximum number of results after re-ranking.

        Returns:
            Re-ranked RetrievedChunk list sorted by Cohere relevance score.

        Raises:
            RetrievalError: If the Cohere API call fails.
        """
        if not chunks:
            return []

        documents = [c.chunk.text for c in chunks]
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(chunks)),
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    _COHERE_RERANK_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            msg = f"Cohere rerank API returned {exc.response.status_code}"
            raise RetrievalError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Cohere rerank request failed: {exc}"
            raise RetrievalError(msg) from exc
        results_data = data.get("results", [])

        reranked: list[RetrievedChunk] = []
        for item in results_data:
            idx = item["index"]
            score = float(item["relevance_score"])
            original = chunks[idx]
            reranked.append(
                RetrievedChunk(
                    chunk=original.chunk,
                    score=score,
                    document_source=original.document_source,
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "Cohere reranked %d → %d chunks (model=%s)",
            len(chunks),
            len(reranked),
            self._model,
        )
        return reranked

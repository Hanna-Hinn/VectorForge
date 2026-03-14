"""Cohere embedding provider."""

from __future__ import annotations

import logging
import os

import httpx

from vectorforge.embedding.base import BaseEmbeddingProvider
from vectorforge.exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)

_MODEL_DIMENSIONS: dict[str, int] = {
    "embed-v4.0": 1024,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}

_BASE_URL = "https://api.cohere.com/v2"


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using the Cohere API.

    Args:
        api_key: Cohere API key. Falls back to VECTORFORGE_COHERE_API_KEY env var.
        model: Model identifier. Defaults to "embed-v4.0".
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-v4.0",
    ) -> None:
        resolved_key = api_key or os.environ.get("VECTORFORGE_COHERE_API_KEY", "")
        if not resolved_key:
            msg = "Cohere API key is required (set VECTORFORGE_COHERE_API_KEY)"
            raise ConfigurationError(msg)
        self._api_key = resolved_key
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def provider_name(self) -> str:
        """Return the provider name."""
        return "cohere"

    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model

    def dimensions(self) -> int:
        """Return the output dimensions for the configured model."""
        if self._model not in _MODEL_DIMENSIONS:
            msg = f"Unknown Cohere model: {self._model}"
            raise ConfigurationError(msg)
        return _MODEL_DIMENSIONS[self._model]

    def max_batch_size(self) -> int:
        """Return the max batch size for Cohere API."""
        return 96

    async def _call_api(
        self, texts: list[str], input_type: str = "search_document"
    ) -> list[list[float]]:
        """Call the Cohere embeddings API.

        Args:
            texts: Texts to embed.
            input_type: "search_document" for indexing, "search_query" for search.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the API call fails.
        """
        payload = {
            "model": self._model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        try:
            response = await self._client.post("/embed", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"Cohere API error {exc.response.status_code}: {exc.response.text}"
            raise EmbeddingError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Cohere API request failed: {exc}"
            raise EmbeddingError(msg) from exc

        data = response.json()
        embeddings: list[list[float]] = data["embeddings"]["float"]
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query using input_type='search_query'.

        Args:
            query: The query text.

        Returns:
            The query embedding vector.
        """
        result = await self._call_api([query], input_type="search_query")
        return result[0]

    async def validate_credentials(self) -> bool:
        """Test credentials by embedding a short string.

        Returns:
            True if API responds successfully, False otherwise.
        """
        try:
            await self._call_api(["test"], input_type="search_document")
        except EmbeddingError:
            return False
        return True

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

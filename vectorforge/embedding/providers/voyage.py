"""Voyage AI embedding provider."""

from __future__ import annotations

import logging
import os

import httpx

from vectorforge.embedding.base import BaseEmbeddingProvider
from vectorforge.exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)

_MODEL_DIMENSIONS: dict[str, int] = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
}

_DEFAULT_BASE_URL = "https://api.voyageai.com/v1"


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using the Voyage AI API.

    Args:
        api_key: Voyage AI API key. Falls back to VECTORFORGE_VOYAGE_API_KEY env var.
        model: Model identifier. Defaults to "voyage-3".
        base_url: API base URL. Defaults to Voyage AI production URL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-3",
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("VECTORFORGE_VOYAGE_API_KEY", "")
        if not resolved_key:
            msg = "Voyage AI API key is required (set VECTORFORGE_VOYAGE_API_KEY)"
            raise ConfigurationError(msg)
        self._api_key = resolved_key
        self._model = model
        self._base_url = base_url or os.environ.get(
            "VECTORFORGE_VOYAGE_BASE_URL", _DEFAULT_BASE_URL
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def provider_name(self) -> str:
        """Return the provider name."""
        return "voyage"

    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model

    def dimensions(self) -> int:
        """Return the output dimensions for the configured model."""
        if self._model not in _MODEL_DIMENSIONS:
            msg = f"Unknown Voyage model: {self._model}"
            raise ConfigurationError(msg)
        return _MODEL_DIMENSIONS[self._model]

    def max_batch_size(self) -> int:
        """Return the max batch size for Voyage API."""
        return 128

    async def _call_api(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        """Call the Voyage AI embeddings API.

        Args:
            texts: Texts to embed.
            input_type: "document" for indexing, "query" for search.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the API call fails.
        """
        payload = {
            "model": self._model,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
        }
        try:
            response = await self._client.post("/embeddings", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"Voyage API error {exc.response.status_code}: {exc.response.text}"
            raise EmbeddingError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Voyage API request failed: {exc}"
            raise EmbeddingError(msg) from exc

        data = response.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query using input_type='query'.

        Args:
            query: The query text.

        Returns:
            The query embedding vector.
        """
        result = await self._call_api([query], input_type="query")
        return result[0]

    async def validate_credentials(self) -> bool:
        """Test credentials by embedding a short string.

        Returns:
            True if API responds successfully, False otherwise.
        """
        try:
            await self._call_api(["test"], input_type="document")
        except EmbeddingError:
            return False
        return True

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

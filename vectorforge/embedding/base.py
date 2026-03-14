"""Base embedding provider ABC.

Defines the interface for all embedding providers with batching,
retry, and dimension validation.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from vectorforge.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Subclasses implement ``_call_api`` for their specific API and
    get batching, retry, and validation for free.
    """

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g. 'voyage', 'cohere')."""

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @abstractmethod
    def dimensions(self) -> int:
        """Return the output embedding dimensions for the configured model."""

    @abstractmethod
    def max_batch_size(self) -> int:
        """Return the maximum batch size for a single API call."""

    @abstractmethod
    async def _call_api(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        """Call the provider API to generate embeddings.

        Args:
            texts: List of text strings to embed.
            input_type: Hint for the provider (e.g. 'document' or 'query').

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the API call fails.
        """

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate that API credentials are functional.

        Returns:
            True if credentials are valid, False otherwise.
        """

    async def close(self) -> None:  # noqa: B027
        """Release any resources held by the provider.

        Subclasses with HTTP clients or other resources should
        override this to clean up.
        """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with automatic batching and retry.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If embedding fails after retries.
        """
        if not texts:
            return []

        batch_size = self.max_batch_size()
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embeddings: list[list[float]] = []

        for batch in batches:
            embeddings = await self._call_api_with_retry(batch)
            all_embeddings.extend(embeddings)

        if len(all_embeddings) != len(texts):
            msg = (
                f"Embedding count mismatch: expected {len(texts)}, "
                f"got {len(all_embeddings)}"
            )
            raise EmbeddingError(msg)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query text.

        Some providers differentiate between document and query embeddings.

        Args:
            query: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        result = await self._call_api_with_retry([query], input_type="query")
        return result[0]

    async def _call_api_with_retry(
        self,
        texts: list[str],
        input_type: str = "document",
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> list[list[float]]:
        """Call the API with exponential backoff retry.

        Args:
            texts: List of text strings.
            input_type: Provider hint for input type.
            max_retries: Maximum retry attempts.
            base_delay: Base delay in seconds for backoff.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If all retries are exhausted.
        """
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self._call_api(texts, input_type)
            except EmbeddingError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Embedding API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        msg = f"Provider {self.provider_name()} failed after {max_retries} retries: {last_error}"
        raise EmbeddingError(msg)

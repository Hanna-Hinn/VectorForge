"""LiteLLM embedding provider — generic proxy for 100+ embedding APIs.

Uses LiteLLM as a unified interface to OpenAI, Azure, Bedrock, Cohere,
and many other providers without requiring separate integrations for each.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from vectorforge.embedding.base import BaseEmbeddingProvider
from vectorforge.exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)


def _import_litellm() -> Any:
    """Lazily import litellm to keep it an optional dependency.

    Returns:
        The ``litellm`` module.

    Raises:
        ConfigurationError: If litellm is not installed.
    """
    try:
        import litellm  # type: ignore[import-not-found]

        return litellm
    except ImportError as exc:
        msg = "litellm package is required (pip install vectorforge[litellm])"
        raise ConfigurationError(msg) from exc


class LiteLLMEmbeddingProvider(BaseEmbeddingProvider):
    """Generic embedding provider backed by LiteLLM.

    LiteLLM provides a unified interface to 100+ embedding providers
    including OpenAI, Azure OpenAI, AWS Bedrock, Cohere, and more.
    Any model string supported by ``litellm.aembedding`` can be used.

    Args:
        model: LiteLLM model string (e.g. ``"text-embedding-3-small"``,
            ``"bedrock/amazon.titan-embed-text-v2:0"``).
        api_key: API key for the underlying provider.
            Falls back to ``VECTORFORGE_LITELLM_API_KEY`` env var.
        api_base: Optional custom API base URL.
        dimensions: Output embedding dimensions for the chosen model.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
        dimensions: int = 1536,
    ) -> None:
        _import_litellm()  # fail fast if not installed

        resolved_key = api_key or os.environ.get("VECTORFORGE_LITELLM_API_KEY", "")
        if not resolved_key:
            msg = (
                "API key is required for LiteLLM provider "
                "(set VECTORFORGE_LITELLM_API_KEY or pass api_key=)"
            )
            raise ConfigurationError(msg)

        self._model = model
        self._api_key = resolved_key
        self._api_base = api_base or os.environ.get("VECTORFORGE_LITELLM_API_BASE")
        self._dimensions = dimensions

    def provider_name(self) -> str:
        """Return the provider name."""
        return "litellm"

    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model

    def dimensions(self) -> int:
        """Return the output embedding dimensions."""
        return self._dimensions

    def max_batch_size(self) -> int:
        """Return the max batch size (provider-dependent, safe default)."""
        return 128

    async def _call_api(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        """Call the LiteLLM embedding API.

        Args:
            texts: Texts to embed.
            input_type: Hint for the provider (unused by most LiteLLM backends).

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the API call fails.
        """
        litellm = _import_litellm()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "api_key": self._api_key,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base

        try:
            response = await litellm.aembedding(**kwargs)
        except Exception as exc:
            msg = f"LiteLLM embedding call failed: {exc}"
            raise EmbeddingError(msg) from exc

        items = sorted(response.data, key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def validate_credentials(self) -> bool:
        """Test credentials by embedding a short string.

        Returns:
            True if the provider responds successfully, False otherwise.
        """
        try:
            await self._call_api(["test"], input_type="document")
        except EmbeddingError:
            return False
        return True

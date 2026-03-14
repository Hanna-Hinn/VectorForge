"""Embedding provider registry with auto-discovery."""

from __future__ import annotations

import logging
import os

from vectorforge.embedding.base import BaseEmbeddingProvider
from vectorforge.exceptions import ConfigurationError, DuplicateError

logger = logging.getLogger(__name__)


class EmbeddingProviderRegistry:
    """Registry of embedding providers with auto-discovery.

    Manages registered providers and selects a default.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseEmbeddingProvider] = {}
        self._default_provider: str = ""

    def register(self, provider: BaseEmbeddingProvider) -> None:
        """Register an embedding provider.

        Args:
            provider: The provider instance.

        Raises:
            DuplicateError: If a provider with this name is already registered.
        """
        name = provider.provider_name()
        if name in self._providers:
            msg = f"Embedding provider '{name}' is already registered"
            raise DuplicateError(msg)
        self._providers[name] = provider
        logger.info("Registered embedding provider: %s", name)

    def get(self, name: str) -> BaseEmbeddingProvider:
        """Get a provider by name.

        Args:
            name: The provider name.

        Returns:
            The registered BaseEmbeddingProvider.

        Raises:
            ConfigurationError: If the provider is not registered.
        """
        if name not in self._providers:
            msg = f"Embedding provider '{name}' not registered"
            raise ConfigurationError(msg)
        return self._providers[name]

    def get_default(self) -> BaseEmbeddingProvider:
        """Get the default embedding provider.

        Returns:
            The default BaseEmbeddingProvider.

        Raises:
            ConfigurationError: If no default is set or no providers registered.
        """
        if not self._default_provider:
            msg = "No default embedding provider configured"
            raise ConfigurationError(msg)
        return self.get(self._default_provider)

    def set_default(self, name: str) -> None:
        """Set the default provider by name.

        Args:
            name: The provider name to set as default.

        Raises:
            ConfigurationError: If the provider is not registered.
        """
        if name not in self._providers:
            msg = f"Cannot set default: provider '{name}' not registered"
            raise ConfigurationError(msg)
        self._default_provider = name
        logger.info("Default embedding provider set to: %s", name)

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            Sorted list of provider name strings.
        """
        return sorted(self._providers.keys())

    def auto_discover(self, default_provider: str = "voyage") -> None:
        """Auto-discover and register providers based on environment variables.

        Checks for API keys in environment and registers providers
        whose credentials are present.

        Args:
            default_provider: Preferred default provider name.
        """
        from vectorforge.embedding.providers.cohere import CohereEmbeddingProvider
        from vectorforge.embedding.providers.litellm import LiteLLMEmbeddingProvider
        from vectorforge.embedding.providers.voyage import VoyageEmbeddingProvider

        provider_map: list[tuple[str, type[BaseEmbeddingProvider], str]] = [
            ("voyage", VoyageEmbeddingProvider, "VECTORFORGE_VOYAGE_API_KEY"),
            ("cohere", CohereEmbeddingProvider, "VECTORFORGE_COHERE_API_KEY"),
            ("litellm", LiteLLMEmbeddingProvider, "VECTORFORGE_LITELLM_API_KEY"),
        ]

        for name, provider_cls, env_key in provider_map:
            api_key = os.environ.get(env_key)
            if api_key:
                try:
                    provider = provider_cls(api_key=api_key)  # type: ignore[call-arg]
                    self.register(provider)
                except Exception as exc:
                    logger.warning("Failed to register %s: %s", name, exc)
            else:
                logger.debug("Skipped embedding provider %s (no %s)", name, env_key)

        if not self._providers:
            logger.warning("No embedding providers discovered from environment")
            return

        if default_provider in self._providers:
            self.set_default(default_provider)
        else:
            first = next(iter(self._providers))
            self.set_default(first)
            logger.warning(
                "Default provider '%s' not available, using '%s'",
                default_provider,
                first,
            )

        logger.info(
            "Embedding registry ready: %d providers, default=%s",
            len(self._providers),
            self._default_provider,
        )

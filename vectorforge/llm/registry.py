"""LLM provider registry with auto-discovery."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from vectorforge.exceptions import ConfigurationError, DuplicateError
from vectorforge.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class BaseLLMProviderLookup(ABC):
    """Abstract lookup interface for LLM providers.

    High-level modules (e.g. QueryService) depend on this
    abstraction instead of the concrete LLMProviderRegistry.
    """

    @abstractmethod
    def get(self, name: str) -> BaseLLMProvider:
        """Get a provider by name."""

    @abstractmethod
    def get_default(self) -> BaseLLMProvider:
        """Get the default LLM provider."""


class LLMProviderRegistry(BaseLLMProviderLookup):
    """Registry of LLM providers with auto-discovery.

    Unlike the embedding registry, an empty LLM registry is allowed —
    VectorForge can still be used for ingestion + retrieval without
    LLM generation.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseLLMProvider] = {}
        self._default_provider: str = ""

    def register(self, provider: BaseLLMProvider) -> None:
        """Register an LLM provider.

        Args:
            provider: The provider instance.

        Raises:
            DuplicateError: If a provider with this name is already registered.
        """
        name = provider.provider_name()
        if name in self._providers:
            msg = f"LLM provider '{name}' is already registered"
            raise DuplicateError(msg)
        self._providers[name] = provider
        logger.info("Registered LLM provider: %s", name)

    def get(self, name: str) -> BaseLLMProvider:
        """Get a provider by name.

        Args:
            name: The provider name.

        Returns:
            The registered BaseLLMProvider.

        Raises:
            ConfigurationError: If the provider is not registered.
        """
        if name not in self._providers:
            msg = f"LLM provider '{name}' not registered"
            raise ConfigurationError(msg)
        return self._providers[name]

    def get_default(self) -> BaseLLMProvider:
        """Get the default LLM provider.

        Returns:
            The default BaseLLMProvider.

        Raises:
            ConfigurationError: If no default is set or no providers registered.
        """
        if not self._default_provider:
            msg = "No default LLM provider configured"
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
            msg = f"Cannot set default: LLM provider '{name}' not registered"
            raise ConfigurationError(msg)
        self._default_provider = name
        logger.info("Default LLM provider set to: %s", name)

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            Sorted list of provider name strings.
        """
        return sorted(self._providers.keys())

    def auto_discover(self, default_provider: str = "openai") -> None:
        """Auto-discover and register LLM providers from environment.

        Checks for API keys / base URLs in environment variables and
        registers providers whose credentials are present.

        Args:
            default_provider: Preferred default provider name.
        """
        from vectorforge.llm.providers.anthropic import AnthropicProvider
        from vectorforge.llm.providers.litellm import LiteLLMProvider
        from vectorforge.llm.providers.openai import OpenAIProvider

        provider_map: list[tuple[str, type[BaseLLMProvider], str]] = [
            ("openai", OpenAIProvider, "VECTORFORGE_OPENAI_API_KEY"),
            ("anthropic", AnthropicProvider, "VECTORFORGE_ANTHROPIC_API_KEY"),
            ("litellm", LiteLLMProvider, "VECTORFORGE_LITELLM_API_KEY"),
        ]

        for name, provider_cls, env_key in provider_map:
            api_key = os.environ.get(env_key)
            if api_key:
                try:
                    provider = provider_cls(api_key=api_key)  # type: ignore[call-arg]
                    self.register(provider)
                except Exception as exc:
                    logger.warning("Failed to register LLM provider %s: %s", name, exc)
            else:
                logger.debug("Skipped LLM provider %s (no %s)", name, env_key)

        if not self._providers:
            logger.warning("No LLM providers discovered — generation disabled")
            return

        if default_provider in self._providers:
            self.set_default(default_provider)
        else:
            first = sorted(self._providers.keys())[0]
            self.set_default(first)
            logger.warning(
                "Default LLM provider '%s' not available, using '%s'",
                default_provider,
                first,
            )

        logger.info(
            "LLM registry ready: %d providers, default=%s",
            len(self._providers),
            self._default_provider,
        )

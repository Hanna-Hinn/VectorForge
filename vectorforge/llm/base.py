"""Base LLM provider ABC.

Defines the interface for all LLM providers with retry logic
and streaming support.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from vectorforge.exceptions import LLMError
from vectorforge.llm.types import LLMRequestConfig, LLMResponse

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses implement ``_call_api`` and ``_call_api_stream`` for their
    specific API. The base class provides retry logic and default config
    handling.
    """

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. ``openai``, ``anthropic``)."""

    @abstractmethod
    def default_model(self) -> str:
        """Return the default model identifier for this provider."""

    @abstractmethod
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> LLMResponse:
        """Execute a non-streaming generation request.

        Args:
            messages: Conversation messages in ``{"role": ..., "content": ...}`` format.
            config: Generation configuration.

        Returns:
            The provider's response.

        Raises:
            LLMError: If the API call fails.
        """

    @abstractmethod
    async def _call_api_stream(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> AsyncIterator[str]:
        """Execute a streaming generation request.

        Args:
            messages: Conversation messages.
            config: Generation configuration.

        Yields:
            Token strings as they arrive.

        Raises:
            LLMError: If the API call fails.
        """
        yield ""  # pragma: no cover — abstract placeholder

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate that API credentials are functional.

        Returns:
            True if credentials are valid.
        """

    async def close(self) -> None:  # noqa: B027
        """Release resources held by the provider.

        Subclasses with HTTP clients should override this.
        """

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig | None = None,
    ) -> LLMResponse:
        """Generate a response with automatic retry.

        Args:
            messages: Conversation messages.
            config: Optional generation configuration.

        Returns:
            The LLM response.

        Raises:
            LLMError: If generation fails after retries.
        """
        cfg = config or LLMRequestConfig(model=self.default_model())
        if not cfg.model:
            cfg = cfg.model_copy(update={"model": self.default_model()})

        start = time.perf_counter()
        response = await self._call_api_with_retry(messages, cfg)
        latency = (time.perf_counter() - start) * 1000
        response.latency_ms = latency
        return response

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response.

        Args:
            messages: Conversation messages.
            config: Optional generation configuration.

        Yields:
            Token strings as they arrive.

        Raises:
            LLMError: If the streaming call fails.
        """
        cfg = config or LLMRequestConfig(model=self.default_model())
        if not cfg.model:
            cfg = cfg.model_copy(update={"model": self.default_model()})

        async for token in self._call_api_stream(messages, cfg):
            yield token

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _call_api_with_retry(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> LLMResponse:
        """Call the API with exponential backoff retry.

        Args:
            messages: Conversation messages.
            config: Generation configuration.
            max_retries: Maximum retry attempts.
            base_delay: Base delay in seconds.

        Returns:
            The LLM response.

        Raises:
            LLMError: If all retries are exhausted.
        """
        last_error: LLMError | None = None
        for attempt in range(max_retries):
            try:
                return await self._call_api(messages, config)
            except LLMError as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "LLM API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        msg = (
            f"Provider {self.provider_name()} failed after "
            f"{max_retries} retries: {last_error}"
        )
        raise LLMError(msg) from last_error

"""LiteLLM LLM provider — generic proxy for 100+ LLM APIs.

Uses LiteLLM as a unified interface to OpenAI, Azure, Bedrock, Cohere,
Anthropic, and many other providers without requiring separate
integrations for each.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from vectorforge.exceptions import ConfigurationError, LLMError
from vectorforge.llm.base import BaseLLMProvider
from vectorforge.llm.types import LLMRequestConfig, LLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o"


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


class LiteLLMProvider(BaseLLMProvider):
    """Generic LLM provider backed by LiteLLM.

    LiteLLM provides a unified interface to 100+ LLM providers
    including OpenAI, Azure OpenAI, AWS Bedrock, Cohere, Anthropic, and
    more.  Any model string supported by ``litellm.acompletion`` can be
    used.

    Args:
        model: LiteLLM model string (e.g. ``"gpt-4o"``,
            ``"bedrock/anthropic.claude-3-sonnet-20240229-v1:0"``).
        api_key: API key for the underlying provider.
            Falls back to ``VECTORFORGE_LITELLM_API_KEY`` env var.
        api_base: Optional custom API base URL.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        api_base: str | None = None,
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

    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "litellm"

    def default_model(self) -> str:
        """Return the default model name."""
        return self._model

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the keyword arguments for ``litellm.acompletion``.

        Args:
            messages: Conversation messages.
            config: Generation configuration.
            stream: Whether to enable streaming.

        Returns:
            Keyword arguments dict.
        """
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "api_key": self._api_key,
            "stream": stream,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences
        return kwargs

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> LLMResponse:
        """Execute a non-streaming chat completion via LiteLLM.

        Args:
            messages: Conversation messages.
            config: Generation configuration.

        Returns:
            The LLM response.

        Raises:
            LLMError: If the API call fails.
        """
        litellm = _import_litellm()
        kwargs = self._build_kwargs(messages, config)

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            msg = f"LiteLLM completion call failed: {exc}"
            raise LLMError(msg) from exc

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or config.model,
            provider="litellm",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    async def _call_api_stream(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> AsyncIterator[str]:
        """Execute a streaming chat completion via LiteLLM.

        Args:
            messages: Conversation messages.
            config: Generation configuration.

        Yields:
            Token strings as they arrive.

        Raises:
            LLMError: If the streaming call fails.
        """
        litellm = _import_litellm()
        kwargs = self._build_kwargs(messages, config, stream=True)

        try:
            response = await litellm.acompletion(**kwargs)
            async for chunk in response:
                delta = chunk.choices[0].delta
                token = delta.content if delta else None
                if token:
                    yield token
        except LLMError:
            raise
        except Exception as exc:
            msg = f"LiteLLM streaming call failed: {exc}"
            raise LLMError(msg) from exc

    async def validate_credentials(self) -> bool:
        """Test credentials by sending a minimal completion request.

        Returns:
            True if the provider responds successfully, False otherwise.
        """
        litellm = _import_litellm()
        try:
            await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                api_key=self._api_key,
            )
        except Exception:
            return False
        return True

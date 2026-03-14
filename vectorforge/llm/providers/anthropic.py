"""Anthropic LLM provider using httpx."""

from __future__ import annotations

import json as _json
import logging
import os
from collections.abc import AsyncIterator

import httpx

from vectorforge.exceptions import ConfigurationError, LLMError
from vectorforge.llm.base import BaseLLMProvider
from vectorforge.llm.types import LLMRequestConfig, LLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicProvider(BaseLLMProvider):
    """LLM provider for Anthropic Claude models.

    Handles the Anthropic-specific message format where the system
    prompt is a separate parameter (not in the messages array).

    Args:
        api_key: Anthropic API key. Falls back to ``VECTORFORGE_ANTHROPIC_API_KEY``.
        model: Default model identifier.
        base_url: API base URL override.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("VECTORFORGE_ANTHROPIC_API_KEY", "")
        if not resolved_key:
            msg = "Anthropic API key is required (set VECTORFORGE_ANTHROPIC_API_KEY)"
            raise ConfigurationError(msg)
        self._api_key = resolved_key
        self._model = model
        self._base_url = base_url or os.environ.get(
            "VECTORFORGE_ANTHROPIC_BASE_URL", _DEFAULT_BASE_URL,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": _ANTHROPIC_API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "anthropic"

    def default_model(self) -> str:
        """Return the default model name."""
        return self._model

    # ------------------------------------------------------------------
    # Message transformation
    # ------------------------------------------------------------------

    @staticmethod
    def _separate_system_message(
        messages: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract and separate the system message from conversation messages.

        Anthropic's API requires the system prompt as a top-level field,
        not inside the messages array.

        Args:
            messages: Standard ``{"role": ..., "content": ...}`` messages.

        Returns:
            Tuple of (system_text, remaining_messages).
        """
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)
            else:
                conversation.append(msg)
        return "\n\n".join(system_parts), conversation

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> LLMResponse:
        """Execute a non-streaming messages request.

        Args:
            messages: Conversation messages.
            config: Generation configuration.

        Returns:
            The LLM response.

        Raises:
            LLMError: If the API call fails.
        """
        system_text, conversation = self._separate_system_message(messages)

        payload: dict[str, object] = {
            "model": config.model,
            "messages": conversation,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        if system_text:
            payload["system"] = system_text

        try:
            resp = await self._client.post("/v1/messages", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"Anthropic API error {exc.response.status_code}: {exc.response.text}"
            raise LLMError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Anthropic request failed: {exc}"
            raise LLMError(msg) from exc

        data = resp.json()
        content_blocks = data.get("content", [])
        content = content_blocks[0].get("text", "") if content_blocks else ""
        usage = data.get("usage", {})

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return LLMResponse(
            content=content,
            model=data.get("model", config.model),
            provider="anthropic",
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    async def _call_api_stream(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> AsyncIterator[str]:
        """Execute a streaming messages request.

        Args:
            messages: Conversation messages.
            config: Generation configuration.

        Yields:
            Token strings as they arrive.

        Raises:
            LLMError: If the streaming call fails.
        """
        system_text, conversation = self._separate_system_message(messages)

        payload: dict[str, object] = {
            "model": config.model,
            "messages": conversation,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
        }
        if system_text:
            payload["system"] = system_text

        try:
            async with self._client.stream(
                "POST", "/v1/messages", json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    chunk = _json.loads(data_str)
                    event_type = chunk.get("type", "")
                    if event_type == "content_block_delta":
                        delta = chunk.get("delta", {})
                        token = delta.get("text")
                        if token:
                            yield token
                    elif event_type == "message_stop":
                        break
        except httpx.HTTPStatusError as exc:
            msg = f"Anthropic streaming error {exc.response.status_code}"
            raise LLMError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"Anthropic streaming request failed: {exc}"
            raise LLMError(msg) from exc

    async def validate_credentials(self) -> bool:
        """Validate credentials with a minimal messages request.

        Returns:
            True if the API key is valid.
        """
        try:
            resp = await self._client.post(
                "/v1/messages",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                },
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Close the httpx client."""
        await self._client.aclose()

    async def __aenter__(self) -> AnthropicProvider:
        """Enter async context manager."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit async context manager, closing the client."""
        await self.close()

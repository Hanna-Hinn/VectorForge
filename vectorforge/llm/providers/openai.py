"""OpenAI LLM provider using httpx."""

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

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(BaseLLMProvider):
    """LLM provider for OpenAI chat completions.

    Uses httpx directly (matching the project convention for external APIs)
    instead of the ``openai`` SDK.

    Args:
        api_key: OpenAI API key. Falls back to ``VECTORFORGE_OPENAI_API_KEY``.
        model: Default model identifier.
        base_url: API base URL override.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("VECTORFORGE_OPENAI_API_KEY", "")
        if not resolved_key:
            msg = "OpenAI API key is required (set VECTORFORGE_OPENAI_API_KEY)"
            raise ConfigurationError(msg)
        self._api_key = resolved_key
        self._model = model
        self._base_url = base_url or os.environ.get(
            "VECTORFORGE_OPENAI_BASE_URL", _DEFAULT_BASE_URL,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "openai"

    def default_model(self) -> str:
        """Return the default model name."""
        return self._model

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> LLMResponse:
        """Execute a non-streaming chat completion.

        Args:
            messages: Conversation messages.
            config: Generation configuration.

        Returns:
            The LLM response.

        Raises:
            LLMError: If the API call fails.
        """
        payload: dict[str, object] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"OpenAI API error {exc.response.status_code}: {exc.response.text}"
            raise LLMError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"OpenAI request failed: {exc}"
            raise LLMError(msg) from exc

        data = resp.json()
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=data.get("model", config.model),
            provider="openai",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    async def _call_api_stream(
        self,
        messages: list[dict[str, str]],
        config: LLMRequestConfig,
    ) -> AsyncIterator[str]:
        """Execute a streaming chat completion.

        Args:
            messages: Conversation messages.
            config: Generation configuration.

        Yields:
            Token strings as they arrive.

        Raises:
            LLMError: If the streaming call fails.
        """
        payload: dict[str, object] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "stream": True,
        }
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = _json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        yield token
        except httpx.HTTPStatusError as exc:
            msg = f"OpenAI streaming error {exc.response.status_code}"
            raise LLMError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"OpenAI streaming request failed: {exc}"
            raise LLMError(msg) from exc

    async def validate_credentials(self) -> bool:
        """Validate credentials by listing models.

        Returns:
            True if the API key is valid.
        """
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        """Close the httpx client."""
        await self._client.aclose()

    async def __aenter__(self) -> OpenAIProvider:
        """Enter async context manager."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit async context manager, closing the client."""
        await self.close()

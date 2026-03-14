"""Unit tests for the LLM module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vectorforge.exceptions import ConfigurationError, DuplicateError, LLMError
from vectorforge.llm.base import BaseLLMProvider
from vectorforge.llm.providers.anthropic import AnthropicProvider
from vectorforge.llm.providers.openai import OpenAIProvider
from vectorforge.llm.registry import LLMProviderRegistry
from vectorforge.llm.types import LLMRequestConfig, LLMResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _openai_response(content: str = "Hello") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}}],
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _anthropic_response(content: str = "Hello") -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": content}],
        "model": "claude-sonnet-4-20250514",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
        },
    }


def _mock_httpx_response(data: dict[str, object], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=data,
        request=httpx.Request("POST", "https://test.com"),
    )


# ---------------------------------------------------------------------------
# LLMRequestConfig tests
# ---------------------------------------------------------------------------


class TestLLMRequestConfig:
    """Tests for LLMRequestConfig."""

    def test_defaults(self) -> None:
        config = LLMRequestConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.top_p == 1.0

    def test_custom_values(self) -> None:
        config = LLMRequestConfig(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=512,
            stop_sequences=["STOP"],
        )
        assert config.model == "gpt-4o"
        assert config.stop_sequences == ["STOP"]


# ---------------------------------------------------------------------------
# LLMResponse tests
# ---------------------------------------------------------------------------


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_creation(self) -> None:
        resp = LLMResponse(
            content="Hi",
            model="gpt-4o",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        assert resp.content == "Hi"
        assert resp.total_tokens == 15


# ---------------------------------------------------------------------------
# OpenAIProvider tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Tests for the OpenAI LLM provider."""

    def test_init_requires_api_key(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigurationError, match="API key"),
        ):
            OpenAIProvider(api_key="")

    def test_init_with_key(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        assert provider.provider_name() == "openai"
        assert provider.default_model() == "gpt-4o"

    async def test_call_api_success(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        mock_resp = _mock_httpx_response(_openai_response("World"))
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        messages = [{"role": "user", "content": "Hello"}]
        config = LLMRequestConfig(model="gpt-4o")
        result = await provider._call_api(messages, config)

        assert result.content == "World"
        assert result.provider == "openai"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    async def test_call_api_http_error(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        error_resp = httpx.Response(
            status_code=401,
            text="Unauthorized",
            request=httpx.Request("POST", "https://test.com"),
        )
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Unauthorized", request=error_resp.request, response=error_resp,
            ),
        )

        with pytest.raises(LLMError, match="401"):
            await provider._call_api(
                [{"role": "user", "content": "Hi"}],
                LLMRequestConfig(model="gpt-4o"),
            )

    async def test_generate_applies_default_model(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        mock_resp = _mock_httpx_response(_openai_response())
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.generate([{"role": "user", "content": "Hi"}])

        assert result.content == "Hello"
        assert result.latency_ms > 0

    async def test_validate_credentials_success(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        mock_resp = httpx.Response(
            status_code=200,
            json={"data": []},
            request=httpx.Request("GET", "https://test.com"),
        )
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(return_value=mock_resp)

        assert await provider.validate_credentials() is True

    async def test_validate_credentials_failure(self) -> None:
        provider = OpenAIProvider(api_key="sk-test")
        provider._client = AsyncMock()
        provider._client.get = AsyncMock(side_effect=httpx.ConnectError("fail"))

        assert await provider.validate_credentials() is False


# ---------------------------------------------------------------------------
# AnthropicProvider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Tests for the Anthropic LLM provider."""

    def test_init_requires_api_key(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ConfigurationError, match="API key"),
        ):
            AnthropicProvider(api_key="")

    def test_init_with_key(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider.provider_name() == "anthropic"
        assert "claude" in provider.default_model()

    def test_separate_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        system, convo = AnthropicProvider._separate_system_message(messages)
        assert system == "You are helpful"
        assert len(convo) == 1
        assert convo[0]["role"] == "user"

    def test_separate_system_message_no_system(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        system, convo = AnthropicProvider._separate_system_message(messages)
        assert system == ""
        assert len(convo) == 1

    def test_separate_system_message_multiple(self) -> None:
        """Multiple system messages are concatenated."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
        ]
        system, convo = AnthropicProvider._separate_system_message(messages)
        assert system == "Be helpful\n\nBe concise"
        assert len(convo) == 1
        assert convo[0]["role"] == "user"

    async def test_call_api_success(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_resp = _mock_httpx_response(_anthropic_response("Bonjour"))
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        messages = [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Hi"},
        ]
        config = LLMRequestConfig(model="claude-sonnet-4-20250514")
        result = await provider._call_api(messages, config)

        assert result.content == "Bonjour"
        assert result.provider == "anthropic"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15

    async def test_call_api_http_error(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        error_resp = httpx.Response(
            status_code=429,
            text="Rate limited",
            request=httpx.Request("POST", "https://test.com"),
        )
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Rate limited", request=error_resp.request, response=error_resp,
            ),
        )

        with pytest.raises(LLMError, match="429"):
            await provider._call_api(
                [{"role": "user", "content": "Hi"}],
                LLMRequestConfig(model="claude-sonnet-4-20250514"),
            )

    async def test_generate_applies_default_model(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_resp = _mock_httpx_response(_anthropic_response("OK"))
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.generate([{"role": "user", "content": "Hi"}])

        assert result.content == "OK"
        assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# LLMProviderRegistry tests
# ---------------------------------------------------------------------------


class TestLLMProviderRegistry:
    """Tests for the LLM provider registry."""

    def _make_mock_provider(self, name: str = "test") -> BaseLLMProvider:
        provider = MagicMock(spec=BaseLLMProvider)
        provider.provider_name.return_value = name
        return provider

    def test_register_and_get(self) -> None:
        registry = LLMProviderRegistry()
        provider = self._make_mock_provider("openai")
        registry.register(provider)

        assert registry.get("openai") is provider

    def test_register_duplicate_raises(self) -> None:
        registry = LLMProviderRegistry()
        registry.register(self._make_mock_provider("openai"))

        with pytest.raises(DuplicateError, match="already registered"):
            registry.register(self._make_mock_provider("openai"))

    def test_get_missing_raises(self) -> None:
        registry = LLMProviderRegistry()

        with pytest.raises(ConfigurationError, match="not registered"):
            registry.get("nonexistent")

    def test_set_default_and_get_default(self) -> None:
        registry = LLMProviderRegistry()
        provider = self._make_mock_provider("openai")
        registry.register(provider)
        registry.set_default("openai")

        assert registry.get_default() is provider

    def test_get_default_no_default_raises(self) -> None:
        registry = LLMProviderRegistry()

        with pytest.raises(ConfigurationError, match="No default"):
            registry.get_default()

    def test_set_default_missing_raises(self) -> None:
        registry = LLMProviderRegistry()

        with pytest.raises(ConfigurationError, match="not registered"):
            registry.set_default("missing")

    def test_list_providers(self) -> None:
        registry = LLMProviderRegistry()
        registry.register(self._make_mock_provider("openai"))
        registry.register(self._make_mock_provider("anthropic"))

        assert registry.list_providers() == ["anthropic", "openai"]

    def test_auto_discover_with_keys(self) -> None:
        registry = LLMProviderRegistry()
        env = {
            "VECTORFORGE_OPENAI_API_KEY": "sk-test",
            "VECTORFORGE_ANTHROPIC_API_KEY": "sk-ant-test",
            "VECTORFORGE_LITELLM_API_KEY": "",
        }
        with patch.dict("os.environ", env, clear=False):
            registry.auto_discover()

        assert "openai" in registry.list_providers()
        assert "anthropic" in registry.list_providers()

    def test_auto_discover_no_keys(self) -> None:
        registry = LLMProviderRegistry()
        env_clear = {
            "VECTORFORGE_OPENAI_API_KEY": "",
            "VECTORFORGE_ANTHROPIC_API_KEY": "",
            "VECTORFORGE_LITELLM_API_KEY": "",
        }
        with patch.dict("os.environ", env_clear, clear=False):
            registry.auto_discover()

        assert registry.list_providers() == []

    def test_auto_discover_fallback_default(self) -> None:
        registry = LLMProviderRegistry()
        env = {
            "VECTORFORGE_ANTHROPIC_API_KEY": "sk-ant-test",
            "VECTORFORGE_LITELLM_API_KEY": "",
        }
        with patch.dict("os.environ", env, clear=False):
            registry.auto_discover(default_provider="openai")

        # OpenAI not available, should fallback to anthropic
        assert registry.get_default().provider_name() == "anthropic"

    def test_auto_discover_litellm_with_key(self) -> None:
        fake_litellm = MagicMock()
        env = {
            "VECTORFORGE_OPENAI_API_KEY": "",
            "VECTORFORGE_ANTHROPIC_API_KEY": "",
            "VECTORFORGE_LITELLM_API_KEY": "sk-lite-test",
        }
        with (
            patch.dict("sys.modules", {"litellm": fake_litellm}),
            patch.dict("os.environ", env, clear=False),
        ):
            registry = LLMProviderRegistry()
            registry.auto_discover()

        assert "litellm" in registry.list_providers()


# ---------------------------------------------------------------------------
# LiteLLMProvider tests
# ---------------------------------------------------------------------------


class TestLiteLLMProvider:
    """Tests for the LiteLLM LLM provider with mocked litellm module."""

    def test_missing_api_key_raises(self) -> None:
        fake_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            with pytest.raises(ConfigurationError, match="API key is required"):
                LiteLLMProvider(api_key="")

    def test_provider_name(self) -> None:
        fake_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            provider = LiteLLMProvider(api_key="sk-test")
            assert provider.provider_name() == "litellm"
            assert provider.default_model() == "gpt-4o"

    def test_custom_model(self) -> None:
        fake_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            provider = LiteLLMProvider(
                api_key="sk-test", model="bedrock/claude-3-sonnet",
            )
            assert provider.default_model() == "bedrock/claude-3-sonnet"

    async def test_call_api_success(self) -> None:
        fake_litellm = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"
        fake_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            provider = LiteLLMProvider(api_key="sk-test")
            messages = [{"role": "user", "content": "Hello"}]
            config = LLMRequestConfig(model="gpt-4o")
            result = await provider._call_api(messages, config)

            assert result.content == "Hello world"
            assert result.provider == "litellm"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 5
            assert result.total_tokens == 15

    async def test_call_api_error_raises(self) -> None:
        fake_litellm = MagicMock()
        fake_litellm.acompletion = AsyncMock(
            side_effect=RuntimeError("API down"),
        )

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            provider = LiteLLMProvider(api_key="sk-test")
            with pytest.raises(LLMError, match="LiteLLM completion call failed"):
                await provider._call_api(
                    [{"role": "user", "content": "Hi"}],
                    LLMRequestConfig(model="gpt-4o"),
                )

    async def test_call_api_stream_yields_tokens(self) -> None:
        fake_litellm = MagicMock()

        async def _fake_stream(**_kwargs: object):  # type: ignore[no-untyped-def]
            for text in ["Hello", " ", "world"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                yield chunk

        fake_litellm.acompletion = AsyncMock(
            return_value=_fake_stream(),
        )

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            provider = LiteLLMProvider(api_key="sk-test")
            config = LLMRequestConfig(model="gpt-4o")
            tokens: list[str] = []
            async for token in provider._call_api_stream(
                [{"role": "user", "content": "Hi"}], config,
            ):
                tokens.append(token)

            assert tokens == ["Hello", " ", "world"]

    async def test_generate_applies_default_model(self) -> None:
        fake_litellm = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "OK"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 1
        mock_usage.total_tokens = 6
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"
        fake_litellm.acompletion = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            provider = LiteLLMProvider(api_key="sk-test")
            result = await provider.generate(
                [{"role": "user", "content": "Hi"}],
            )

            assert result.content == "OK"
            assert result.latency_ms > 0

    async def test_validate_credentials_ok(self) -> None:
        fake_litellm = MagicMock()
        fake_litellm.acompletion = AsyncMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            provider = LiteLLMProvider(api_key="sk-test")
            assert await provider.validate_credentials() is True

    async def test_validate_credentials_fail(self) -> None:
        fake_litellm = MagicMock()
        fake_litellm.acompletion = AsyncMock(
            side_effect=RuntimeError("bad key"),
        )

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.llm.providers.litellm import LiteLLMProvider

            provider = LiteLLMProvider(api_key="sk-test")
            assert await provider.validate_credentials() is False

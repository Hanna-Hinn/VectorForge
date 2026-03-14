"""Unit tests for VectorForge embedding providers (mocked API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vectorforge.embedding.base import BaseEmbeddingProvider
from vectorforge.embedding.registry import EmbeddingProviderRegistry
from vectorforge.exceptions import ConfigurationError, EmbeddingError

# ---------------------------------------------------------------------------
# Fake provider for testing base class logic
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Test double that returns fixed-dimension vectors."""

    def __init__(self, dim: int = 4, fail_after: int = -1) -> None:
        self._dim = dim
        self._call_count = 0
        self._fail_after = fail_after

    def provider_name(self) -> str:
        return "fake"

    def model_name(self) -> str:
        return "fake-model"

    def dimensions(self) -> int:
        return self._dim

    def max_batch_size(self) -> int:
        return 3

    async def _call_api(
        self, texts: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        self._call_count += 1
        if 0 <= self._fail_after < self._call_count:
            msg = "Simulated API failure"
            raise RuntimeError(msg)
        return [[0.1] * self._dim for _ in texts]

    async def validate_credentials(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# BaseEmbeddingProvider tests (via FakeEmbeddingProvider)
# ---------------------------------------------------------------------------


class TestBaseEmbeddingProvider:
    """Tests for base embedding provider batching and retry."""

    @pytest.mark.asyncio
    async def test_embed_empty_list(self) -> None:
        provider = FakeEmbeddingProvider()
        result = await provider.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_single_text(self) -> None:
        provider = FakeEmbeddingProvider(dim=4)
        result = await provider.embed(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_embed_batching(self) -> None:
        """Texts exceeding max_batch_size are split into batches."""
        provider = FakeEmbeddingProvider(dim=4)
        texts = [f"text_{i}" for i in range(7)]
        result = await provider.embed(texts)
        assert len(result) == 7
        # max_batch_size=3 → 3 API calls (3 + 3 + 1)
        assert provider._call_count == 3

    @pytest.mark.asyncio
    async def test_embed_query(self) -> None:
        provider = FakeEmbeddingProvider(dim=8)
        result = await provider.embed_query("search query")
        assert len(result) == 8

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self) -> None:
        """Retry with backoff on transient (non-EmbeddingError) exceptions."""

        class AlwaysFailProvider(FakeEmbeddingProvider):
            async def _call_api(
                self, texts: list[str], input_type: str = "document"
            ) -> list[list[float]]:
                raise RuntimeError("transient failure")

        provider = AlwaysFailProvider(dim=4)
        with (
            patch("vectorforge.embedding.base.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(EmbeddingError, match="failed after"),
        ):
            await provider._call_api_with_retry(["test"], max_retries=2, base_delay=0.01)

    @pytest.mark.asyncio
    async def test_embedding_error_not_retried(self) -> None:
        """EmbeddingError is raised immediately without retry."""

        class FailProvider(FakeEmbeddingProvider):
            async def _call_api(
                self, texts: list[str], input_type: str = "document"
            ) -> list[list[float]]:
                raise EmbeddingError("permanent failure")

        provider = FailProvider()
        with pytest.raises(EmbeddingError, match="permanent failure"):
            await provider.embed(["test"])


# ---------------------------------------------------------------------------
# EmbeddingProviderRegistry tests
# ---------------------------------------------------------------------------


class TestEmbeddingProviderRegistry:
    """Tests for the embedding provider registry."""

    def test_register_and_get(self) -> None:
        registry = EmbeddingProviderRegistry()
        provider = FakeEmbeddingProvider()
        registry.register(provider)
        assert registry.get("fake") is provider

    def test_get_unknown_raises(self) -> None:
        registry = EmbeddingProviderRegistry()
        with pytest.raises(ConfigurationError, match="not registered"):
            registry.get("fake")

    def test_set_and_get_default(self) -> None:
        registry = EmbeddingProviderRegistry()
        provider = FakeEmbeddingProvider()
        registry.register(provider)
        registry.set_default("fake")
        assert registry.get_default() is provider

    def test_get_default_fallback(self) -> None:
        """get_default returns default after explicitly setting it."""
        registry = EmbeddingProviderRegistry()
        provider = FakeEmbeddingProvider()
        registry.register(provider)
        registry.set_default("fake")
        assert registry.get_default() is provider

    def test_get_default_empty_raises(self) -> None:
        registry = EmbeddingProviderRegistry()
        with pytest.raises(ConfigurationError, match="No default"):
            registry.get_default()

    def test_list_providers(self) -> None:
        registry = EmbeddingProviderRegistry()
        registry.register(FakeEmbeddingProvider())
        assert "fake" in registry.list_providers()


# ---------------------------------------------------------------------------
# LiteLLM provider tests (mocked)
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingProvider:
    """Tests for the LiteLLM embedding provider with mocked litellm module."""

    def test_missing_api_key_raises(self) -> None:
        from unittest.mock import MagicMock

        fake_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.embedding.providers.litellm import LiteLLMEmbeddingProvider

            with pytest.raises(ConfigurationError, match="API key is required"):
                LiteLLMEmbeddingProvider(api_key="")

    def test_provider_name(self) -> None:
        from unittest.mock import MagicMock

        fake_litellm = MagicMock()
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.embedding.providers.litellm import LiteLLMEmbeddingProvider

            provider = LiteLLMEmbeddingProvider(api_key="sk-test")
            assert provider.provider_name() == "litellm"
            assert provider.model_name() == "text-embedding-3-small"
            assert provider.dimensions() == 1536
            assert provider.max_batch_size() == 128

    @pytest.mark.asyncio
    async def test_embed_calls_litellm(self) -> None:
        from unittest.mock import MagicMock

        fake_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"index": 1, "embedding": [0.4, 0.5, 0.6]},
        ]
        fake_litellm.aembedding = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.embedding.providers.litellm import LiteLLMEmbeddingProvider

            provider = LiteLLMEmbeddingProvider(
                api_key="sk-test", model="test-model", dimensions=3
            )
            result = await provider._call_api(["hello", "world"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_api_error_raises(self) -> None:
        from unittest.mock import MagicMock

        fake_litellm = MagicMock()
        fake_litellm.aembedding = AsyncMock(side_effect=RuntimeError("API down"))

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.embedding.providers.litellm import LiteLLMEmbeddingProvider

            provider = LiteLLMEmbeddingProvider(api_key="sk-test")
            with pytest.raises(EmbeddingError, match="LiteLLM embedding call failed"):
                await provider._call_api(["hello"])

    @pytest.mark.asyncio
    async def test_validate_credentials_ok(self) -> None:
        from unittest.mock import MagicMock

        fake_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"index": 0, "embedding": [0.1]}]
        fake_litellm.aembedding = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.embedding.providers.litellm import LiteLLMEmbeddingProvider

            provider = LiteLLMEmbeddingProvider(api_key="sk-test", dimensions=1)
            assert await provider.validate_credentials() is True

    @pytest.mark.asyncio
    async def test_validate_credentials_fail(self) -> None:
        from unittest.mock import MagicMock

        fake_litellm = MagicMock()
        fake_litellm.aembedding = AsyncMock(
            side_effect=EmbeddingError("bad key")
        )

        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            from vectorforge.embedding.providers.litellm import LiteLLMEmbeddingProvider

            provider = LiteLLMEmbeddingProvider(api_key="sk-test")
            assert await provider.validate_credentials() is False

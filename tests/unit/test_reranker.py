"""Unit tests for the reranker module (base, cohere, cross-encoder)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vectorforge.exceptions import RetrievalError
from vectorforge.models.domain import Chunk, RetrievedChunk
from vectorforge.retriever.reranker import BaseReranker
from vectorforge.retriever.rerankers.cohere import CohereReranker
from vectorforge.retriever.rerankers.cross_encoder import CrossEncoderReranker

_COHERE_CLIENT_PATH = (
    "vectorforge.retriever.rerankers.cohere.httpx.AsyncClient"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(index: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=f"Chunk text {index}",
        index=index,
        start_char=0,
        end_char=20,
        metadata={},
        created_at=datetime.now(UTC),
    )


def _make_retrieved(score: float = 0.9, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(index),
        score=score,
        document_source="test.txt",
    )


# ---------------------------------------------------------------------------
# BaseReranker tests
# ---------------------------------------------------------------------------


class TestBaseReranker:
    """Tests for the BaseReranker ABC."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]

    def test_subclass_must_implement_rerank(self) -> None:
        class IncompleteReranker(BaseReranker):
            @property
            def reranker_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteReranker()  # type: ignore[abstract]

    def test_subclass_must_implement_reranker_name(self) -> None:
        class IncompleteReranker(BaseReranker):
            async def rerank(
                self,
                query: str,
                chunks: list[RetrievedChunk],
                top_k: int = 10,
            ) -> list[RetrievedChunk]:
                return chunks

        with pytest.raises(TypeError):
            IncompleteReranker()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# CohereReranker tests
# ---------------------------------------------------------------------------


class TestCohereReranker:
    """Tests for the CohereReranker."""

    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            CohereReranker(api_key="")

    def test_reranker_name(self) -> None:
        reranker = CohereReranker(api_key="test-key", model="rerank-v3")
        assert reranker.reranker_name == "cohere:rerank-v3"

    async def test_rerank_empty_chunks_returns_empty(self) -> None:
        reranker = CohereReranker(api_key="test-key")
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []

    async def test_rerank_calls_api(self) -> None:
        chunks = [_make_retrieved(0.9, 0), _make_retrieved(0.8, 1)]

        response_data = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
            ]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(_COHERE_CLIENT_PATH, return_value=mock_client):
            reranker = CohereReranker(api_key="test-key")
            results = await reranker.rerank("test query", chunks, top_k=2)

        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[1].score == 0.85

    async def test_rerank_api_error_raises(self) -> None:
        chunks = [_make_retrieved(0.9, 0)]

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(_COHERE_CLIENT_PATH, return_value=mock_client):
            reranker = CohereReranker(api_key="bad-key")
            with pytest.raises(RetrievalError, match="returned 401"):
                await reranker.rerank("test", chunks)

    async def test_rerank_http_error_raises(self) -> None:
        chunks = [_make_retrieved(0.9, 0)]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(_COHERE_CLIENT_PATH, return_value=mock_client):
            reranker = CohereReranker(api_key="test-key")
            with pytest.raises(RetrievalError, match="request failed"):
                await reranker.rerank("test", chunks)

    async def test_rerank_respects_top_k(self) -> None:
        chunks = [_make_retrieved(0.9, i) for i in range(5)]

        response_data = {
            "results": [
                {"index": i, "relevance_score": 0.9 - i * 0.1}
                for i in range(2)
            ]
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(_COHERE_CLIENT_PATH, return_value=mock_client):
            reranker = CohereReranker(api_key="test-key")
            await reranker.rerank("test", chunks, top_k=2)

        # The payload should have top_n=2
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["top_n"] == 2

    async def test_rerank_sends_correct_payload(self) -> None:
        chunks = [_make_retrieved(0.9, 0)]

        response_data = {"results": [{"index": 0, "relevance_score": 0.9}]}

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(_COHERE_CLIENT_PATH, return_value=mock_client):
            reranker = CohereReranker(api_key="test-key", model="rerank-v3")
            await reranker.rerank("my query", chunks, top_k=5)

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["query"] == "my query"
        assert payload["model"] == "rerank-v3"
        assert payload["documents"] == ["Chunk text 0"]

        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer test-key"


# ---------------------------------------------------------------------------
# CrossEncoderReranker tests
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """Tests for the CrossEncoderReranker."""

    def test_reranker_name(self) -> None:
        reranker = CrossEncoderReranker(model_name="my-model")
        assert reranker.reranker_name == "cross-encoder:my-model"

    async def test_rerank_empty_chunks_returns_empty(self) -> None:
        reranker = CrossEncoderReranker()
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []

    async def test_rerank_scores_and_sorts(self) -> None:
        chunks = [_make_retrieved(0.5, 0), _make_retrieved(0.5, 1), _make_retrieved(0.5, 2)]

        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])

        reranker = CrossEncoderReranker()
        reranker._model = mock_model

        results = await reranker.rerank("test query", chunks, top_k=3)

        assert len(results) == 3
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.6)
        assert results[2].score == pytest.approx(0.3)

    async def test_rerank_respects_top_k(self) -> None:
        chunks = [_make_retrieved(0.5, i) for i in range(5)]

        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.5, 0.3, 0.9, 0.7])

        reranker = CrossEncoderReranker()
        reranker._model = mock_model

        results = await reranker.rerank("test", chunks, top_k=2)

        assert len(results) == 2
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.7)

    async def test_rerank_passes_correct_pairs(self) -> None:
        chunks = [_make_retrieved(0.5, 0), _make_retrieved(0.5, 1)]

        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.6])

        reranker = CrossEncoderReranker()
        reranker._model = mock_model

        await reranker.rerank("my query", chunks)

        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0] == ["my query", "Chunk text 0"]
        assert pairs[1] == ["my query", "Chunk text 1"]

    async def test_rerank_prediction_failure_raises(self) -> None:
        chunks = [_make_retrieved(0.5, 0)]

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")

        reranker = CrossEncoderReranker()
        reranker._model = mock_model

        with pytest.raises(RetrievalError, match="prediction failed"):
            await reranker.rerank("test", chunks)

    def test_lazy_load_missing_package_raises(self) -> None:
        reranker = CrossEncoderReranker()

        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(RetrievalError, match="sentence-transformers is required"),
        ):
            reranker._load_model()

    def test_lazy_load_caches_model(self) -> None:
        mock_model = MagicMock()
        reranker = CrossEncoderReranker()
        reranker._model = mock_model

        result = reranker._load_model()
        assert result is mock_model

    def test_lazy_load_model_failure_raises(self) -> None:
        reranker = CrossEncoderReranker()

        mock_ce_class = MagicMock(side_effect=RuntimeError("download failed"))
        mock_module = MagicMock()
        mock_module.CrossEncoder = mock_ce_class

        with (
            patch.dict("sys.modules", {"sentence_transformers": mock_module}),
            pytest.raises(RetrievalError, match="Failed to load"),
        ):
            reranker._load_model()

"""Unit tests for retrieval quality evaluators (Stage 6B)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from vectorforge.evaluation.evaluators.chunk_coverage import ChunkCoverageEvaluator
from vectorforge.evaluation.evaluators.embedding_drift import EmbeddingDriftDetector
from vectorforge.evaluation.evaluators.retrieval_relevance import (
    RetrievalRelevanceEvaluator,
)
from vectorforge.evaluation.types import ChunkWithScore, EvaluationSample
from vectorforge.llm.types import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    query: str = "What is Python?",
    answer: str = "Python is a programming language.",
    chunks: list[ChunkWithScore] | None = None,
) -> EvaluationSample:
    """Build an EvaluationSample for testing."""
    if chunks is None:
        chunks = [
            ChunkWithScore(
                chunk_id=uuid4(),
                text="Python is a high-level programming language.",
                score=0.9,
            ),
            ChunkWithScore(
                chunk_id=uuid4(),
                text="Python supports multiple paradigms.",
                score=0.8,
            ),
        ]
    return EvaluationSample(
        query_log_id=uuid4(),
        query=query,
        chunks=chunks,
        answer=answer,
    )


def _mock_llm(responses: list[dict]) -> MagicMock:
    """Create a mock LLM that returns sequential JSON responses."""
    llm = MagicMock()
    llm.default_model.return_value = "test-model"
    side_effects = [
        LLMResponse(
            content=json.dumps(r),
            model="test-model",
            provider="test",
        )
        for r in responses
    ]
    llm.generate = AsyncMock(side_effect=side_effects)
    return llm


# ---------------------------------------------------------------------------
# RetrievalRelevanceEvaluator tests
# ---------------------------------------------------------------------------


class TestRetrievalRelevanceEvaluator:
    """Tests for RetrievalRelevanceEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_with_relevant_chunks(self) -> None:
        llm = _mock_llm([
            {"score": 0.9, "reasoning": "Directly relevant"},
            {"score": 0.7, "reasoning": "Relevant info"},
        ])
        evaluator = RetrievalRelevanceEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.evaluator_name == "retrieval_relevance"
        assert result.score == 0.8  # mean(0.9, 0.7)
        assert result.details["relevant_count"] == 2
        assert result.details["total_chunks"] == 2
        assert result.details["precision_at_k"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_with_irrelevant_chunks(self) -> None:
        llm = _mock_llm([
            {"score": 0.1, "reasoning": "Not relevant"},
            {"score": 0.2, "reasoning": "Not relevant"},
        ])
        evaluator = RetrievalRelevanceEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 0.15  # mean(0.1, 0.2)
        assert result.details["relevant_count"] == 0
        assert result.details["precision_at_k"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_empty_chunks(self) -> None:
        llm = _mock_llm([])
        evaluator = RetrievalRelevanceEvaluator(llm)
        sample = _make_sample(chunks=[])

        result = await evaluator.evaluate(sample)

        assert result.score == 0.0
        assert result.details["reason"] == "no_chunks"

    @pytest.mark.asyncio
    async def test_evaluate_llm_failure_falls_back(self) -> None:
        llm = MagicMock()
        llm.default_model.return_value = "test-model"
        llm.generate = AsyncMock(side_effect=RuntimeError("API down"))
        evaluator = RetrievalRelevanceEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        # Fallback to 0.0 for all failed chunks
        assert result.score == 0.0
        assert result.details["total_chunks"] == 2

    def test_properties(self) -> None:
        llm = MagicMock()
        evaluator = RetrievalRelevanceEvaluator(llm)
        assert evaluator.name == "retrieval_relevance"
        assert evaluator.category == "retrieval"

    @pytest.mark.asyncio
    async def test_score_clamped_to_range(self) -> None:
        llm = _mock_llm([
            {"score": 1.5, "reasoning": "Over max"},
            {"score": -0.5, "reasoning": "Under min"},
        ])
        evaluator = RetrievalRelevanceEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        # Clamped: 1.0 and 0.0 → mean = 0.5
        assert result.score == 0.5


# ---------------------------------------------------------------------------
# ChunkCoverageEvaluator tests
# ---------------------------------------------------------------------------


class TestChunkCoverageEvaluator:
    """Tests for ChunkCoverageEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_full_coverage(self) -> None:
        llm = _mock_llm([
            {"aspects": ["language features", "use cases"]},
            {"covered": True, "chunk_index": 0, "reasoning": "Found"},
            {"covered": True, "chunk_index": 1, "reasoning": "Found"},
        ])
        evaluator = ChunkCoverageEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.evaluator_name == "chunk_coverage"
        assert result.score == 1.0
        assert result.details["coverage_ratio"] == 1.0
        assert len(result.details["covered_aspects"]) == 2
        assert len(result.details["uncovered_aspects"]) == 0

    @pytest.mark.asyncio
    async def test_evaluate_partial_coverage(self) -> None:
        llm = _mock_llm([
            {"aspects": ["feature A", "feature B", "feature C"]},
            {"covered": True, "chunk_index": 0, "reasoning": "Yes"},
            {"covered": False, "chunk_index": None, "reasoning": "Not found"},
            {"covered": True, "chunk_index": 1, "reasoning": "Yes"},
        ])
        evaluator = ChunkCoverageEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert abs(result.score - 0.6667) < 0.01  # 2/3
        assert len(result.details["uncovered_aspects"]) == 1

    @pytest.mark.asyncio
    async def test_evaluate_empty_chunks(self) -> None:
        llm = _mock_llm([])
        evaluator = ChunkCoverageEvaluator(llm)
        sample = _make_sample(chunks=[])

        result = await evaluator.evaluate(sample)

        assert result.score == 0.0
        assert result.details["reason"] == "no_chunks"

    @pytest.mark.asyncio
    async def test_evaluate_no_aspects(self) -> None:
        llm = _mock_llm([{"aspects": []}])
        evaluator = ChunkCoverageEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 1.0
        assert result.details["reason"] == "no_aspects_identified"

    @pytest.mark.asyncio
    async def test_aspect_extraction_failure(self) -> None:
        llm = MagicMock()
        llm.default_model.return_value = "test-model"
        llm.generate = AsyncMock(side_effect=RuntimeError("API down"))
        evaluator = ChunkCoverageEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        # No aspects extracted → treated as fully covered
        assert result.score == 1.0

    def test_properties(self) -> None:
        llm = MagicMock()
        evaluator = ChunkCoverageEvaluator(llm)
        assert evaluator.name == "chunk_coverage"
        assert evaluator.category == "retrieval"


# ---------------------------------------------------------------------------
# EmbeddingDriftDetector tests
# ---------------------------------------------------------------------------


class TestEmbeddingDriftDetector:
    """Tests for EmbeddingDriftDetector."""

    @pytest.mark.asyncio
    async def test_no_drift(self) -> None:
        detector = EmbeddingDriftDetector(
            historical_stats={"mean": 0.8, "std": 0.1}
        )
        sample = _make_sample(
            chunks=[
                ChunkWithScore(chunk_id=uuid4(), text="a", score=0.78),
                ChunkWithScore(chunk_id=uuid4(), text="b", score=0.82),
            ]
        )

        result = await detector.evaluate(sample)

        assert result.score == 1.0
        assert result.details["drift_detected"] is False
        assert result.details["severity"] == "low"

    @pytest.mark.asyncio
    async def test_medium_drift(self) -> None:
        detector = EmbeddingDriftDetector(
            historical_stats={"mean": 0.8, "std": 0.1}
        )
        sample = _make_sample(
            chunks=[
                ChunkWithScore(chunk_id=uuid4(), text="a", score=0.6),
                ChunkWithScore(chunk_id=uuid4(), text="b", score=0.6),
            ]
        )

        result = await detector.evaluate(sample)

        # recent_mean=0.6, historical=0.8, shift=-25% → high
        assert result.score == 0.2
        assert result.details["drift_detected"] is True

    @pytest.mark.asyncio
    async def test_high_drift(self) -> None:
        detector = EmbeddingDriftDetector(
            historical_stats={"mean": 0.8, "std": 0.1}
        )
        sample = _make_sample(
            chunks=[
                ChunkWithScore(chunk_id=uuid4(), text="a", score=0.4),
                ChunkWithScore(chunk_id=uuid4(), text="b", score=0.4),
            ]
        )

        result = await detector.evaluate(sample)

        assert result.score == 0.2
        assert result.details["severity"] == "high"

    @pytest.mark.asyncio
    async def test_no_historical_baseline(self) -> None:
        detector = EmbeddingDriftDetector(historical_stats=None)
        sample = _make_sample()

        result = await detector.evaluate(sample)

        assert result.score == 1.0
        assert result.details["reason"] == "no_historical_baseline"

    @pytest.mark.asyncio
    async def test_no_scores(self) -> None:
        detector = EmbeddingDriftDetector(
            historical_stats={"mean": 0.8, "std": 0.1}
        )
        sample = _make_sample(chunks=[])

        result = await detector.evaluate(sample)

        assert result.score == 1.0
        assert result.details["reason"] == "no_scores"

    @pytest.mark.asyncio
    async def test_zero_historical_mean(self) -> None:
        detector = EmbeddingDriftDetector(
            historical_stats={"mean": 0.0, "std": 0.0}
        )
        sample = _make_sample()

        result = await detector.evaluate(sample)

        assert result.score == 1.0
        assert result.details["reason"] == "zero_historical_mean"

    def test_properties(self) -> None:
        detector = EmbeddingDriftDetector()
        assert detector.name == "embedding_drift"
        assert detector.category == "retrieval"

    def test_classify_drift_boundaries(self) -> None:
        assert EmbeddingDriftDetector._classify_drift(0.10) == (False, "low", 1.0)
        assert EmbeddingDriftDetector._classify_drift(0.20) == (True, "medium", 0.5)
        assert EmbeddingDriftDetector._classify_drift(0.30) == (True, "high", 0.2)
        assert EmbeddingDriftDetector._classify_drift(-0.20) == (True, "medium", 0.5)

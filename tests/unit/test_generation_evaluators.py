"""Unit tests for generation quality evaluators (Stage 6C)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from vectorforge.evaluation.evaluators.answer_relevance import (
    AnswerRelevanceEvaluator,
)
from vectorforge.evaluation.evaluators.faithfulness import FaithfulnessEvaluator
from vectorforge.evaluation.evaluators.hallucination import HallucinationDetector
from vectorforge.evaluation.types import ChunkWithScore, EvaluationSample
from vectorforge.llm.types import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    query: str = "What is Python?",
    answer: str = "Python is a high-level programming language.",
    chunks: list[ChunkWithScore] | None = None,
) -> EvaluationSample:
    """Build an EvaluationSample for testing."""
    if chunks is None:
        chunks = [
            ChunkWithScore(
                chunk_id=uuid4(),
                text="Python is a high-level, general-purpose programming language.",
                score=0.9,
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
# FaithfulnessEvaluator tests
# ---------------------------------------------------------------------------


class TestFaithfulnessEvaluator:
    """Tests for FaithfulnessEvaluator."""

    @pytest.mark.asyncio
    async def test_all_claims_supported(self) -> None:
        llm = _mock_llm([
            {"claims": ["Python is a programming language", "It is high-level"]},
            {"verdict": "supported", "reasoning": "Found in context", "supporting_text": "..."},
            {"verdict": "supported", "reasoning": "Found in context", "supporting_text": "..."},
        ])
        evaluator = FaithfulnessEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.evaluator_name == "faithfulness"
        assert result.score == 1.0
        assert result.details["total_claims"] == 2
        assert result.details["supported_claims"] == 2
        assert result.details["unsupported_claims"] == 0

    @pytest.mark.asyncio
    async def test_partial_support(self) -> None:
        llm = _mock_llm([
            {"claims": ["claim A", "claim B", "claim C"]},
            {"verdict": "supported", "reasoning": "Yes"},
            {"verdict": "unsupported", "reasoning": "Not in context"},
            {"verdict": "ambiguous", "reasoning": "Unclear"},
        ])
        evaluator = FaithfulnessEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert abs(result.score - 0.3333) < 0.01  # 1/3
        assert result.details["supported_claims"] == 1
        assert result.details["unsupported_claims"] == 1
        assert result.details["ambiguous_claims"] == 1

    @pytest.mark.asyncio
    async def test_empty_answer(self) -> None:
        llm = _mock_llm([])
        evaluator = FaithfulnessEvaluator(llm)
        sample = _make_sample(answer="")

        result = await evaluator.evaluate(sample)

        assert result.score == 1.0
        assert result.details["reason"] == "empty_answer"

    @pytest.mark.asyncio
    async def test_no_claims_found(self) -> None:
        llm = _mock_llm([{"claims": []}])
        evaluator = FaithfulnessEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 1.0
        assert result.details["reason"] == "no_claims"

    @pytest.mark.asyncio
    async def test_claim_extraction_failure(self) -> None:
        llm = MagicMock()
        llm.default_model.return_value = "test-model"
        llm.generate = AsyncMock(side_effect=RuntimeError("API down"))
        evaluator = FaithfulnessEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        # Failed extraction → no claims → perfect score
        assert result.score == 1.0

    def test_properties(self) -> None:
        llm = MagicMock()
        evaluator = FaithfulnessEvaluator(llm)
        assert evaluator.name == "faithfulness"
        assert evaluator.category == "generation"


# ---------------------------------------------------------------------------
# AnswerRelevanceEvaluator tests
# ---------------------------------------------------------------------------


class TestAnswerRelevanceEvaluator:
    """Tests for AnswerRelevanceEvaluator."""

    @pytest.mark.asyncio
    async def test_highly_relevant(self) -> None:
        llm = _mock_llm([
            {"score": 0.95, "reasoning": "Directly answers the question"},
        ])
        evaluator = AnswerRelevanceEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.evaluator_name == "answer_relevance"
        assert result.score == 0.95
        assert result.details["direct_relevance_score"] == 0.95

    @pytest.mark.asyncio
    async def test_low_relevance(self) -> None:
        llm = _mock_llm([
            {"score": 0.2, "reasoning": "Off-topic"},
        ])
        evaluator = AnswerRelevanceEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_empty_answer(self) -> None:
        llm = _mock_llm([])
        evaluator = AnswerRelevanceEvaluator(llm)
        sample = _make_sample(answer="")

        result = await evaluator.evaluate(sample)

        assert result.score == 0.0
        assert result.details["reason"] == "empty_answer"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self) -> None:
        llm = MagicMock()
        llm.default_model.return_value = "test-model"
        llm.generate = AsyncMock(side_effect=RuntimeError("API down"))
        evaluator = AnswerRelevanceEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_score_clamped(self) -> None:
        llm = _mock_llm([
            {"score": 1.5, "reasoning": "Over max"},
        ])
        evaluator = AnswerRelevanceEvaluator(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 1.0

    def test_properties(self) -> None:
        llm = MagicMock()
        evaluator = AnswerRelevanceEvaluator(llm)
        assert evaluator.name == "answer_relevance"
        assert evaluator.category == "generation"


# ---------------------------------------------------------------------------
# HallucinationDetector tests
# ---------------------------------------------------------------------------


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""

    @pytest.mark.asyncio
    async def test_no_hallucinations(self) -> None:
        llm = _mock_llm([{
            "has_hallucinations": False,
            "hallucinations": [],
            "overall_assessment": "No hallucinations detected.",
        }])
        evaluator = HallucinationDetector(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.evaluator_name == "hallucination"
        assert result.score == 1.0
        assert result.details["has_hallucinations"] is False
        assert result.details["hallucination_count"] == 0

    @pytest.mark.asyncio
    async def test_minor_hallucination(self) -> None:
        llm = _mock_llm([{
            "has_hallucinations": True,
            "hallucinations": [
                {"text_span": "exact year", "reasoning": "Not in context", "severity": "minor"},
            ],
            "overall_assessment": "One minor embellishment found.",
        }])
        evaluator = HallucinationDetector(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 0.9  # 1.0 - 0.1
        assert result.details["severity_breakdown"]["minor"] == 1

    @pytest.mark.asyncio
    async def test_major_hallucination(self) -> None:
        llm = _mock_llm([{
            "has_hallucinations": True,
            "hallucinations": [
                {"text_span": "wrong fact", "reasoning": "Contradicts context",
                 "severity": "major"},
            ],
            "overall_assessment": "One major fabrication.",
        }])
        evaluator = HallucinationDetector(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 0.7  # 1.0 - 0.3
        assert result.details["severity_breakdown"]["major"] == 1

    @pytest.mark.asyncio
    async def test_critical_hallucination(self) -> None:
        llm = _mock_llm([{
            "has_hallucinations": True,
            "hallucinations": [
                {"text_span": "fake citation", "reasoning": "No such source",
                 "severity": "critical"},
            ],
            "overall_assessment": "Fabricated citation.",
        }])
        evaluator = HallucinationDetector(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        assert result.score == 0.5  # 1.0 - 0.5
        assert result.details["severity_breakdown"]["critical"] == 1

    @pytest.mark.asyncio
    async def test_multiple_hallucinations(self) -> None:
        llm = _mock_llm([{
            "has_hallucinations": True,
            "hallucinations": [
                {"text_span": "a", "reasoning": "r1", "severity": "minor"},
                {"text_span": "b", "reasoning": "r2", "severity": "major"},
                {"text_span": "c", "reasoning": "r3", "severity": "critical"},
            ],
            "overall_assessment": "Multiple issues.",
        }])
        evaluator = HallucinationDetector(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        # 1.0 - (0.1 + 0.3 + 0.5) = 0.1
        assert abs(result.score - 0.1) < 0.01
        assert result.details["hallucination_count"] == 3

    @pytest.mark.asyncio
    async def test_empty_answer(self) -> None:
        llm = _mock_llm([])
        evaluator = HallucinationDetector(llm)
        sample = _make_sample(answer="")

        result = await evaluator.evaluate(sample)

        assert result.score == 1.0
        assert result.details["reason"] == "empty_answer"

    @pytest.mark.asyncio
    async def test_no_context(self) -> None:
        llm = _mock_llm([])
        evaluator = HallucinationDetector(llm)
        sample = _make_sample(chunks=[])

        result = await evaluator.evaluate(sample)

        assert result.score == 0.0
        assert result.details["reason"] == "no_context"

    @pytest.mark.asyncio
    async def test_llm_failure_defaults_safe(self) -> None:
        llm = MagicMock()
        llm.default_model.return_value = "test-model"
        llm.generate = AsyncMock(side_effect=RuntimeError("API down"))
        evaluator = HallucinationDetector(llm)
        sample = _make_sample()

        result = await evaluator.evaluate(sample)

        # Failure defaults to no hallucinations detected
        assert result.score == 1.0

    def test_properties(self) -> None:
        llm = MagicMock()
        evaluator = HallucinationDetector(llm)
        assert evaluator.name == "hallucination"
        assert evaluator.category == "generation"

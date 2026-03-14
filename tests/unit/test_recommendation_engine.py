"""Unit tests for VectorForge RecommendationEngine (Stage 6D)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vectorforge.evaluation.config import EvaluationConfig
from vectorforge.evaluation.recommendation import (
    RecommendationEngine,
    _classify_severity,
    _get_worst_n,
)
from vectorforge.evaluation.types import (
    EvaluationResult,
    RecommendationCategory,
    RecommendationSeverity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUN_ID = uuid4()


def _make_result(
    evaluator: str,
    score: float,
    *,
    query_log_id=None,
    reasoning: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        query_log_id=query_log_id or uuid4(),
        evaluator_name=evaluator,
        score=score,
        details={},
        reasoning=reasoning,
    )


@pytest.fixture()
def engine() -> RecommendationEngine:
    return RecommendationEngine()


@pytest.fixture()
def custom_engine() -> RecommendationEngine:
    config = EvaluationConfig(
        faithfulness_threshold=0.8,
        relevance_threshold=0.7,
        hallucination_threshold=0.2,
        coverage_threshold=0.6,
    )
    return RecommendationEngine(config)


# ---------------------------------------------------------------------------
# _classify_severity
# ---------------------------------------------------------------------------


class TestClassifySeverity:
    """Tests for the severity classification helper."""

    def test_critical_gap(self) -> None:
        assert _classify_severity(0.3, 0.7) == RecommendationSeverity.CRITICAL

    def test_high_gap(self) -> None:
        assert _classify_severity(0.5, 0.7) == RecommendationSeverity.HIGH

    def test_medium_gap(self) -> None:
        assert _classify_severity(0.6, 0.7) == RecommendationSeverity.MEDIUM

    def test_low_gap(self) -> None:
        assert _classify_severity(0.68, 0.7) == RecommendationSeverity.LOW

    def test_exact_boundary_critical(self) -> None:
        # gap = 0.31 > 0.3 → CRITICAL
        assert _classify_severity(0.39, 0.7) == RecommendationSeverity.CRITICAL

    def test_exact_boundary_high(self) -> None:
        # gap = 0.16 > 0.15 → HIGH
        assert _classify_severity(0.54, 0.7) == RecommendationSeverity.HIGH


# ---------------------------------------------------------------------------
# _get_worst_n
# ---------------------------------------------------------------------------


class TestGetWorstN:
    """Tests for worst-N extraction helper."""

    def test_returns_n_worst(self) -> None:
        results = [
            _make_result("test", 0.9),
            _make_result("test", 0.1),
            _make_result("test", 0.5),
            _make_result("test", 0.2),
        ]
        worst = _get_worst_n(results, "test", 2)
        assert len(worst) == 2
        assert worst[0]["score"] == 0.1
        assert worst[1]["score"] == 0.2

    def test_filters_by_evaluator(self) -> None:
        results = [
            _make_result("a", 0.1),
            _make_result("b", 0.2),
            _make_result("a", 0.3),
        ]
        worst = _get_worst_n(results, "a", 5)
        assert len(worst) == 2

    def test_skips_none_scores(self) -> None:
        results = [
            EvaluationResult(
                query_log_id=uuid4(),
                evaluator_name="test",
                score=None,
            ),
            _make_result("test", 0.5),
        ]
        worst = _get_worst_n(results, "test", 5)
        assert len(worst) == 1

    def test_empty_results(self) -> None:
        assert _get_worst_n([], "test", 3) == []


# ---------------------------------------------------------------------------
# RecommendationEngine.analyze
# ---------------------------------------------------------------------------


class TestRecommendationEngineAnalyze:
    """Tests for the full analyze pipeline."""

    def test_returns_empty_when_all_scores_above_threshold(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {
            "retrieval_relevance": {"avg": 0.9},
            "chunk_coverage": {"avg": 0.9},
            "embedding_drift": {"avg": 0.9},
            "faithfulness": {"avg": 0.9},
            "hallucination": {"avg": 0.9},
            "answer_relevance": {"avg": 0.9},
        }
        recs = engine.analyze(RUN_ID, summary, [])
        assert recs == []

    def test_low_retrieval_generates_recommendation(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {"retrieval_relevance": {"avg": 0.3}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert len(recs) == 1
        assert recs[0].category == RecommendationCategory.RETRIEVAL
        assert recs[0].title == "Low Retrieval Relevance"

    def test_low_chunk_coverage_generates_recommendation(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {"chunk_coverage": {"avg": 0.2}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert len(recs) == 1
        assert recs[0].category == RecommendationCategory.CHUNKING
        assert recs[0].title == "Incomplete Chunk Coverage"

    def test_embedding_drift_below_point_five(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {"embedding_drift": {"avg": 0.3}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert len(recs) == 1
        assert recs[0].category == RecommendationCategory.EMBEDDING
        assert recs[0].severity == RecommendationSeverity.HIGH

    def test_embedding_drift_above_threshold_no_rec(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {"embedding_drift": {"avg": 0.6}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert recs == []

    def test_low_faithfulness_generates_recommendation(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {"faithfulness": {"avg": 0.4}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert len(recs) == 1
        assert recs[0].category == RecommendationCategory.GENERATION
        assert "Faithfulness" in recs[0].title

    def test_hallucination_critical_when_rate_above_50(
        self, engine: RecommendationEngine
    ) -> None:
        # score = 0.3 → hallucination_rate = 0.7 > 0.5 → CRITICAL
        summary = {"hallucination": {"avg": 0.3}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert len(recs) == 1
        assert recs[0].severity == RecommendationSeverity.CRITICAL

    def test_hallucination_high_when_rate_above_threshold(
        self, engine: RecommendationEngine
    ) -> None:
        # score = 0.6 → rate = 0.4; threshold_score = 1.0 - 0.3 = 0.7
        # 0.6 < 0.7 → fires, rate = 0.4 ≤ 0.5 → HIGH
        summary = {"hallucination": {"avg": 0.6}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert len(recs) == 1
        assert recs[0].severity == RecommendationSeverity.HIGH

    def test_hallucination_passes_when_score_above_threshold(
        self, engine: RecommendationEngine
    ) -> None:
        # score = 0.8; threshold_score = 1.0 - 0.3 = 0.7; 0.8 >= 0.7 → pass
        summary = {"hallucination": {"avg": 0.8}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert recs == []

    def test_low_answer_relevance_generates_recommendation(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {"answer_relevance": {"avg": 0.3}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert len(recs) == 1
        assert "Relevance" in recs[0].title

    def test_cross_cutting_critical_fires(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {
            "retrieval_relevance": {"avg": 0.3},
            "faithfulness": {"avg": 0.3},
        }
        recs = engine.analyze(RUN_ID, summary, [])
        # retrieval_relevance rec + faithfulness rec + cross-cutting
        cross = [r for r in recs if "Systemic" in r.title]
        assert len(cross) == 1
        assert cross[0].severity == RecommendationSeverity.CRITICAL

    def test_cross_cutting_does_not_fire_when_one_above(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {
            "retrieval_relevance": {"avg": 0.3},
            "faithfulness": {"avg": 0.7},
        }
        recs = engine.analyze(RUN_ID, summary, [])
        cross = [r for r in recs if "Systemic" in r.title]
        assert len(cross) == 0

    def test_results_sorted_by_severity(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {
            "retrieval_relevance": {"avg": 0.55},  # medium severity
            "hallucination": {"avg": 0.3},  # critical severity
        }
        recs = engine.analyze(RUN_ID, summary, [])
        assert len(recs) >= 2
        assert recs[0].severity == RecommendationSeverity.CRITICAL

    def test_custom_thresholds(self, custom_engine: RecommendationEngine) -> None:
        # relevance_threshold=0.7, so avg=0.65 should trigger
        summary = {"retrieval_relevance": {"avg": 0.65}}
        recs = custom_engine.analyze(RUN_ID, summary, [])
        assert len(recs) == 1

    def test_evidence_includes_worst_queries(
        self, engine: RecommendationEngine
    ) -> None:
        results = [_make_result("retrieval_relevance", 0.2, reasoning="bad")]
        summary = {"retrieval_relevance": {"avg": 0.3}}
        recs = engine.analyze(RUN_ID, summary, results)
        assert len(recs) == 1
        assert "worst_queries" in recs[0].evidence
        assert len(recs[0].evidence["worst_queries"]) == 1

    def test_empty_summary(self, engine: RecommendationEngine) -> None:
        recs = engine.analyze(RUN_ID, {}, [])
        assert recs == []

    def test_missing_evaluator_skipped(
        self, engine: RecommendationEngine
    ) -> None:
        summary = {"unknown_evaluator": {"avg": 0.1}}
        recs = engine.analyze(RUN_ID, summary, [])
        assert recs == []

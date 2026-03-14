"""Unit tests for VectorForge EvaluationReportBuilder (Stage 6D)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from vectorforge.evaluation.report import EvaluationReportBuilder, _classify_direction
from vectorforge.evaluation.types import (
    EvaluationResult,
    EvaluationRun,
    EvaluationRunStatus,
    Recommendation,
    RecommendationCategory,
    RecommendationSeverity,
    RecommendationStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run(
    summary_scores: dict[str, Any] | None = None,
    *,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    sample_size: int = 50,
    status: EvaluationRunStatus = EvaluationRunStatus.COMPLETED,
) -> EvaluationRun:
    now = datetime.now()
    return EvaluationRun(
        id=uuid4(),
        status=status,
        started_at=started_at or now - timedelta(minutes=5),
        completed_at=completed_at or now,
        sample_size=sample_size,
        config={"sample_strategy": "recent"},
        summary_scores=summary_scores or {},
    )


def _make_result(
    evaluator: str,
    score: float,
    *,
    query_log_id=None,
    run_id=None,
) -> EvaluationResult:
    return EvaluationResult(
        query_log_id=query_log_id or uuid4(),
        evaluator_name=evaluator,
        score=score,
        details={},
        reasoning=None,
    )


def _make_rec(*, run_id=None) -> Recommendation:
    return Recommendation(
        id=uuid4(),
        run_id=run_id or uuid4(),
        category=RecommendationCategory.RETRIEVAL,
        severity=RecommendationSeverity.HIGH,
        title="Test Recommendation",
        description="Test description",
        evidence={"test": True},
        status=RecommendationStatus.PENDING,
    )


@pytest.fixture()
def builder() -> EvaluationReportBuilder:
    return EvaluationReportBuilder()


# ---------------------------------------------------------------------------
# _classify_direction
# ---------------------------------------------------------------------------


class TestClassifyDirection:
    """Tests for the trend direction classifier."""

    def test_improving(self) -> None:
        assert _classify_direction(10.0) == "improving"

    def test_degrading(self) -> None:
        assert _classify_direction(-10.0) == "degrading"

    def test_stable_positive(self) -> None:
        assert _classify_direction(3.0) == "stable"

    def test_stable_negative(self) -> None:
        assert _classify_direction(-3.0) == "stable"

    def test_stable_zero(self) -> None:
        assert _classify_direction(0.0) == "stable"

    def test_boundary_improving(self) -> None:
        # 5.1 > 5.0 → improving
        assert _classify_direction(5.1) == "improving"

    def test_boundary_exact_not_improving(self) -> None:
        # 5.0 is NOT > 5.0 → stable
        assert _classify_direction(5.0) == "stable"


# ---------------------------------------------------------------------------
# build - header
# ---------------------------------------------------------------------------


class TestBuildHeader:
    """Tests for the report header section."""

    def test_header_contains_run_metadata(
        self, builder: EvaluationReportBuilder
    ) -> None:
        run = _make_run()
        report = builder.build(run, [], [])
        header = report.header
        assert header["run_id"] == str(run.id)
        assert header["sample_size"] == 50
        assert header["status"] == "completed"
        assert header["duration_seconds"] is not None
        assert header["duration_seconds"] > 0

    def test_header_no_timestamps(
        self, builder: EvaluationReportBuilder
    ) -> None:
        run = _make_run(started_at=None, completed_at=None)
        # Override the auto-set values
        run.started_at = None
        run.completed_at = None
        report = builder.build(run, [], [])
        assert report.header["timestamp"] is None
        assert report.header["duration_seconds"] is None


# ---------------------------------------------------------------------------
# build - score summary
# ---------------------------------------------------------------------------


class TestBuildScoreSummary:
    """Tests for per-evaluator score summary."""

    def test_builds_rows_from_summary(
        self, builder: EvaluationReportBuilder
    ) -> None:
        summary = {
            "retrieval_relevance": {
                "avg": 0.8,
                "min": 0.5,
                "max": 1.0,
                "p50": 0.8,
                "below_threshold": 2,
                "sample_count": 50,
            },
        }
        run = _make_run(summary)
        report = builder.build(run, [], [], thresholds={"retrieval_relevance": 0.6})
        assert len(report.score_summary) == 1
        row = report.score_summary[0]
        assert row.evaluator == "retrieval_relevance"
        assert row.avg == 0.8
        assert row.status == "pass"

    def test_fail_status_below_threshold(
        self, builder: EvaluationReportBuilder
    ) -> None:
        summary = {
            "faithfulness": {
                "avg": 0.4, "min": 0.1, "max": 0.6,
                "p50": 0.4, "below_threshold": 30, "sample_count": 50,
            },
        }
        run = _make_run(summary)
        report = builder.build(run, [], [], thresholds={"faithfulness": 0.7})
        assert report.score_summary[0].status == "fail"

    def test_default_threshold_is_half(
        self, builder: EvaluationReportBuilder
    ) -> None:
        summary = {
            "test_eval": {
                "avg": 0.45, "min": 0.2, "max": 0.7,
                "p50": 0.45, "below_threshold": 0, "sample_count": 10,
            },
        }
        run = _make_run(summary)
        report = builder.build(run, [], [])
        assert report.score_summary[0].status == "fail"  # 0.45 < 0.5

    def test_skips_internal_keys(
        self, builder: EvaluationReportBuilder
    ) -> None:
        summary = {
            "_internal": {"avg": 0.5},
            "public": {
                "avg": 0.8, "min": 0.5, "max": 1.0,
                "p50": 0.8, "below_threshold": 0, "sample_count": 10,
            },
        }
        run = _make_run(summary)
        report = builder.build(run, [], [])
        assert len(report.score_summary) == 1
        assert report.score_summary[0].evaluator == "public"

    def test_skips_non_dict_values(
        self, builder: EvaluationReportBuilder
    ) -> None:
        summary = {
            "bad_entry": "not a dict",
            "good": {
                "avg": 0.7, "min": 0.5, "max": 0.9,
                "p50": 0.7, "below_threshold": 0, "sample_count": 5,
            },
        }
        run = _make_run(summary)
        report = builder.build(run, [], [])
        assert len(report.score_summary) == 1


# ---------------------------------------------------------------------------
# build - trends
# ---------------------------------------------------------------------------


class TestBuildTrends:
    """Tests for trend computation."""

    def test_single_run_is_stable(
        self, builder: EvaluationReportBuilder
    ) -> None:
        summary = {"eval_a": {"avg": 0.8}}
        run = _make_run(summary)
        report = builder.build(run, [], [])
        assert len(report.trends) == 1
        assert report.trends[0].direction == "stable"
        assert report.trends[0].change_pct == 0.0

    def test_improving_trend(
        self, builder: EvaluationReportBuilder
    ) -> None:
        old_run = _make_run({"eval_a": {"avg": 0.5}})
        new_run = _make_run({"eval_a": {"avg": 0.7}})
        report = builder.build(new_run, [], [], previous_runs=[old_run])
        trend = report.trends[0]
        assert trend.direction == "improving"
        assert trend.change_pct > 5.0

    def test_degrading_trend(
        self, builder: EvaluationReportBuilder
    ) -> None:
        old_run = _make_run({"eval_a": {"avg": 0.8}})
        new_run = _make_run({"eval_a": {"avg": 0.5}})
        report = builder.build(new_run, [], [], previous_runs=[old_run])
        trend = report.trends[0]
        assert trend.direction == "degrading"
        assert trend.change_pct < -5.0

    def test_stable_trend(
        self, builder: EvaluationReportBuilder
    ) -> None:
        old_run = _make_run({"eval_a": {"avg": 0.8}})
        new_run = _make_run({"eval_a": {"avg": 0.81}})
        report = builder.build(new_run, [], [], previous_runs=[old_run])
        assert report.trends[0].direction == "stable"

    def test_trend_collects_all_scores(
        self, builder: EvaluationReportBuilder
    ) -> None:
        # previous_runs are newest-first; builder reverses to oldest-first
        runs = [
            _make_run({"eval_a": {"avg": 0.6}}),
            _make_run({"eval_a": {"avg": 0.5}}),
        ]
        current = _make_run({"eval_a": {"avg": 0.7}})
        report = builder.build(current, [], [], previous_runs=runs)
        assert report.trends[0].scores == [0.5, 0.6, 0.7]

    def test_trend_zero_baseline(
        self, builder: EvaluationReportBuilder
    ) -> None:
        old_run = _make_run({"eval_a": {"avg": 0.0}})
        new_run = _make_run({"eval_a": {"avg": 0.5}})
        report = builder.build(new_run, [], [], previous_runs=[old_run])
        assert report.trends[0].change_pct == 100.0


# ---------------------------------------------------------------------------
# build - worst queries
# ---------------------------------------------------------------------------


class TestBuildWorstQueries:
    """Tests for worst query identification."""

    def test_returns_worst_by_composite(
        self, builder: EvaluationReportBuilder
    ) -> None:
        q1 = uuid4()
        q2 = uuid4()
        results = [
            _make_result("eval_a", 0.9, query_log_id=q1),
            _make_result("eval_b", 0.8, query_log_id=q1),
            _make_result("eval_a", 0.2, query_log_id=q2),
            _make_result("eval_b", 0.3, query_log_id=q2),
        ]
        run = _make_run()
        report = builder.build(run, results, [])
        assert len(report.worst_queries) == 2
        # q2 (avg=0.25) should be first (worst)
        assert report.worst_queries[0].query_log_id == q2

    def test_key_issues_lists_low_scores(
        self, builder: EvaluationReportBuilder
    ) -> None:
        q1 = uuid4()
        results = [
            _make_result("eval_a", 0.3, query_log_id=q1),
            _make_result("eval_b", 0.8, query_log_id=q1),
        ]
        run = _make_run()
        report = builder.build(run, results, [])
        issues = report.worst_queries[0].key_issues
        assert len(issues) == 1
        assert "eval_a" in issues[0]

    def test_limits_to_ten(
        self, builder: EvaluationReportBuilder
    ) -> None:
        results = []
        for _ in range(15):
            qid = uuid4()
            results.append(_make_result("eval_a", 0.3, query_log_id=qid))
        run = _make_run()
        report = builder.build(run, results, [])
        assert len(report.worst_queries) == 10

    def test_skips_none_scores(
        self, builder: EvaluationReportBuilder
    ) -> None:
        q1 = uuid4()
        results = [
            EvaluationResult(
                query_log_id=q1,
                evaluator_name="eval_a",
                score=None,
            ),
        ]
        run = _make_run()
        report = builder.build(run, results, [])
        # q1 has no valid scores → composite is 0.0
        assert report.worst_queries[0].composite_score == 0.0

    def test_empty_results(
        self, builder: EvaluationReportBuilder
    ) -> None:
        run = _make_run()
        report = builder.build(run, [], [])
        assert report.worst_queries == []


# ---------------------------------------------------------------------------
# build - full report
# ---------------------------------------------------------------------------


class TestFullReport:
    """Tests for the complete report build."""

    def test_includes_all_sections(
        self, builder: EvaluationReportBuilder
    ) -> None:
        summary = {
            "eval_a": {
                "avg": 0.7, "min": 0.5, "max": 0.9,
                "p50": 0.7, "below_threshold": 2, "sample_count": 50,
            },
        }
        run = _make_run(summary)
        rec = _make_rec(run_id=run.id)
        results = [_make_result("eval_a", 0.5)]
        report = builder.build(run, results, [rec])

        assert report.header["run_id"] == str(run.id)
        assert len(report.score_summary) == 1
        assert len(report.recommendations) == 1
        assert len(report.worst_queries) == 1
        assert report.raw_result_count == 1

    def test_raw_result_count_matches(
        self, builder: EvaluationReportBuilder
    ) -> None:
        run = _make_run()
        results = [_make_result("e", 0.5) for _ in range(7)]
        report = builder.build(run, results, [])
        assert report.raw_result_count == 7

    def test_recommendations_passed_through(
        self, builder: EvaluationReportBuilder
    ) -> None:
        run = _make_run()
        recs = [_make_rec(run_id=run.id), _make_rec(run_id=run.id)]
        report = builder.build(run, [], recs)
        assert len(report.recommendations) == 2

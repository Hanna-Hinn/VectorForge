"""Report builder — constructs evaluation reports from run data."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from vectorforge.evaluation.types import (
    EvaluationReport,
    EvaluationResult,
    EvaluationRun,
    Recommendation,
    ScoreSummaryRow,
    TrendData,
    WorstQuery,
)

logger = logging.getLogger(__name__)


class EvaluationReportBuilder:
    """Builds a complete evaluation report from run data.

    Aggregates score summaries, trend data, worst queries, and
    recommendations into an EvaluationReport.
    """

    def build(
        self,
        run: EvaluationRun,
        results: list[EvaluationResult],
        recommendations: list[Recommendation],
        *,
        previous_runs: list[EvaluationRun] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> EvaluationReport:
        """Build a full evaluation report.

        Args:
            run: The completed evaluation run.
            results: Individual evaluation results for this run.
            recommendations: Recommendations generated for this run.
            previous_runs: Recent previous runs for trend data.
            thresholds: Evaluator thresholds for pass/fail status.

        Returns:
            A complete EvaluationReport.
        """
        th = thresholds or {}
        header = self._build_header(run)
        score_summary = self._build_score_summary(run.summary_scores, th)
        trends = self._build_trends(run, previous_runs or [])
        worst_queries = self._build_worst_queries(results)

        return EvaluationReport(
            header=header,
            score_summary=score_summary,
            trends=trends,
            recommendations=recommendations,
            worst_queries=worst_queries,
            raw_result_count=len(results),
        )

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_header(self, run: EvaluationRun) -> dict[str, Any]:
        """Build report header from run metadata."""
        duration: float | None = None
        if run.started_at and run.completed_at:
            duration = (run.completed_at - run.started_at).total_seconds()

        return {
            "run_id": str(run.id),
            "timestamp": run.completed_at.isoformat() if run.completed_at else None,
            "sample_size": run.sample_size,
            "duration_seconds": duration,
            "config_snapshot": run.config,
            "status": str(run.status),
        }

    def _build_score_summary(
        self,
        summary_scores: dict[str, Any],
        thresholds: dict[str, float],
    ) -> list[ScoreSummaryRow]:
        """Build per-evaluator score summary table."""
        rows: list[ScoreSummaryRow] = []
        for evaluator, scores in summary_scores.items():
            if evaluator.startswith("_"):
                continue
            if not isinstance(scores, dict):
                continue
            avg = float(scores.get("avg", 0.0))
            threshold = thresholds.get(evaluator, 0.5)
            rows.append(
                ScoreSummaryRow(
                    evaluator=evaluator,
                    avg=avg,
                    min_score=float(scores.get("min", 0.0)),
                    max_score=float(scores.get("max", 0.0)),
                    p50=float(scores.get("p50", 0.0)),
                    below_threshold=int(scores.get("below_threshold", 0)),
                    sample_count=int(scores.get("sample_count", 0)),
                    status="pass" if avg >= threshold else "fail",
                )
            )
        return rows

    def _build_trends(
        self,
        current_run: EvaluationRun,
        previous_runs: list[EvaluationRun],
    ) -> list[TrendData]:
        """Build trend data from current and previous runs."""
        # Collect all runs oldest-first
        all_runs = [*list(reversed(previous_runs)), current_run]
        # Collect evaluator names from the current run
        evaluator_names: set[str] = set()
        for run in all_runs:
            if run.summary_scores:
                for key in run.summary_scores:
                    if not key.startswith("_") and isinstance(
                        run.summary_scores[key], dict
                    ):
                        evaluator_names.add(key)

        trends: list[TrendData] = []
        for name in sorted(evaluator_names):
            scores: list[float] = []
            for run in all_runs:
                evaluator_data = run.summary_scores.get(name)
                if isinstance(evaluator_data, dict) and "avg" in evaluator_data:
                    scores.append(float(evaluator_data["avg"]))

            if len(scores) < 2:
                direction = "stable"
                change_pct = 0.0
            else:
                oldest = scores[0]
                newest = scores[-1]
                if oldest > 0:
                    change_pct = ((newest - oldest) / oldest) * 100
                else:
                    change_pct = 100.0 if newest > 0 else 0.0
                direction = _classify_direction(change_pct)

            trends.append(
                TrendData(
                    evaluator=name,
                    scores=scores,
                    direction=direction,
                    change_pct=round(change_pct, 2),
                )
            )
        return trends

    def _build_worst_queries(
        self,
        results: list[EvaluationResult],
        *,
        limit: int = 10,
    ) -> list[WorstQuery]:
        """Identify the worst-performing queries by composite score."""
        # Group results by query_log_id
        by_query: dict[UUID, list[EvaluationResult]] = defaultdict(list)
        for r in results:
            by_query[r.query_log_id].append(r)

        composites: list[WorstQuery] = []
        for query_log_id, query_results in by_query.items():
            scores: dict[str, float] = {}
            issues: list[str] = []
            for r in query_results:
                if r.score is not None:
                    scores[r.evaluator_name] = r.score
                    if r.score < 0.5:
                        issues.append(f"Low {r.evaluator_name}: {r.score:.2f}")

            valid_scores = list(scores.values())
            composite = (
                sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            )

            composites.append(
                WorstQuery(
                    query_log_id=query_log_id,
                    query="",  # Populated at query time if needed
                    composite_score=round(composite, 4),
                    per_evaluator_scores=scores,
                    key_issues=issues,
                )
            )

        composites.sort(key=lambda w: w.composite_score)
        return composites[:limit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_direction(change_pct: float) -> str:
    """Classify trend direction from percentage change.

    Args:
        change_pct: Percentage change from oldest to newest.

    Returns:
        "improving", "degrading", or "stable".
    """
    if change_pct > 5.0:
        return "improving"
    if change_pct < -5.0:
        return "degrading"
    return "stable"

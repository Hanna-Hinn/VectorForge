"""EmbeddingDriftDetector — detects degradation in embedding quality via distribution analysis."""

from __future__ import annotations

import logging
import statistics
from typing import Any

from vectorforge.evaluation.base import BaseEvaluator
from vectorforge.evaluation.types import EvaluationResult, EvaluationSample

logger = logging.getLogger(__name__)


class EmbeddingDriftDetector(BaseEvaluator):
    """Detects embedding quality drift by comparing similarity score distributions.

    Unlike the LLM-as-judge evaluators, this evaluator analyses the
    *distribution* of retrieval similarity scores across recent vs.
    historical queries.  It does not call an LLM judge.

    It works on a per-sample basis by comparing each sample's chunk
    scores against historical baselines provided via ``historical_stats``.

    Args:
        historical_stats: Pre-computed stats from a reference period.
            Expected keys: ``mean``, ``std``.  When ``None`` the evaluator
            uses the sample's own scores and produces a neutral result.
    """

    def __init__(
        self,
        historical_stats: dict[str, float] | None = None,
    ) -> None:
        self._historical = historical_stats

    @property
    def name(self) -> str:
        return "embedding_drift"

    @property
    def category(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return (
            "Detects if embedding quality has degraded by comparing "
            "recent vs historical query-chunk similarity distributions"
        )

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate embedding drift for a single sample's chunk scores.

        Args:
            sample: The query-answer pair with chunk similarity scores.

        Returns:
            EvaluationResult with drift magnitude and detection details.
        """
        recent_scores = [c.score for c in sample.chunks if c.score is not None]

        if not recent_scores:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=1.0,
                details={"reason": "no_scores"},
                reasoning="No similarity scores available for analysis.",
            )

        recent_mean = statistics.mean(recent_scores)

        if self._historical is None or "mean" not in self._historical:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=1.0,
                details={
                    "recent_mean": round(recent_mean, 4),
                    "reason": "no_historical_baseline",
                },
                reasoning="No historical baseline available; skipping drift detection.",
            )

        historical_mean = self._historical["mean"]
        if historical_mean == 0:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=1.0,
                details={"reason": "zero_historical_mean"},
                reasoning="Historical mean is zero; cannot compute drift.",
            )

        mean_shift = recent_mean - historical_mean
        relative_shift = mean_shift / historical_mean

        drift_detected, severity, score = self._classify_drift(relative_shift)

        details: dict[str, Any] = {
            "recent_mean": round(recent_mean, 4),
            "historical_mean": round(historical_mean, 4),
            "mean_shift": round(mean_shift, 4),
            "relative_shift": round(relative_shift, 4),
            "drift_detected": drift_detected,
            "severity": severity,
            "recent_sample_size": len(recent_scores),
        }

        return EvaluationResult(
            query_log_id=sample.query_log_id,
            evaluator_name=self.name,
            score=score,
            details=details,
            reasoning=(
                f"Embedding drift: {relative_shift:+.1%} shift "
                f"(recent mean={recent_mean:.3f} vs "
                f"historical mean={historical_mean:.3f})"
            ),
        )

    @staticmethod
    def _classify_drift(
        relative_shift: float,
    ) -> tuple[bool, str, float]:
        """Classify the severity of drift.

        Args:
            relative_shift: Fractional shift (recent - historical) / historical.

        Returns:
            Tuple of (drift_detected, severity, score).
        """
        abs_shift = abs(relative_shift)
        if abs_shift > 0.25:
            return True, "high", 0.2
        if abs_shift > 0.15:
            return True, "medium", 0.5
        return False, "low", 1.0

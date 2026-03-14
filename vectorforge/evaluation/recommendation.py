"""Recommendation engine — analyzes evaluation results and generates improvement suggestions."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from vectorforge.evaluation.config import EvaluationConfig
from vectorforge.evaluation.types import (
    CreateRecommendationDTO,
    EvaluationResult,
    RecommendationCategory,
    RecommendationSeverity,
)

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Analyzes evaluation summaries and generates ranked improvement recommendations.

    Compares per-evaluator aggregate scores against configured thresholds
    and produces actionable recommendations with severity classification
    and supporting evidence.

    Args:
        config: Evaluation configuration containing threshold values.
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self._config = config or EvaluationConfig()

    def analyze(
        self,
        run_id: UUID,
        summary: dict[str, dict[str, Any]],
        results: list[EvaluationResult],
    ) -> list[CreateRecommendationDTO]:
        """Generate recommendations from evaluation summary and results.

        Args:
            run_id: The evaluation run UUID.
            summary: Per-evaluator aggregate scores (from EvaluationService._compute_summary).
            results: Individual evaluation results for evidence extraction.

        Returns:
            List of recommendation DTOs sorted by severity (critical first).
        """
        recommendations: list[CreateRecommendationDTO] = []

        recommendations.extend(self._check_retrieval_relevance(run_id, summary, results))
        recommendations.extend(self._check_chunk_coverage(run_id, summary, results))
        recommendations.extend(self._check_embedding_drift(run_id, summary))
        recommendations.extend(self._check_faithfulness(run_id, summary, results))
        recommendations.extend(self._check_hallucination(run_id, summary, results))
        recommendations.extend(self._check_answer_relevance(run_id, summary, results))
        recommendations.extend(self._check_cross_cutting(run_id, summary))

        severity_order = {
            RecommendationSeverity.CRITICAL: 0,
            RecommendationSeverity.HIGH: 1,
            RecommendationSeverity.MEDIUM: 2,
            RecommendationSeverity.LOW: 3,
        }
        recommendations.sort(key=lambda r: severity_order.get(r.severity, 99))

        logger.info(
            "Generated %d recommendations for run %s",
            len(recommendations),
            run_id,
        )
        return recommendations

    # ------------------------------------------------------------------
    # Per-evaluator checks
    # ------------------------------------------------------------------

    def _check_retrieval_relevance(
        self,
        run_id: UUID,
        summary: dict[str, dict[str, Any]],
        results: list[EvaluationResult],
    ) -> list[CreateRecommendationDTO]:
        scores = summary.get("retrieval_relevance")
        if not scores:
            return []
        avg = float(scores.get("avg", 1.0))
        threshold = self._config.relevance_threshold
        if avg >= threshold:
            return []
        return [
            CreateRecommendationDTO(
                run_id=run_id,
                category=RecommendationCategory.RETRIEVAL,
                severity=_classify_severity(avg, threshold),
                title="Low Retrieval Relevance",
                description=(
                    "Retrieved chunks have low relevance to queries. "
                    "Consider: (1) Switching embedding model, "
                    "(2) Adjusting chunk size/overlap, "
                    "(3) Adding metadata filters, "
                    "(4) Increasing top_k and adding re-ranking."
                ),
                evidence={
                    "avg_score": avg,
                    "threshold": threshold,
                    "worst_queries": _get_worst_n(results, "retrieval_relevance", 5),
                },
            )
        ]

    def _check_chunk_coverage(
        self,
        run_id: UUID,
        summary: dict[str, dict[str, Any]],
        results: list[EvaluationResult],
    ) -> list[CreateRecommendationDTO]:
        scores = summary.get("chunk_coverage")
        if not scores:
            return []
        avg = float(scores.get("avg", 1.0))
        threshold = self._config.coverage_threshold
        if avg >= threshold:
            return []
        return [
            CreateRecommendationDTO(
                run_id=run_id,
                category=RecommendationCategory.CHUNKING,
                severity=_classify_severity(avg, threshold),
                title="Incomplete Chunk Coverage",
                description=(
                    "Retrieved chunks don't cover all aspects of queries. "
                    "Consider: (1) Reducing chunk size for finer granularity, "
                    "(2) Increasing top_k to retrieve more chunks, "
                    "(3) Using semantic chunking for better boundaries, "
                    "(4) Adding hybrid search for keyword-based recall."
                ),
                evidence={
                    "avg_coverage": avg,
                    "threshold": threshold,
                    "worst_queries": _get_worst_n(results, "chunk_coverage", 5),
                },
            )
        ]

    def _check_embedding_drift(
        self,
        run_id: UUID,
        summary: dict[str, dict[str, Any]],
    ) -> list[CreateRecommendationDTO]:
        scores = summary.get("embedding_drift")
        if not scores:
            return []
        avg = float(scores.get("avg", 1.0))
        if avg >= 0.5:
            return []
        return [
            CreateRecommendationDTO(
                run_id=run_id,
                category=RecommendationCategory.EMBEDDING,
                severity=RecommendationSeverity.HIGH,
                title="Embedding Quality Drift Detected",
                description=(
                    "Similarity score distribution has shifted significantly. "
                    "Consider: (1) Re-embedding documents with current model version, "
                    "(2) Investigating data distribution changes, "
                    "(3) Upgrading embedding model."
                ),
                evidence=dict(scores),
            )
        ]

    def _check_faithfulness(
        self,
        run_id: UUID,
        summary: dict[str, dict[str, Any]],
        results: list[EvaluationResult],
    ) -> list[CreateRecommendationDTO]:
        scores = summary.get("faithfulness")
        if not scores:
            return []
        avg = float(scores.get("avg", 1.0))
        threshold = self._config.faithfulness_threshold
        if avg >= threshold:
            return []
        return [
            CreateRecommendationDTO(
                run_id=run_id,
                category=RecommendationCategory.GENERATION,
                severity=_classify_severity(avg, threshold),
                title="Low Answer Faithfulness",
                description=(
                    "Generated answers contain claims not supported by context. "
                    "Consider: (1) Strengthening system prompt grounding instructions, "
                    "(2) Reducing temperature, "
                    "(3) Using a more instruction-following model, "
                    "(4) Adding explicit citation requirements."
                ),
                evidence={
                    "avg_faithfulness": avg,
                    "threshold": threshold,
                    "worst_queries": _get_worst_n(results, "faithfulness", 5),
                },
            )
        ]

    def _check_hallucination(
        self,
        run_id: UUID,
        summary: dict[str, dict[str, Any]],
        results: list[EvaluationResult],
    ) -> list[CreateRecommendationDTO]:
        scores = summary.get("hallucination")
        if not scores:
            return []
        avg = float(scores.get("avg", 1.0))
        # hallucination_threshold is rate-based (e.g., 0.3); score is 1.0 - rate
        threshold_score = 1.0 - self._config.hallucination_threshold
        if avg >= threshold_score:
            return []
        hallucination_rate = 1.0 - avg
        severity = (
            RecommendationSeverity.CRITICAL
            if hallucination_rate > 0.5
            else RecommendationSeverity.HIGH
        )
        return [
            CreateRecommendationDTO(
                run_id=run_id,
                category=RecommendationCategory.GENERATION,
                severity=severity,
                title="Hallucination Rate Above Threshold",
                description=(
                    "Generated answers contain fabricated information. "
                    "URGENT: (1) Add explicit 'only use provided context' instruction, "
                    "(2) Reduce temperature to 0, "
                    "(3) Switch to a more grounded model, "
                    "(4) Add post-generation fact-checking step."
                ),
                evidence={
                    "hallucination_rate": round(hallucination_rate, 4),
                    "avg_score": avg,
                    "worst_queries": _get_worst_n(results, "hallucination", 3),
                },
            )
        ]

    def _check_answer_relevance(
        self,
        run_id: UUID,
        summary: dict[str, dict[str, Any]],
        results: list[EvaluationResult],
    ) -> list[CreateRecommendationDTO]:
        scores = summary.get("answer_relevance")
        if not scores:
            return []
        avg = float(scores.get("avg", 1.0))
        threshold = self._config.relevance_threshold
        if avg >= threshold:
            return []
        return [
            CreateRecommendationDTO(
                run_id=run_id,
                category=RecommendationCategory.GENERATION,
                severity=_classify_severity(avg, threshold),
                title="Low Answer Relevance",
                description=(
                    "Answers don't adequately address user queries. "
                    "Consider: (1) Improving system prompt with query-focus instructions, "
                    "(2) Adding query classification for better routing, "
                    "(3) Providing more relevant context (improve retrieval first)."
                ),
                evidence={
                    "avg_relevance": avg,
                    "threshold": threshold,
                    "worst_queries": _get_worst_n(results, "answer_relevance", 5),
                },
            )
        ]

    def _check_cross_cutting(
        self,
        run_id: UUID,
        summary: dict[str, dict[str, Any]],
    ) -> list[CreateRecommendationDTO]:
        retrieval = summary.get("retrieval_relevance", {})
        faithfulness = summary.get("faithfulness", {})
        ret_avg = float(retrieval.get("avg", 1.0))
        faith_avg = float(faithfulness.get("avg", 1.0))

        if ret_avg >= 0.5 or faith_avg >= 0.5:
            return []

        return [
            CreateRecommendationDTO(
                run_id=run_id,
                category=RecommendationCategory.RETRIEVAL,
                severity=RecommendationSeverity.CRITICAL,
                title="Systemic Quality Issue: Poor Retrieval Cascading to Poor Generation",
                description=(
                    "Both retrieval and generation quality are low. "
                    "The root cause is likely retrieval — the LLM can't produce "
                    "faithful answers from irrelevant context. "
                    "Priority: Fix retrieval first, then re-evaluate generation."
                ),
                evidence={
                    "retrieval_avg": ret_avg,
                    "faithfulness_avg": faith_avg,
                },
            )
        ]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _classify_severity(score: float, threshold: float) -> RecommendationSeverity:
    """Classify severity based on the gap between score and threshold.

    Args:
        score: The actual score (0-1).
        threshold: The required threshold (0-1).

    Returns:
        Severity level.
    """
    gap = threshold - score
    if gap > 0.3:
        return RecommendationSeverity.CRITICAL
    if gap > 0.15:
        return RecommendationSeverity.HIGH
    if gap > 0.05:
        return RecommendationSeverity.MEDIUM
    return RecommendationSeverity.LOW


def _get_worst_n(
    results: list[EvaluationResult],
    evaluator_name: str,
    n: int,
) -> list[dict[str, Any]]:
    """Extract the N worst-scoring results for a given evaluator.

    Args:
        results: All evaluation results.
        evaluator_name: Name of the evaluator to filter by.
        n: Number of worst results to return.

    Returns:
        List of dicts with query_log_id, score, and reasoning.
    """
    filtered = [
        r for r in results
        if r.evaluator_name == evaluator_name and r.score is not None
    ]
    filtered.sort(key=lambda r: r.score or 0.0)
    return [
        {
            "query_log_id": str(r.query_log_id),
            "score": r.score,
            "reasoning": r.reasoning,
        }
        for r in filtered[:n]
    ]

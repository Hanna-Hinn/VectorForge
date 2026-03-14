"""Evaluation data models — samples, results, runs, and recommendations."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvaluationRunStatus(enum.StrEnum):
    """Status of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RecommendationSeverity(enum.StrEnum):
    """Severity level of a recommendation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationCategory(enum.StrEnum):
    """Category of a recommendation."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"


class RecommendationStatus(enum.StrEnum):
    """Lifecycle status of a recommendation."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


# ---------------------------------------------------------------------------
# Input / Intermediate Models
# ---------------------------------------------------------------------------


class EvaluationSample(BaseModel):
    """A single query-answer pair to be evaluated."""

    query_log_id: UUID
    query: str
    chunks: list[ChunkWithScore]
    answer: str
    ground_truth: str | None = None


class ChunkWithScore(BaseModel):
    """Minimal chunk representation for evaluation context."""

    chunk_id: UUID
    text: str
    score: float
    document_source: str = ""


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Output of a single evaluator on a single sample."""

    query_log_id: UUID
    evaluator_name: str
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None


# ---------------------------------------------------------------------------
# Domain Models (read representations)
# ---------------------------------------------------------------------------


class EvaluationRun(BaseModel):
    """A completed (or in-progress) evaluation run."""

    id: UUID
    status: EvaluationRunStatus = EvaluationRunStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    sample_size: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    summary_scores: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class Recommendation(BaseModel):
    """An actionable recommendation from an evaluation run."""

    id: UUID
    run_id: UUID
    category: RecommendationCategory
    severity: RecommendationSeverity
    title: str
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: RecommendationStatus = RecommendationStatus.PENDING
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class EvaluationResultRead(BaseModel):
    """Persisted evaluation result (read model)."""

    id: UUID
    run_id: UUID
    query_log_id: UUID
    evaluator_name: str
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# DTOs (Create / Update)
# ---------------------------------------------------------------------------


class CreateEvaluationRunDTO(BaseModel):
    """Data required to create an evaluation run."""

    status: EvaluationRunStatus = EvaluationRunStatus.PENDING
    sample_size: int = 0
    config: dict[str, Any] = Field(default_factory=dict)


class CreateEvaluationResultDTO(BaseModel):
    """Data required to store a single evaluation result."""

    run_id: UUID
    query_log_id: UUID
    evaluator_name: str
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None


class CreateRecommendationDTO(BaseModel):
    """Data required to create a recommendation."""

    run_id: UUID
    category: RecommendationCategory
    severity: RecommendationSeverity
    title: str
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class UpdateRecommendationStatusDTO(BaseModel):
    """Partial update for recommendation status."""

    status: RecommendationStatus


# ---------------------------------------------------------------------------
# Report Models
# ---------------------------------------------------------------------------


class ScoreSummaryRow(BaseModel):
    """Per-evaluator score summary for a run."""

    evaluator: str
    avg: float
    min_score: float
    max_score: float
    p50: float
    below_threshold: int
    sample_count: int
    status: str  # "pass" | "fail"


class TrendData(BaseModel):
    """Score trend over recent evaluation runs."""

    evaluator: str
    scores: list[float]
    direction: str  # "improving" | "stable" | "degrading"
    change_pct: float


class WorstQuery(BaseModel):
    """A poorly-performing query from an evaluation run."""

    query_log_id: UUID
    query: str
    composite_score: float
    per_evaluator_scores: dict[str, float]
    key_issues: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Full evaluation report aggregating all run data."""

    header: dict[str, Any]
    score_summary: list[ScoreSummaryRow]
    trends: list[TrendData]
    recommendations: list[Recommendation]
    worst_queries: list[WorstQuery]
    raw_result_count: int

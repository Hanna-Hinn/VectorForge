"""Evaluation repositories — data access for evaluation runs, results, and recommendations."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from vectorforge.db.repositories.base import BaseRepository
from vectorforge.evaluation.types import (
    CreateEvaluationResultDTO,
    CreateEvaluationRunDTO,
    CreateRecommendationDTO,
    EvaluationResultRead,
    EvaluationRun,
    Recommendation,
    UpdateRecommendationStatusDTO,
)
from vectorforge.exceptions import NotFoundError
from vectorforge.models.db import (
    EvaluationResultModel,
    EvaluationRunModel,
    RecommendationModel,
)


class EvaluationRunRepository(BaseRepository[EvaluationRun]):
    """Repository for managing evaluation run records."""

    _model_class = EvaluationRunModel

    def _to_domain(self, instance: EvaluationRunModel) -> EvaluationRun:
        """Convert an EvaluationRunModel ORM instance to an EvaluationRun domain model."""
        return EvaluationRun(
            id=instance.id,
            status=instance.status,
            started_at=instance.started_at,
            completed_at=instance.completed_at,
            sample_size=instance.sample_size,
            config=instance.config or {},
            summary_scores=instance.summary_scores or {},
            error_message=instance.error_message,
            created_at=instance.created_at,
        )

    async def create(self, data: CreateEvaluationRunDTO) -> EvaluationRun:
        """Insert a new evaluation run record.

        Args:
            data: A CreateEvaluationRunDTO with run creation fields.

        Returns:
            The newly created EvaluationRun domain model.
        """
        instance = EvaluationRunModel(**data.model_dump())
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def update_status(
        self,
        run_id: uuid.UUID,
        *,
        status: str,
        summary_scores: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> EvaluationRun:
        """Update evaluation run status and optional summary fields.

        Args:
            run_id: The UUID of the evaluation run.
            status: New status value.
            summary_scores: Aggregated scores (set on completion).
            error_message: Error details (set on failure).

        Returns:
            The updated EvaluationRun domain model.

        Raises:
            NotFoundError: If no run exists with the given id.
        """
        from datetime import UTC, datetime

        result = await self._session.execute(
            select(EvaluationRunModel).where(EvaluationRunModel.id == run_id)
        )
        instance = result.scalar_one_or_none()
        if instance is None:
            msg = f"EvaluationRunModel with id={run_id} not found"
            raise NotFoundError(msg)

        instance.status = status
        if status == "running" and instance.started_at is None:
            instance.started_at = datetime.now(UTC)
        if status in {"completed", "failed"}:
            instance.completed_at = datetime.now(UTC)
        if summary_scores is not None:
            instance.summary_scores = summary_scores
        if error_message is not None:
            instance.error_message = error_message

        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def find_recent(self, limit: int = 10) -> list[EvaluationRun]:
        """Find the most recent evaluation runs.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            List of EvaluationRun ordered by most recent first.
        """
        result = await self._session.execute(
            select(EvaluationRunModel)
            .order_by(EvaluationRunModel.created_at.desc())
            .limit(limit)
        )
        return [self._to_domain(row) for row in result.scalars().all()]


class EvaluationResultRepository(BaseRepository[EvaluationResultRead]):
    """Repository for managing evaluation result records."""

    _model_class = EvaluationResultModel

    def _to_domain(self, instance: EvaluationResultModel) -> EvaluationResultRead:
        """Convert an EvaluationResultModel ORM instance to an EvaluationResultRead."""
        return EvaluationResultRead(
            id=instance.id,
            run_id=instance.run_id,
            query_log_id=instance.query_log_id,
            evaluator_name=instance.evaluator_name,
            score=instance.score,
            details=instance.details or {},
            reasoning=instance.reasoning,
            created_at=instance.created_at,
        )

    async def create(self, data: CreateEvaluationResultDTO) -> EvaluationResultRead:
        """Insert a new evaluation result record.

        Args:
            data: A CreateEvaluationResultDTO with result fields.

        Returns:
            The newly created EvaluationResultRead domain model.
        """
        instance = EvaluationResultModel(**data.model_dump())
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def create_batch(
        self, items: list[CreateEvaluationResultDTO]
    ) -> list[EvaluationResultRead]:
        """Insert multiple evaluation results in a single flush.

        Args:
            items: List of DTOs to insert.

        Returns:
            List of newly created EvaluationResultRead domain models.
        """
        instances = [EvaluationResultModel(**dto.model_dump()) for dto in items]
        self._session.add_all(instances)
        await self._session.flush()
        for inst in instances:
            await self._session.refresh(inst)
        return [self._to_domain(inst) for inst in instances]

    async def find_by_run(
        self,
        run_id: uuid.UUID,
        limit: int = 500,
        offset: int = 0,
    ) -> list[EvaluationResultRead]:
        """Find all evaluation results for a specific run.

        Args:
            run_id: The evaluation run UUID.
            limit: Maximum results to return.
            offset: Number of records to skip.

        Returns:
            List of EvaluationResultRead ordered by creation time.
        """
        result = await self._session.execute(
            select(EvaluationResultModel)
            .where(EvaluationResultModel.run_id == run_id)
            .order_by(EvaluationResultModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [self._to_domain(row) for row in result.scalars().all()]

    async def find_by_evaluator(
        self,
        run_id: uuid.UUID,
        evaluator_name: str,
    ) -> list[EvaluationResultRead]:
        """Find results for a specific evaluator within a run.

        Args:
            run_id: The evaluation run UUID.
            evaluator_name: Name of the evaluator.

        Returns:
            List of EvaluationResultRead for the given evaluator.
        """
        result = await self._session.execute(
            select(EvaluationResultModel)
            .where(
                EvaluationResultModel.run_id == run_id,
                EvaluationResultModel.evaluator_name == evaluator_name,
            )
            .order_by(EvaluationResultModel.created_at.desc())
        )
        return [self._to_domain(row) for row in result.scalars().all()]


class RecommendationRepository(BaseRepository[Recommendation]):
    """Repository for managing recommendation records."""

    _model_class = RecommendationModel

    def _to_domain(self, instance: RecommendationModel) -> Recommendation:
        """Convert a RecommendationModel ORM instance to a Recommendation domain model."""
        return Recommendation(
            id=instance.id,
            run_id=instance.run_id,
            category=instance.category,
            severity=instance.severity,
            title=instance.title,
            description=instance.description,
            evidence=instance.evidence or {},
            status=instance.status,
            created_at=instance.created_at,
        )

    async def create(self, data: CreateRecommendationDTO) -> Recommendation:
        """Insert a new recommendation record.

        Args:
            data: A CreateRecommendationDTO with recommendation fields.

        Returns:
            The newly created Recommendation domain model.
        """
        instance = RecommendationModel(**data.model_dump())
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def find_by_run(self, run_id: uuid.UUID) -> list[Recommendation]:
        """Find all recommendations for a specific evaluation run.

        Args:
            run_id: The evaluation run UUID.

        Returns:
            List of Recommendations ordered by severity.
        """
        severity_order = ["critical", "high", "medium", "low"]
        result = await self._session.execute(
            select(RecommendationModel)
            .where(RecommendationModel.run_id == run_id)
            .order_by(RecommendationModel.created_at.desc())
        )
        rows = [self._to_domain(row) for row in result.scalars().all()]
        return sorted(rows, key=lambda r: severity_order.index(r.severity))

    async def update_status(
        self, rec_id: uuid.UUID, data: UpdateRecommendationStatusDTO
    ) -> Recommendation:
        """Update the status of a recommendation.

        Args:
            rec_id: The recommendation UUID.
            data: DTO containing the new status.

        Returns:
            The updated Recommendation domain model.

        Raises:
            NotFoundError: If no recommendation exists with the given id.
        """
        result = await self._session.execute(
            select(RecommendationModel).where(RecommendationModel.id == rec_id)
        )
        instance = result.scalar_one_or_none()
        if instance is None:
            msg = f"RecommendationModel with id={rec_id} not found"
            raise NotFoundError(msg)

        instance.status = data.status
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def find_pending(self, limit: int = 50) -> list[Recommendation]:
        """Find pending recommendations across all runs.

        Args:
            limit: Maximum number of recommendations to return.

        Returns:
            List of pending Recommendations.
        """
        result = await self._session.execute(
            select(RecommendationModel)
            .where(RecommendationModel.status == "pending")
            .order_by(RecommendationModel.created_at.desc())
            .limit(limit)
        )
        return [self._to_domain(row) for row in result.scalars().all()]

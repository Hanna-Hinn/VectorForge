"""Evaluation endpoints for running and viewing RAG quality evaluations."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from server.dependencies import ApiKey, DbSession
from server.schemas import (
    EvaluationResultListResponse,
    EvaluationResultResponse,
    EvaluationRunListResponse,
    EvaluationRunResponse,
    RecommendationListResponse,
    RecommendationResponse,
    RunEvaluationRequest,
    TrendDataResponse,
    TrendListResponse,
    UpdateRecommendationRequest,
)
from vectorforge.evaluation.config import EvaluationConfig
from vectorforge.evaluation.recommendation import RecommendationEngine
from vectorforge.evaluation.registry import EvaluatorRegistry
from vectorforge.evaluation.report import EvaluationReportBuilder
from vectorforge.evaluation.service import EvaluationService
from vectorforge.evaluation.types import (
    EvaluationResult,
    EvaluationRun,
    RecommendationStatus,
    UpdateRecommendationStatusDTO,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post(
    "/run",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_evaluation(
    session: DbSession,
    _key: ApiKey,
    body: RunEvaluationRequest | None = None,
) -> EvaluationRunResponse:
    """Trigger an evaluation run."""
    config = EvaluationConfig()
    if body:
        overrides: dict[str, Any] = {}
        if body.sample_size is not None:
            overrides["sample_size"] = body.sample_size
        if body.sample_strategy is not None:
            overrides["sample_strategy"] = body.sample_strategy
        if overrides:
            config = EvaluationConfig(**overrides)

    registry = EvaluatorRegistry()
    registry.register_defaults()
    service = EvaluationService(session, registry, config)

    run = await service.run_evaluation()
    await session.commit()

    # Generate recommendations
    from vectorforge.db.repositories.evaluation_repo import (
        EvaluationResultRepository,
        RecommendationRepository,
    )

    result_repo = EvaluationResultRepository(session)
    rec_repo = RecommendationRepository(session)
    result_models = await result_repo.find_by_run(run.id)
    eval_results = [
        EvaluationResult(
            query_log_id=r.query_log_id,
            evaluator_name=r.evaluator_name,
            score=r.score,
            details=r.details,
            reasoning=r.reasoning,
        )
        for r in result_models
    ]
    engine = RecommendationEngine(config)
    rec_dtos = engine.analyze(run.id, run.summary_scores, eval_results)
    for dto in rec_dtos:
        await rec_repo.create(dto)
    await session.commit()

    return _run_to_response(run)


@router.get("/runs", response_model=EvaluationRunListResponse)
async def list_runs(
    session: DbSession,
    _key: ApiKey,
    limit: int = Query(default=10, ge=1, le=100),
) -> EvaluationRunListResponse:
    """List recent evaluation runs."""
    from vectorforge.db.repositories.evaluation_repo import EvaluationRunRepository

    repo = EvaluationRunRepository(session)
    runs = await repo.find_recent(limit=limit)
    return EvaluationRunListResponse(
        runs=[_run_model_to_response(r) for r in runs],
    )


@router.get("/runs/{run_id}", response_model=EvaluationRunResponse)
async def get_run(
    run_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
) -> EvaluationRunResponse:
    """Get detailed evaluation run."""
    from vectorforge.db.repositories.evaluation_repo import EvaluationRunRepository

    repo = EvaluationRunRepository(session)
    runs = await repo.find_recent(limit=100)
    target = next((r for r in runs if r.id == run_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return _run_model_to_response(target)


@router.get("/runs/{run_id}/results", response_model=EvaluationResultListResponse)
async def get_run_results(
    run_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
    evaluator: str | None = Query(default=None),
) -> EvaluationResultListResponse:
    """Get individual evaluation results for a run."""
    from vectorforge.db.repositories.evaluation_repo import EvaluationResultRepository

    repo = EvaluationResultRepository(session)
    if evaluator:
        results = await repo.find_by_evaluator(run_id, evaluator)
    else:
        results = await repo.find_by_run(run_id)

    return EvaluationResultListResponse(
        results=[
            EvaluationResultResponse(
                id=str(r.id),
                run_id=str(r.run_id),
                query_log_id=str(r.query_log_id),
                evaluator_name=r.evaluator_name,
                score=r.score,
                details=r.details or {},
                reasoning=r.reasoning,
            )
            for r in results
        ],
    )


@router.get("/recommendations", response_model=RecommendationListResponse)
async def list_recommendations(
    session: DbSession,
    _key: ApiKey,
    rec_status: str | None = Query(default=None, alias="status"),
    category: str | None = Query(default=None),
) -> RecommendationListResponse:
    """List evaluation recommendations."""
    from sqlalchemy import select

    from vectorforge.models.db import RecommendationModel

    stmt = select(RecommendationModel).order_by(
        RecommendationModel.created_at.desc()
    )
    if rec_status:
        stmt = stmt.where(RecommendationModel.status == rec_status)
    if category:
        stmt = stmt.where(RecommendationModel.category == category)
    stmt = stmt.limit(100)

    result = await session.execute(stmt)
    recs = list(result.scalars().all())

    return RecommendationListResponse(
        recommendations=[_rec_model_to_response(r) for r in recs],
    )


@router.patch(
    "/recommendations/{rec_id}",
    response_model=RecommendationResponse,
)
async def update_recommendation(
    rec_id: uuid.UUID,
    body: UpdateRecommendationRequest,
    session: DbSession,
    _key: ApiKey,
) -> RecommendationResponse:
    """Update a recommendation's status."""
    try:
        new_status = RecommendationStatus(body.status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid status: {body.status}. "
                "Must be one of: pending, acknowledged, resolved, dismissed"
            ),
        ) from None

    from vectorforge.db.repositories.evaluation_repo import RecommendationRepository

    repo = RecommendationRepository(session)
    updated = await repo.update_status(rec_id, UpdateRecommendationStatusDTO(status=new_status))
    if updated is None:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    await session.commit()
    return _rec_model_to_response(updated)


@router.get("/trends", response_model=TrendListResponse)
async def get_trends(
    session: DbSession,
    _key: ApiKey,
    limit: int = Query(default=10, ge=2, le=50),
) -> TrendListResponse:
    """Get score trends over recent evaluation runs."""
    from vectorforge.db.repositories.evaluation_repo import EvaluationRunRepository

    repo = EvaluationRunRepository(session)
    runs = await repo.find_recent(limit=limit)
    run_domains = [EvaluationRun.model_validate(r) for r in runs]

    if not run_domains:
        return TrendListResponse(trends=[])

    builder = EvaluationReportBuilder()
    current = run_domains[0]
    previous = run_domains[1:] if len(run_domains) > 1 else []
    trends = builder._build_trends(current, previous)

    return TrendListResponse(
        trends=[
            TrendDataResponse(
                evaluator=t.evaluator,
                scores=t.scores,
                direction=t.direction,
                change_pct=t.change_pct,
            )
            for t in trends
        ],
    )


# ---------------------------------------------------------------------------
# Response converters
# ---------------------------------------------------------------------------


def _run_to_response(run: EvaluationRun) -> EvaluationRunResponse:
    """Convert domain EvaluationRun to response."""
    return EvaluationRunResponse(
        run_id=str(run.id),
        status=str(run.status),
        sample_size=run.sample_size,
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        summary_scores=run.summary_scores,
        created_at=run.created_at.isoformat() if run.created_at else None,
    )


def _run_model_to_response(run: Any) -> EvaluationRunResponse:
    """Convert ORM EvaluationRunModel to response."""
    return EvaluationRunResponse(
        run_id=str(run.id),
        status=str(run.status),
        sample_size=run.sample_size,
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        summary_scores=run.summary_scores or {},
        created_at=run.created_at.isoformat() if run.created_at else None,
    )


def _rec_model_to_response(rec: Any) -> RecommendationResponse:
    """Convert ORM RecommendationModel to response."""
    return RecommendationResponse(
        id=str(rec.id),
        run_id=str(rec.run_id),
        category=str(rec.category),
        severity=str(rec.severity),
        title=rec.title,
        description=rec.description,
        evidence=rec.evidence or {},
        status=str(rec.status),
    )

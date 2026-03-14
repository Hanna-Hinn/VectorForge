"""Unit tests for VectorForge evaluation infrastructure (Stage 6A)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from vectorforge.evaluation.base import BaseEvaluator
from vectorforge.evaluation.config import EvaluationConfig
from vectorforge.evaluation.registry import EvaluatorRegistry
from vectorforge.evaluation.scheduler import BackgroundScheduler
from vectorforge.evaluation.types import (
    ChunkWithScore,
    CreateEvaluationResultDTO,
    CreateEvaluationRunDTO,
    CreateRecommendationDTO,
    EvaluationReport,
    EvaluationResult,
    EvaluationResultRead,
    EvaluationRun,
    EvaluationRunStatus,
    EvaluationSample,
    Recommendation,
    RecommendationCategory,
    RecommendationSeverity,
    RecommendationStatus,
    ScoreSummaryRow,
    TrendData,
    UpdateRecommendationStatusDTO,
    WorstQuery,
)
from vectorforge.exceptions import ConfigurationError, DuplicateError, EvaluationError

# ---------------------------------------------------------------------------
# Fake evaluator for testing
# ---------------------------------------------------------------------------


class FakeEvaluator(BaseEvaluator):
    """Test double returning predictable scores."""

    def __init__(self, score: float = 0.8) -> None:
        self._score = score

    @property
    def name(self) -> str:
        return "fake_evaluator"

    @property
    def category(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "A fake evaluator for testing."

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        return EvaluationResult(
            query_log_id=sample.query_log_id,
            evaluator_name=self.name,
            score=self._score,
            details={"test": True},
            reasoning="Fake reasoning",
        )


class FakeGenerationEvaluator(BaseEvaluator):
    """Test double for generation-category evaluator."""

    @property
    def name(self) -> str:
        return "fake_generation"

    @property
    def category(self) -> str:
        return "generation"

    @property
    def description(self) -> str:
        return "Fake generation evaluator."

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        return EvaluationResult(
            query_log_id=sample.query_log_id,
            evaluator_name=self.name,
            score=0.9,
        )


class FailingEvaluator(BaseEvaluator):
    """Test double that always raises."""

    @property
    def name(self) -> str:
        return "failing_evaluator"

    @property
    def category(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "Always fails."

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        msg = "Simulated failure"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    query: str = "test query",
    answer: str = "test answer",
    chunk_count: int = 2,
) -> EvaluationSample:
    """Build a minimal EvaluationSample for testing."""
    chunks = [
        ChunkWithScore(
            chunk_id=uuid4(),
            text=f"chunk {i}",
            score=0.9 - i * 0.1,
        )
        for i in range(chunk_count)
    ]
    return EvaluationSample(
        query_log_id=uuid4(),
        query=query,
        chunks=chunks,
        answer=answer,
    )


# ---------------------------------------------------------------------------
# EvaluationConfig tests
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    """Tests for EvaluationConfig validation and defaults."""

    def test_defaults(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.enabled is False
        assert cfg.schedule_interval_hours == 24
        assert cfg.sample_size == 50
        assert cfg.sample_strategy == "recent"
        assert cfg.max_concurrent_evaluators == 3
        assert cfg.evaluation_timeout_seconds == 300
        assert cfg.judge_provider == "openai"
        assert cfg.judge_model == "gpt-4o-mini"

    def test_thresholds_default(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.faithfulness_threshold == 0.7
        assert cfg.relevance_threshold == 0.6
        assert cfg.hallucination_threshold == 0.3
        assert cfg.coverage_threshold == 0.5

    def test_valid_overrides(self) -> None:
        cfg = EvaluationConfig(
            enabled=True,
            sample_size=100,
            schedule_interval_hours=6,
            faithfulness_threshold=0.8,
        )
        assert cfg.enabled is True
        assert cfg.sample_size == 100
        assert cfg.schedule_interval_hours == 6
        assert cfg.faithfulness_threshold == 0.8

    def test_invalid_sample_size(self) -> None:
        with pytest.raises(ValueError):
            EvaluationConfig(sample_size=0)

    def test_invalid_schedule_interval(self) -> None:
        with pytest.raises(ValueError):
            EvaluationConfig(schedule_interval_hours=0)

    def test_invalid_threshold_above_1(self) -> None:
        with pytest.raises(ValueError):
            EvaluationConfig(faithfulness_threshold=1.5)

    def test_invalid_threshold_below_0(self) -> None:
        with pytest.raises(ValueError):
            EvaluationConfig(relevance_threshold=-0.1)

    def test_invalid_sample_strategy(self) -> None:
        with pytest.raises(ValueError):
            EvaluationConfig(sample_strategy="invalid")


# ---------------------------------------------------------------------------
# Evaluation types tests
# ---------------------------------------------------------------------------


class TestEvaluationTypes:
    """Tests for evaluation data models."""

    def test_evaluation_sample_creation(self) -> None:
        sample = _make_sample()
        assert sample.query == "test query"
        assert sample.answer == "test answer"
        assert len(sample.chunks) == 2
        assert sample.ground_truth is None

    def test_evaluation_result_creation(self) -> None:
        result = EvaluationResult(
            query_log_id=uuid4(),
            evaluator_name="test",
            score=0.85,
            details={"key": "value"},
            reasoning="good",
        )
        assert result.score == 0.85
        assert result.evaluator_name == "test"

    def test_evaluation_result_optional_fields(self) -> None:
        result = EvaluationResult(
            query_log_id=uuid4(),
            evaluator_name="test",
        )
        assert result.score is None
        assert result.details == {}
        assert result.reasoning is None

    def test_evaluation_run_status_enum(self) -> None:
        assert EvaluationRunStatus.PENDING == "pending"
        assert EvaluationRunStatus.RUNNING == "running"
        assert EvaluationRunStatus.COMPLETED == "completed"
        assert EvaluationRunStatus.FAILED == "failed"

    def test_recommendation_severity_enum(self) -> None:
        assert RecommendationSeverity.CRITICAL == "critical"
        assert RecommendationSeverity.LOW == "low"

    def test_recommendation_category_enum(self) -> None:
        assert RecommendationCategory.RETRIEVAL == "retrieval"
        assert RecommendationCategory.EMBEDDING == "embedding"

    def test_recommendation_status_enum(self) -> None:
        assert RecommendationStatus.PENDING == "pending"
        assert RecommendationStatus.DISMISSED == "dismissed"

    def test_chunk_with_score(self) -> None:
        chunk = ChunkWithScore(
            chunk_id=uuid4(), text="hello", score=0.95
        )
        assert chunk.document_source == ""
        assert chunk.score == 0.95

    def test_create_evaluation_run_dto(self) -> None:
        dto = CreateEvaluationRunDTO(sample_size=25)
        assert dto.status == EvaluationRunStatus.PENDING
        assert dto.config == {}

    def test_create_evaluation_result_dto(self) -> None:
        run_id = uuid4()
        dto = CreateEvaluationResultDTO(
            run_id=run_id,
            query_log_id=uuid4(),
            evaluator_name="test",
            score=0.7,
        )
        assert dto.run_id == run_id
        assert dto.details == {}

    def test_create_recommendation_dto(self) -> None:
        dto = CreateRecommendationDTO(
            run_id=uuid4(),
            category=RecommendationCategory.RETRIEVAL,
            severity=RecommendationSeverity.HIGH,
            title="Test",
            description="Test desc",
        )
        assert dto.evidence == {}

    def test_update_recommendation_status_dto(self) -> None:
        dto = UpdateRecommendationStatusDTO(
            status=RecommendationStatus.ACKNOWLEDGED,
        )
        assert dto.status == RecommendationStatus.ACKNOWLEDGED

    def test_evaluation_run_from_attributes(self) -> None:
        run = EvaluationRun(
            id=uuid4(),
            status=EvaluationRunStatus.COMPLETED,
            sample_size=50,
        )
        assert run.summary_scores == {}
        assert run.error_message is None

    def test_recommendation_from_attributes(self) -> None:
        rec = Recommendation(
            id=uuid4(),
            run_id=uuid4(),
            category=RecommendationCategory.GENERATION,
            severity=RecommendationSeverity.MEDIUM,
            title="Fix it",
            description="Details here",
        )
        assert rec.status == RecommendationStatus.PENDING

    def test_evaluation_result_read(self) -> None:
        result = EvaluationResultRead(
            id=uuid4(),
            run_id=uuid4(),
            query_log_id=uuid4(),
            evaluator_name="test",
            score=0.5,
        )
        assert result.details == {}

    def test_score_summary_row(self) -> None:
        row = ScoreSummaryRow(
            evaluator="test",
            avg=0.8,
            min_score=0.5,
            max_score=1.0,
            p50=0.8,
            below_threshold=2,
            sample_count=10,
            status="pass",
        )
        assert row.evaluator == "test"

    def test_trend_data(self) -> None:
        trend = TrendData(
            evaluator="test",
            scores=[0.7, 0.8, 0.9],
            direction="improving",
            change_pct=28.57,
        )
        assert trend.direction == "improving"

    def test_worst_query(self) -> None:
        wq = WorstQuery(
            query_log_id=uuid4(),
            query="bad query",
            composite_score=0.2,
            per_evaluator_scores={"test": 0.2},
        )
        assert wq.key_issues == []

    def test_evaluation_report(self) -> None:
        report = EvaluationReport(
            header={"run_id": str(uuid4())},
            score_summary=[],
            trends=[],
            recommendations=[],
            worst_queries=[],
            raw_result_count=0,
        )
        assert report.raw_result_count == 0


# ---------------------------------------------------------------------------
# BaseEvaluator tests
# ---------------------------------------------------------------------------


class TestBaseEvaluator:
    """Tests for BaseEvaluator ABC and default batch behavior."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self) -> None:
        evaluator = FakeEvaluator(score=0.75)
        sample = _make_sample()
        result = await evaluator.evaluate(sample)
        assert result.score == 0.75
        assert result.evaluator_name == "fake_evaluator"

    @pytest.mark.asyncio
    async def test_evaluate_batch_default(self) -> None:
        evaluator = FakeEvaluator(score=0.6)
        samples = [_make_sample() for _ in range(3)]
        results = await evaluator.evaluate_batch(samples)
        assert len(results) == 3
        assert all(r.score == 0.6 for r in results)

    def test_abstract_properties(self) -> None:
        evaluator = FakeEvaluator()
        assert evaluator.name == "fake_evaluator"
        assert evaluator.category == "retrieval"
        assert evaluator.description == "A fake evaluator for testing."


# ---------------------------------------------------------------------------
# EvaluatorRegistry tests
# ---------------------------------------------------------------------------


class TestEvaluatorRegistry:
    """Tests for EvaluatorRegistry registration and lookup."""

    def test_register_and_get(self) -> None:
        registry = EvaluatorRegistry()
        evaluator = FakeEvaluator()
        registry.register(evaluator)
        retrieved = registry.get("fake_evaluator")
        assert retrieved is evaluator

    def test_register_duplicate_raises(self) -> None:
        registry = EvaluatorRegistry()
        registry.register(FakeEvaluator())
        with pytest.raises(DuplicateError, match="already registered"):
            registry.register(FakeEvaluator())

    def test_get_unknown_raises(self) -> None:
        registry = EvaluatorRegistry()
        with pytest.raises(ConfigurationError, match="not registered"):
            registry.get("nonexistent")

    def test_list_available(self) -> None:
        registry = EvaluatorRegistry()
        assert registry.list_available() == []
        registry.register(FakeEvaluator())
        assert registry.list_available() == ["fake_evaluator"]

    def test_get_by_category(self) -> None:
        registry = EvaluatorRegistry()
        registry.register(FakeEvaluator())
        registry.register(FakeGenerationEvaluator())
        retrieval = registry.get_by_category("retrieval")
        generation = registry.get_by_category("generation")
        assert len(retrieval) == 1
        assert len(generation) == 1
        assert retrieval[0].name == "fake_evaluator"
        assert generation[0].name == "fake_generation"

    def test_get_by_category_empty(self) -> None:
        registry = EvaluatorRegistry()
        assert registry.get_by_category("retrieval") == []


# ---------------------------------------------------------------------------
# EvaluationService tests
# ---------------------------------------------------------------------------


class TestEvaluationService:
    """Tests for EvaluationService orchestration logic."""

    @pytest.mark.asyncio
    async def test_run_evaluation_no_samples(self) -> None:
        from vectorforge.evaluation.service import EvaluationService

        session = AsyncMock()
        # Make the query return empty
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=execute_result)

        # Mock the run repo
        run_id = uuid4()
        created_run = EvaluationRun(
            id=run_id,
            status=EvaluationRunStatus.RUNNING,
            sample_size=0,
        )
        completed_run = EvaluationRun(
            id=run_id,
            status=EvaluationRunStatus.COMPLETED,
            sample_size=0,
            summary_scores={"_note": "no_samples"},
        )

        registry = EvaluatorRegistry()

        with (
            patch.object(
                EvaluationService, "__init__", lambda self, *a, **k: None
            ),
        ):
            service = EvaluationService.__new__(EvaluationService)
            service._session = session
            service._registry = registry
            service._config = EvaluationConfig()
            service._run_repo = AsyncMock()
            service._run_repo.create = AsyncMock(return_value=created_run)
            service._run_repo.update_status = AsyncMock(return_value=completed_run)
            service._result_repo = AsyncMock()
            service._rec_repo = AsyncMock()
            service._query_log_repo = AsyncMock()

            result = await service.run_evaluation()
            assert result.status == EvaluationRunStatus.COMPLETED
            assert result.summary_scores == {"_note": "no_samples"}

    @pytest.mark.asyncio
    async def test_run_evaluation_with_evaluators(self) -> None:
        from vectorforge.evaluation.service import EvaluationService

        session = AsyncMock()
        registry = EvaluatorRegistry()
        registry.register(FakeEvaluator(score=0.8))

        run_id = uuid4()
        created_run = EvaluationRun(
            id=run_id, status=EvaluationRunStatus.RUNNING, sample_size=2
        )
        completed_run = EvaluationRun(
            id=run_id,
            status=EvaluationRunStatus.COMPLETED,
            sample_size=2,
            summary_scores={"fake_evaluator": {"avg": 0.8}},
        )

        samples = [_make_sample(), _make_sample()]

        with patch.object(
            EvaluationService, "__init__", lambda self, *a, **k: None
        ):
            service = EvaluationService.__new__(EvaluationService)
            service._session = session
            service._registry = registry
            service._config = EvaluationConfig()
            service._run_repo = AsyncMock()
            service._run_repo.create = AsyncMock(return_value=created_run)
            service._run_repo.update_status = AsyncMock(return_value=completed_run)
            service._result_repo = AsyncMock()
            service._result_repo.create_batch = AsyncMock(return_value=[])
            service._rec_repo = AsyncMock()
            service._query_log_repo = AsyncMock()

            # Mock the private methods to control flow
            service._sample_queries = AsyncMock(return_value=["ql1", "ql2"])
            service._enrich_samples = AsyncMock(return_value=samples)

            result = await service.run_evaluation()
            assert result.status == EvaluationRunStatus.COMPLETED
            service._result_repo.create_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_evaluation_failure_marks_failed(self) -> None:
        from vectorforge.evaluation.service import EvaluationService

        run_id = uuid4()
        created_run = EvaluationRun(
            id=run_id, status=EvaluationRunStatus.RUNNING, sample_size=0
        )
        failed_run = EvaluationRun(
            id=run_id,
            status=EvaluationRunStatus.FAILED,
            error_message="boom",
        )

        with patch.object(
            EvaluationService, "__init__", lambda self, *a, **k: None
        ):
            service = EvaluationService.__new__(EvaluationService)
            service._session = AsyncMock()
            service._registry = EvaluatorRegistry()
            service._config = EvaluationConfig()
            service._run_repo = AsyncMock()
            service._run_repo.create = AsyncMock(return_value=created_run)
            service._run_repo.update_status = AsyncMock(return_value=failed_run)
            service._result_repo = AsyncMock()
            service._rec_repo = AsyncMock()
            service._query_log_repo = AsyncMock()

            # Make _sample_queries blow up
            service._sample_queries = AsyncMock(side_effect=RuntimeError("boom"))

            with pytest.raises(EvaluationError, match="boom"):
                await service.run_evaluation()

            service._run_repo.update_status.assert_called_with(
                run_id, status="failed", error_message="boom"
            )

    def test_compute_summary(self) -> None:
        from vectorforge.evaluation.service import EvaluationService

        config = EvaluationConfig()
        results = [
            EvaluationResult(query_log_id=uuid4(), evaluator_name="faithfulness", score=0.9),
            EvaluationResult(query_log_id=uuid4(), evaluator_name="faithfulness", score=0.5),
            EvaluationResult(query_log_id=uuid4(), evaluator_name="faithfulness", score=0.8),
        ]
        summary = EvaluationService._compute_summary(results, config)
        assert "faithfulness" in summary
        stats = summary["faithfulness"]
        assert stats["sample_count"] == 3
        assert stats["min"] == 0.5
        assert stats["max"] == 0.9
        # 0.5 is below 0.7 threshold
        assert stats["below_threshold"] == 1

    def test_compute_summary_empty(self) -> None:
        from vectorforge.evaluation.service import EvaluationService

        config = EvaluationConfig()
        summary = EvaluationService._compute_summary([], config)
        assert summary == {}

    def test_compute_summary_none_scores_excluded(self) -> None:
        from vectorforge.evaluation.service import EvaluationService

        config = EvaluationConfig()
        results = [
            EvaluationResult(query_log_id=uuid4(), evaluator_name="test", score=None),
            EvaluationResult(query_log_id=uuid4(), evaluator_name="test", score=0.8),
        ]
        summary = EvaluationService._compute_summary(results, config)
        assert summary["test"]["sample_count"] == 1


# ---------------------------------------------------------------------------
# BackgroundScheduler tests
# ---------------------------------------------------------------------------


class TestBackgroundScheduler:
    """Tests for BackgroundScheduler start/stop/trigger."""

    def test_disabled_config_does_not_start(self) -> None:
        factory = AsyncMock()
        scheduler = BackgroundScheduler(factory, EvaluationConfig(enabled=False))
        scheduler.start()
        assert not scheduler.is_running

    def test_start_creates_task(self) -> None:
        factory = AsyncMock()
        config = EvaluationConfig(enabled=True, schedule_interval_hours=1)
        scheduler = BackgroundScheduler(factory, config)

        loop = asyncio.new_event_loop()
        try:
            async def _test() -> None:
                scheduler.start()
                assert scheduler.is_running
                await scheduler.stop()
                assert not scheduler.is_running

            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_double_start_ignored(self) -> None:
        factory = AsyncMock()
        config = EvaluationConfig(enabled=True, schedule_interval_hours=1)
        scheduler = BackgroundScheduler(factory, config)

        loop = asyncio.new_event_loop()
        try:
            async def _test() -> None:
                scheduler.start()
                scheduler.start()  # second start is a no-op
                assert scheduler.is_running
                await scheduler.stop()

            loop.run_until_complete(_test())
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_trigger_now(self) -> None:
        fake_run = EvaluationRun(
            id=uuid4(), status=EvaluationRunStatus.COMPLETED, sample_size=10
        )
        mock_service = AsyncMock()
        mock_service.run_evaluation = AsyncMock(return_value=fake_run)

        factory = AsyncMock(return_value=mock_service)
        scheduler = BackgroundScheduler(factory, EvaluationConfig())

        result = await scheduler.trigger_now()
        assert result.status == EvaluationRunStatus.COMPLETED
        factory.assert_called_once()
        mock_service.run_evaluation.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self) -> None:
        scheduler = BackgroundScheduler(AsyncMock(), EvaluationConfig())
        await scheduler.stop()  # should not raise
        assert not scheduler.is_running

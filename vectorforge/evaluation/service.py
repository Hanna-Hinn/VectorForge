"""Evaluation orchestrator — runs evaluators, collects results, computes summaries.

Pipeline flow:
  1. Create an EvaluationRun record (status=RUNNING)
  2. Sample query logs according to the configured strategy
  3. Enrich samples with retrieved chunk data
  4. Dispatch each evaluator over the enriched samples
  5. Persist per-query results
  6. Compute summary scores
  7. Mark the run as COMPLETED (or FAILED)
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vectorforge.db.repositories.evaluation_repo import (
    EvaluationResultRepository,
    EvaluationRunRepository,
    RecommendationRepository,
)
from vectorforge.db.repositories.query_log_repo import QueryLogRepository
from vectorforge.evaluation.config import EvaluationConfig
from vectorforge.evaluation.registry import EvaluatorRegistry
from vectorforge.evaluation.types import (
    ChunkWithScore,
    CreateEvaluationResultDTO,
    CreateEvaluationRunDTO,
    EvaluationResult,
    EvaluationRun,
    EvaluationRunStatus,
    EvaluationSample,
)
from vectorforge.exceptions import EvaluationError
from vectorforge.models.db import ChunkModel, QueryLogModel

logger = logging.getLogger(__name__)


class EvaluationService:
    """Central orchestrator for running RAG quality evaluations.

    Samples queries, dispatches to evaluators, collects results, and
    computes aggregate scores.  The caller provides the async session
    and is responsible for committing / rolling-back the transaction.

    Args:
        session: An active async database session.
        registry: The evaluator registry with registered evaluators.
        config: Evaluation configuration.
    """

    def __init__(
        self,
        session: AsyncSession,
        registry: EvaluatorRegistry,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._session = session
        self._registry = registry
        self._config = config or EvaluationConfig()

        # Repositories scoped to the session
        self._run_repo = EvaluationRunRepository(session)
        self._result_repo = EvaluationResultRepository(session)
        self._rec_repo = RecommendationRepository(session)
        self._query_log_repo = QueryLogRepository(session)

    async def run_evaluation(
        self,
        config_override: EvaluationConfig | None = None,
    ) -> EvaluationRun:
        """Execute a full evaluation run.

        Args:
            config_override: Optional config to use instead of the default.

        Returns:
            The completed (or failed) EvaluationRun domain model.

        Raises:
            EvaluationError: If evaluation fails catastrophically.
        """
        config = config_override or self._config
        run = await self._run_repo.create(
            CreateEvaluationRunDTO(
                status=EvaluationRunStatus.RUNNING,
                sample_size=config.sample_size,
                config=config.model_dump(mode="json"),
            )
        )
        run = await self._run_repo.update_status(run.id, status="running")

        try:
            samples = await self._sample_queries(config)
            if not samples:
                logger.warning("No query logs available for evaluation")
                return await self._run_repo.update_status(
                    run.id,
                    status="completed",
                    summary_scores={"_note": "no_samples"},
                )

            enriched = await self._enrich_samples(samples)
            all_results = await self._execute_evaluators(run.id, enriched, config)
            summary = self._compute_summary(all_results, config)

            run = await self._run_repo.update_status(
                run.id,
                status="completed",
                summary_scores=summary,
            )
            logger.info(
                "Evaluation run %s completed: %d samples, %d results",
                run.id,
                len(enriched),
                len(all_results),
            )
            return run

        except Exception as exc:
            logger.error("Evaluation run %s failed: %s", run.id, exc, exc_info=True)
            run = await self._run_repo.update_status(
                run.id,
                status="failed",
                error_message=str(exc),
            )
            raise EvaluationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    async def _sample_queries(
        self, config: EvaluationConfig
    ) -> list[QueryLogModel]:
        """Fetch query log records according to the sampling strategy.

        Args:
            config: Evaluation config with strategy and sample size.

        Returns:
            List of QueryLogModel ORM instances.
        """
        limit = config.sample_size
        strategy = config.sample_strategy

        if strategy == "random":
            stmt = (
                select(QueryLogModel)
                .order_by(QueryLogModel.id)  # deterministic base before random
                .limit(limit * 3)  # over-fetch then sample
            )
            result = await self._session.execute(stmt)
            rows = list(result.scalars().all())
            import random
            return random.sample(rows, min(limit, len(rows))) if rows else []

        if strategy == "worst_performing":
            stmt = (
                select(QueryLogModel)
                .where(QueryLogModel.latency_ms.is_not(None))
                .order_by(QueryLogModel.latency_ms.desc())
                .limit(limit)
            )
        else:  # "recent" (default)
            stmt = (
                select(QueryLogModel)
                .order_by(QueryLogModel.created_at.desc())
                .limit(limit)
            )

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    async def _enrich_samples(
        self, query_logs: list[QueryLogModel]
    ) -> list[EvaluationSample]:
        """Build EvaluationSample objects from raw query log records.

        Fetches associated chunk text and similarity scores for each
        query log that has retrieved_chunk_ids.

        Args:
            query_logs: Raw query log ORM instances.

        Returns:
            List of enriched EvaluationSample objects.
        """
        samples: list[EvaluationSample] = []
        for ql in query_logs:
            chunks = await self._fetch_chunks(ql)
            samples.append(
                EvaluationSample(
                    query_log_id=ql.id,
                    query=ql.query_text,
                    chunks=chunks,
                    answer=ql.generated_response or "",
                )
            )
        return samples

    async def _fetch_chunks(
        self, query_log: QueryLogModel
    ) -> list[ChunkWithScore]:
        """Fetch chunk text for the retrieved_chunk_ids stored in a query log.

        Args:
            query_log: A query log ORM instance.

        Returns:
            List of ChunkWithScore objects.
        """
        chunk_data = query_log.retrieved_chunk_ids
        if not chunk_data:
            return []

        # retrieved_chunk_ids stores a list of {"chunk_id": ..., "score": ...}
        chunk_entries: list[dict[str, Any]] = []
        if isinstance(chunk_data, list):
            chunk_entries = chunk_data
        elif isinstance(chunk_data, dict) and "chunks" in chunk_data:
            chunk_entries = chunk_data["chunks"]
        else:
            return []

        if not chunk_entries:
            return []

        chunk_ids = []
        score_map: dict[str, float] = {}
        for entry in chunk_entries:
            cid = str(entry.get("chunk_id", ""))
            if cid:
                chunk_ids.append(cid)
                score_map[cid] = float(entry.get("score", 0.0))

        if not chunk_ids:
            return []

        import uuid as _uuid

        parsed_ids = []
        for cid in chunk_ids:
            try:
                parsed_ids.append(_uuid.UUID(cid))
            except ValueError:
                continue

        if not parsed_ids:
            return []

        stmt = (
            select(ChunkModel)
            .where(ChunkModel.id.in_(parsed_ids))
        )
        result = await self._session.execute(stmt)
        chunk_models = result.scalars().all()

        chunks: list[ChunkWithScore] = []
        for cm in chunk_models:
            chunks.append(
                ChunkWithScore(
                    chunk_id=cm.id,
                    text=cm.content,
                    score=score_map.get(str(cm.id), 0.0),
                    document_source=cm.chunk_metadata.get("source", "")
                    if cm.chunk_metadata
                    else "",
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Evaluator Dispatch
    # ------------------------------------------------------------------

    async def _execute_evaluators(
        self,
        run_id: Any,
        samples: list[EvaluationSample],
        config: EvaluationConfig,
    ) -> list[EvaluationResult]:
        """Run all registered evaluators over the sample set.

        Evaluators run with bounded concurrency via an asyncio semaphore.

        Args:
            run_id: The evaluation run UUID.
            samples: Enriched evaluation samples.
            config: Configuration with concurrency limits.

        Returns:
            Flat list of all evaluation results.
        """
        evaluator_names = self._registry.list_available()
        if not evaluator_names:
            logger.warning("No evaluators registered — skipping evaluation dispatch")
            return []

        semaphore = asyncio.Semaphore(config.max_concurrent_evaluators)
        all_results: list[EvaluationResult] = []

        async def _run_one(name: str) -> list[EvaluationResult]:
            async with semaphore:
                return await self._run_evaluator(run_id, name, samples)

        tasks = [_run_one(name) for name in evaluator_names]
        results_per_evaluator = await asyncio.gather(*tasks, return_exceptions=True)

        for name, batch in zip(evaluator_names, results_per_evaluator, strict=True):
            if isinstance(batch, BaseException):
                logger.error("Evaluator %s raised an exception: %s", name, batch)
                continue
            all_results.extend(batch)

        return all_results

    async def _run_evaluator(
        self,
        run_id: Any,
        evaluator_name: str,
        samples: list[EvaluationSample],
    ) -> list[EvaluationResult]:
        """Run a single evaluator over all samples and persist results.

        Args:
            run_id: The evaluation run UUID.
            evaluator_name: Name of the evaluator to run.
            samples: Enriched evaluation samples.

        Returns:
            List of evaluation results from this evaluator.
        """
        evaluator = self._registry.get(evaluator_name)
        results: list[EvaluationResult] = []

        try:
            results = await evaluator.evaluate_batch(samples)
        except Exception as exc:
            logger.error(
                "Evaluator %s failed on batch: %s", evaluator_name, exc, exc_info=True
            )
            # Create a failed placeholder result for each sample
            for sample in samples:
                results.append(
                    EvaluationResult(
                        query_log_id=sample.query_log_id,
                        evaluator_name=evaluator_name,
                        score=None,
                        details={"error": str(exc)},
                        reasoning=None,
                    )
                )

        # Persist results
        dtos = [
            CreateEvaluationResultDTO(
                run_id=run_id,
                query_log_id=r.query_log_id,
                evaluator_name=r.evaluator_name,
                score=r.score,
                details=r.details,
                reasoning=r.reasoning,
            )
            for r in results
        ]
        if dtos:
            await self._result_repo.create_batch(dtos)

        return results

    # ------------------------------------------------------------------
    # Summary Computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_summary(
        results: list[EvaluationResult],
        config: EvaluationConfig,
    ) -> dict[str, Any]:
        """Compute aggregate scores per evaluator.

        Args:
            results: All evaluation results from the run.
            config: Config with threshold values.

        Returns:
            Dict mapping evaluator name → summary statistics.
        """
        threshold_map: dict[str, float] = {
            "faithfulness": config.faithfulness_threshold,
            "retrieval_relevance": config.relevance_threshold,
            "hallucination": config.hallucination_threshold,
            "chunk_coverage": config.coverage_threshold,
            "answer_relevance": config.relevance_threshold,
            "embedding_drift": config.relevance_threshold,
        }

        by_evaluator: dict[str, list[float]] = {}
        for r in results:
            if r.score is not None:
                by_evaluator.setdefault(r.evaluator_name, []).append(r.score)

        summary: dict[str, Any] = {}
        for name, scores in by_evaluator.items():
            threshold = threshold_map.get(name, 0.5)
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            p50_idx = n // 2
            summary[name] = {
                "avg": round(statistics.mean(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "p50": round(sorted_scores[p50_idx], 4),
                "below_threshold": sum(1 for s in scores if s < threshold),
                "sample_count": n,
            }

        return summary

"""CLI commands for running and viewing RAG quality evaluations."""

from __future__ import annotations

import uuid

import typer

from vectorforge.cli._helpers import run_async

evaluate_app = typer.Typer(no_args_is_help=True)


@evaluate_app.command("run")
def run_evaluation(
    sample_size: int | None = typer.Option(
        None, "--sample-size", "-n", help="Number of queries to evaluate.",
    ),
    strategy: str | None = typer.Option(
        None, "--strategy", "-s",
        help="Sampling strategy: recent, random, worst_performing.",
    ),
) -> None:
    """Trigger an evaluation run against recent queries."""
    run_async(_run_evaluation(sample_size=sample_size, strategy=strategy))


@evaluate_app.command("report")
def report(
    run_id: str | None = typer.Option(
        None, "--run-id", "-r", help="Evaluation run UUID (default: latest).",
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table or json.",
    ),
) -> None:
    """Display the evaluation report for a run."""
    parsed_id: uuid.UUID | None = None
    if run_id:
        try:
            parsed_id = uuid.UUID(run_id)
        except ValueError:
            typer.echo(f"Invalid UUID: {run_id}", err=True)
            raise typer.Exit(code=1) from None
    run_async(_show_report(run_id=parsed_id, output_format=output_format))


@evaluate_app.command("history")
def history(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent runs to show."),
) -> None:
    """List recent evaluation runs."""
    run_async(_show_history(limit=limit))


@evaluate_app.command("recommendations")
def recommendations(
    status: str | None = typer.Option(
        None, "--status",
        help="Filter: pending, acknowledged, resolved, dismissed.",
    ),
    category: str | None = typer.Option(
        None, "--category",
        help="Filter: retrieval, generation, chunking, embedding.",
    ),
) -> None:
    """List evaluation recommendations."""
    run_async(_show_recommendations(status=status, category=category))


# ---------------------------------------------------------------------------
# Async implementations
# ---------------------------------------------------------------------------


async def _run_evaluation(
    *,
    sample_size: int | None,
    strategy: str | None,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vectorforge.cli._helpers import managed_engine
    from vectorforge.evaluation.config import EvaluationConfig
    from vectorforge.evaluation.recommendation import RecommendationEngine
    from vectorforge.evaluation.registry import EvaluatorRegistry
    from vectorforge.evaluation.service import EvaluationService

    async with managed_engine() as db:
        session_factory = async_sessionmaker(
            db.engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with session_factory() as session:
            config = EvaluationConfig()
            if sample_size is not None:
                config = EvaluationConfig(sample_size=sample_size)
            if strategy is not None:
                config = EvaluationConfig(
                    sample_size=sample_size or config.sample_size,
                    sample_strategy=strategy,
                )

            registry = EvaluatorRegistry()
            registry.register_defaults()

            service = EvaluationService(session, registry, config)

            typer.echo("Starting evaluation run...")
            run = await service.run_evaluation()
            await session.commit()

            typer.echo(f"\nRun ID: {run.id}")
            typer.echo(f"Status: {run.status}")
            typer.echo(f"Samples: {run.sample_size}")
            typer.echo("")

            if run.summary_scores and not run.summary_scores.get("_note"):
                _print_summary_table(run.summary_scores, config)

                # Generate recommendations
                from vectorforge.db.repositories.evaluation_repo import (
                    EvaluationResultRepository,
                    RecommendationRepository,
                )

                result_repo = EvaluationResultRepository(session)
                rec_repo = RecommendationRepository(session)
                results = await result_repo.find_by_run(run.id)
                engine = RecommendationEngine(config)

                from vectorforge.evaluation.types import EvaluationResult as EvalResult

                eval_results = [
                    EvalResult(
                        query_log_id=r.query_log_id,
                        evaluator_name=r.evaluator_name,
                        score=r.score,
                        details=r.details,
                        reasoning=r.reasoning,
                    )
                    for r in results
                ]
                rec_dtos = engine.analyze(run.id, run.summary_scores, eval_results)
                for dto in rec_dtos:
                    await rec_repo.create(dto)
                await session.commit()

                if rec_dtos:
                    typer.echo(
                        f"\n⚠ {len(rec_dtos)} recommendations generated. "
                        "Run 'vectorforge evaluate report' for details."
                    )
            else:
                typer.echo("No query logs available for evaluation.")


async def _show_report(
    *,
    run_id: uuid.UUID | None,
    output_format: str,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vectorforge.cli._helpers import managed_engine
    from vectorforge.db.repositories.evaluation_repo import (
        EvaluationResultRepository,
        EvaluationRunRepository,
        RecommendationRepository,
    )
    from vectorforge.evaluation.config import EvaluationConfig
    from vectorforge.evaluation.report import EvaluationReportBuilder
    from vectorforge.evaluation.types import (
        EvaluationResult,
        EvaluationRun,
        Recommendation,
    )

    async with managed_engine() as db:
        session_factory = async_sessionmaker(
            db.engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with session_factory() as session:
            run_repo = EvaluationRunRepository(session)
            result_repo = EvaluationResultRepository(session)
            rec_repo = RecommendationRepository(session)

            if run_id:
                runs = await run_repo.find_recent(limit=1)
                target = None
                for r in runs:
                    if r.id == run_id:
                        target = r
                        break
                if target is None:
                    typer.echo(f"Run {run_id} not found.", err=True)
                    raise typer.Exit(code=1)
            else:
                runs = await run_repo.find_recent(limit=1)
                if not runs:
                    typer.echo("No evaluation runs found.", err=True)
                    raise typer.Exit(code=1)
                target = runs[0]

            run_domain = EvaluationRun.model_validate(target)
            result_models = await result_repo.find_by_run(target.id)
            rec_models = await rec_repo.find_by_run(target.id)

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
            recs = [Recommendation.model_validate(r) for r in rec_models]

            config = EvaluationConfig()
            thresholds = {
                "retrieval_relevance": config.relevance_threshold,
                "chunk_coverage": config.coverage_threshold,
                "faithfulness": config.faithfulness_threshold,
                "answer_relevance": config.relevance_threshold,
                "hallucination": 1.0 - config.hallucination_threshold,
                "embedding_drift": 0.5,
            }

            previous = await run_repo.find_recent(limit=6)
            prev_runs = [
                EvaluationRun.model_validate(r)
                for r in previous
                if r.id != target.id
            ][:5]

            builder = EvaluationReportBuilder()
            report = builder.build(
                run_domain,
                eval_results,
                recs,
                previous_runs=prev_runs,
                thresholds=thresholds,
            )

            if output_format == "json":
                typer.echo(report.model_dump_json(indent=2))
            else:
                _print_report(report)


async def _show_history(*, limit: int) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vectorforge.cli._helpers import managed_engine
    from vectorforge.db.repositories.evaluation_repo import EvaluationRunRepository

    async with managed_engine() as db:
        session_factory = async_sessionmaker(
            db.engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with session_factory() as session:
            repo = EvaluationRunRepository(session)
            runs = await repo.find_recent(limit=limit)

            if not runs:
                typer.echo("No evaluation runs found.")
                return

            typer.echo(f"{'Run ID':<40} {'Date':<22} {'Status':<12} {'Samples':<8}")
            typer.echo("-" * 82)
            for run in runs:
                date_str = (
                    run.created_at.strftime("%Y-%m-%d %H:%M")
                    if run.created_at
                    else "N/A"
                )
                typer.echo(
                    f"{run.id!s:<40} {date_str:<22} {run.status:<12} {run.sample_size:<8}"
                )


async def _show_recommendations(
    *,
    status: str | None,
    category: str | None,
) -> None:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vectorforge.cli._helpers import managed_engine
    from vectorforge.models.db import RecommendationModel

    async with managed_engine() as db:
        session_factory = async_sessionmaker(
            db.engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with session_factory() as session:
            stmt = select(RecommendationModel).order_by(
                RecommendationModel.created_at.desc()
            )
            if status:
                stmt = stmt.where(RecommendationModel.status == status)
            if category:
                stmt = stmt.where(RecommendationModel.category == category)
            stmt = stmt.limit(50)

            result = await session.execute(stmt)
            recs = list(result.scalars().all())

            if not recs:
                typer.echo("No recommendations found.")
                return

            typer.echo(f"{'#':<4} {'Severity':<10} {'Category':<12} {'Title':<50} {'Status':<12}")
            typer.echo("-" * 88)
            for i, rec in enumerate(recs, 1):
                typer.echo(
                    f"{i:<4} {rec.severity:<10} {rec.category:<12} "
                    f"{rec.title[:48]:<50} {rec.status:<12}"
                )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_summary_table(
    summary_scores: dict,
    config: object,
) -> None:
    """Print a summary table of evaluator scores."""
    from vectorforge.evaluation.config import EvaluationConfig

    cfg = config if isinstance(config, EvaluationConfig) else EvaluationConfig()
    thresholds = {
        "retrieval_relevance": cfg.relevance_threshold,
        "chunk_coverage": cfg.coverage_threshold,
        "faithfulness": cfg.faithfulness_threshold,
        "answer_relevance": cfg.relevance_threshold,
        "hallucination": 1.0 - cfg.hallucination_threshold,
        "embedding_drift": 0.5,
    }

    typer.echo(f"{'Evaluator':<25} {'Avg Score':<12} {'Status':<8}")
    typer.echo("-" * 45)
    for name, scores in summary_scores.items():
        if name.startswith("_") or not isinstance(scores, dict):
            continue
        avg = float(scores.get("avg", 0.0))
        threshold = thresholds.get(name, 0.5)
        status_str = "✅ PASS" if avg >= threshold else "❌ FAIL"
        typer.echo(f"{name:<25} {avg:<12.2f} {status_str}")


def _print_report(report: object) -> None:
    """Print a formatted evaluation report."""
    from vectorforge.evaluation.types import EvaluationReport

    if not isinstance(report, EvaluationReport):
        return

    typer.echo("=" * 60)
    typer.echo("EVALUATION REPORT")
    typer.echo("=" * 60)
    typer.echo(f"Run ID: {report.header.get('run_id', 'N/A')}")
    typer.echo(f"Timestamp: {report.header.get('timestamp', 'N/A')}")
    typer.echo(f"Samples: {report.header.get('sample_size', 0)}")
    duration = report.header.get("duration_seconds")
    if duration is not None:
        typer.echo(f"Duration: {duration:.1f}s")
    typer.echo("")

    # Score summary
    typer.echo("Score Summary:")
    typer.echo(f"{'Evaluator':<25} {'Avg':<8} {'Min':<8} {'Max':<8} {'Status':<8}")
    typer.echo("-" * 57)
    for row in report.score_summary:
        typer.echo(
            f"{row.evaluator:<25} {row.avg:<8.2f} {row.min_score:<8.2f} "
            f"{row.max_score:<8.2f} {row.status:<8}"
        )
    typer.echo("")

    # Trends
    if report.trends:
        typer.echo("Trends:")
        arrows = {"improving": "↑", "stable": "→", "degrading": "↓"}
        for t in report.trends:
            arrow = arrows.get(t.direction, "?")
            typer.echo(f"  {t.evaluator}: {arrow} {t.direction} ({t.change_pct:+.1f}%)")
        typer.echo("")

    # Recommendations
    if report.recommendations:
        typer.echo(f"Recommendations ({len(report.recommendations)}):")
        for rec in report.recommendations:
            icon = {"critical": "🔴", "high": "🟡", "medium": "🟢", "low": "⚪"}.get(
                rec.severity, "•"
            )
            typer.echo(f"  {icon} [{rec.severity}] {rec.title}")
            typer.echo(f"    {rec.description[:100]}...")
        typer.echo("")

    # Worst queries
    if report.worst_queries:
        typer.echo(f"Worst Queries (top {len(report.worst_queries)}):")
        for wq in report.worst_queries[:5]:
            typer.echo(f"  Score: {wq.composite_score:.2f} - {', '.join(wq.key_issues[:3])}")

"""CLI commands for executing RAG queries."""

from __future__ import annotations

import uuid

import typer

from vectorforge.cli._helpers import run_async

query_app = typer.Typer(no_args_is_help=True)


@query_app.command("run")
def run_query(
    collection_id: str = typer.Argument(help="Collection UUID."),
    question: str = typer.Argument(help="The query / question to ask."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of chunks to retrieve."),
    min_score: float = typer.Option(0.0, "--min-score", help="Minimum similarity score."),
    llm_provider: str | None = typer.Option(None, "--llm", help="Override LLM provider."),
    llm_model: str | None = typer.Option(None, "--model", help="Override LLM model."),
    show_sources: bool = typer.Option(False, "--sources", "-s", help="Show source chunks."),
) -> None:
    """Execute a RAG query against a collection."""
    try:
        cid = uuid.UUID(collection_id)
    except ValueError:
        typer.echo(f"Invalid UUID: {collection_id}", err=True)
        raise typer.Exit(code=1) from None

    run_async(
        _run_query(
            cid, question,
            top_k=top_k,
            min_score=min_score,
            llm_provider=llm_provider,
            llm_model=llm_model,
            show_sources=show_sources,
        )
    )


async def _run_query(
    collection_id: uuid.UUID,
    question: str,
    *,
    top_k: int,
    min_score: float,
    llm_provider: str | None,
    llm_model: str | None,
    show_sources: bool,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vectorforge.cli._helpers import managed_engine
    from vectorforge.db.repositories.collection_repo import CollectionRepository
    from vectorforge.db.repositories.query_log_repo import QueryLogRepository
    from vectorforge.embedding.registry import EmbeddingProviderRegistry
    from vectorforge.llm.registry import LLMProviderRegistry
    from vectorforge.pipeline.context import ContextBuilder
    from vectorforge.pipeline.rag import QueryService
    from vectorforge.pipeline.types import QueryConfig
    from vectorforge.retriever.dense import DenseRetriever
    from vectorforge.vectorstore.pgvector import PgVectorStore

    async with managed_engine() as db:
        session_factory = async_sessionmaker(
            db.engine, class_=AsyncSession, expire_on_commit=False,
        )

        async with db.get_session() as session:
            collection_repo = CollectionRepository(session)

            collection = await collection_repo.find_by_id(collection_id)
            if collection is None:
                typer.echo(f"Collection {collection_id} not found.", err=True)
                raise typer.Exit(code=1)

            embedding_registry = EmbeddingProviderRegistry()
            embedding_registry.auto_discover()

            vector_store = PgVectorStore(session_factory)
            retriever = DenseRetriever(
                embedding_registry=embedding_registry,
                vector_store=vector_store,
                collection_repo=collection_repo,
            )
            context_builder = ContextBuilder()

            llm_registry = LLMProviderRegistry()
            llm_registry.auto_discover()

            query_log_repo = QueryLogRepository(session)

            service = QueryService(
                retriever=retriever,
                context_builder=context_builder,
                llm_registry=llm_registry,
                query_log_repo=query_log_repo,
            )

            query_config = QueryConfig(
                top_k=top_k,
                min_score=min_score,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )

            result = await service.query(question, collection_id, query_config)

        typer.echo(f"\nAnswer:\n{result.answer}\n")
        typer.echo(
            f"Latency: retrieval={result.retrieval_latency_ms:.0f}ms, "
            f"generation={result.generation_latency_ms:.0f}ms, "
            f"total={result.total_latency_ms:.0f}ms"
        )

        if show_sources and result.sources:
            typer.echo("\nSources:")
            for src in result.sources:
                typer.echo(f"  - {src.document_source} (chunk {src.chunk_index})")

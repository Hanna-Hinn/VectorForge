"""RAG query service — the main pipeline orchestrator.

Ties retrieval, context building, and LLM generation into a
single ``query()`` entry point.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator

from vectorforge.db.repositories.query_log_repo import QueryLogRepository
from vectorforge.llm.registry import BaseLLMProviderLookup
from vectorforge.llm.types import LLMRequestConfig
from vectorforge.models.domain import CreateQueryLogDTO
from vectorforge.monitoring.metrics import get_metrics_collector
from vectorforge.pipeline.context import BaseContextBuilder, ContextConfig
from vectorforge.pipeline.types import QueryConfig, QueryResult
from vectorforge.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)

_NO_RESULTS_ANSWER = "No relevant documents found for your query."


class QueryService:
    """Orchestrates the full RAG pipeline.

    Combines retrieval → context assembly → LLM generation into
    a single query interface.

    Args:
        retriever: The retrieval strategy to use.
        context_builder: Builder for assembling LLM prompts.
        llm_registry: Lookup interface for LLM providers.
        query_log_repo: Repository for persisting query logs (optional).
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        context_builder: BaseContextBuilder,
        llm_registry: BaseLLMProviderLookup,
        query_log_repo: QueryLogRepository | None = None,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._llm_registry = llm_registry
        self._query_log_repo = query_log_repo
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def query(
        self,
        query: str,
        collection_id: uuid.UUID,
        config: QueryConfig | None = None,
    ) -> QueryResult:
        """Execute a full RAG query.

        Steps:
          1. Retrieve relevant chunks.
          2. Build context payload.
          3. Generate an answer via LLM.
          4. Log the query (non-blocking).

        Args:
            query: The user question.
            collection_id: The collection to search.
            config: Optional query configuration.

        Returns:
            A QueryResult with the answer, chunks, and latencies.
        """
        cfg = config or QueryConfig()
        metrics = get_metrics_collector()
        total_start = time.perf_counter()

        # --- Step 1: Retrieve ---
        retrieval_start = time.perf_counter()
        retrieved_chunks = await self._retriever.retrieve(
            query=query,
            collection_id=collection_id,
            top_k=cfg.top_k,
            filters=cfg.filters,
            min_score=cfg.min_score,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        metrics.observe(
            "retriever.search.duration_ms", retrieval_ms,
            tags={"retriever_type": "dense"},
        )

        # --- Handle empty results ---
        if not retrieved_chunks:
            total_ms = (time.perf_counter() - total_start) * 1000
            return QueryResult(
                query=query,
                answer=_NO_RESULTS_ANSWER,
                retrieved_chunks=[],
                sources=[],
                retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=0.0,
                total_latency_ms=total_ms,
                llm_response=None,
            )

        # --- Step 2: Build context ---
        context_start = time.perf_counter()
        context_config = ContextConfig(
            max_context_tokens=cfg.max_context_tokens,
            include_sources=cfg.include_sources,
        )
        context_payload = self._context_builder.build(
            query=query,
            chunks=retrieved_chunks,
            config=context_config,
        )
        context_ms = (time.perf_counter() - context_start) * 1000
        metrics.observe("context.build.duration_ms", context_ms)
        metrics.observe(
            "context.tokens_used", float(context_payload.context_token_count),
        )
        metrics.observe(
            "context.chunks_included", float(len(context_payload.sources)),
        )

        # --- Step 3: Generate answer ---
        generation_start = time.perf_counter()

        llm_provider = (
            self._llm_registry.get(cfg.llm_provider)
            if cfg.llm_provider
            else self._llm_registry.get_default()
        )

        messages = [
            {"role": "system", "content": context_payload.system_prompt},
            {"role": "user", "content": context_payload.user_message},
        ]

        llm_config = LLMRequestConfig(
            model=cfg.llm_model or llm_provider.default_model(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        llm_response = await llm_provider.generate(messages, llm_config)
        generation_ms = (time.perf_counter() - generation_start) * 1000

        provider_name = llm_provider.provider_name()
        model_name = llm_config.model
        metrics.observe(
            "llm.generate.duration_ms", generation_ms,
            tags={"provider": provider_name, "model": model_name},
        )
        metrics.increment(
            "llm.tokens_input", float(llm_response.prompt_tokens),
            tags={"provider": provider_name, "model": model_name},
        )
        metrics.increment(
            "llm.tokens_output", float(llm_response.completion_tokens),
            tags={"provider": provider_name, "model": model_name},
        )

        # --- Step 4: Build result ---
        total_ms = (time.perf_counter() - total_start) * 1000
        metrics.observe("pipeline.query.duration_ms", total_ms)
        metrics.increment("pipeline.query.calls")

        result = QueryResult(
            query=query,
            answer=llm_response.content,
            retrieved_chunks=retrieved_chunks,
            sources=context_payload.sources,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            total_latency_ms=total_ms,
            llm_response=llm_response,
        )

        # --- Step 5: Log query (non-blocking) ---
        if self._query_log_repo is not None:
            task = asyncio.create_task(
                self._log_query(query, collection_id, result),
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return result

    async def query_stream(
        self,
        query: str,
        collection_id: uuid.UUID,
        config: QueryConfig | None = None,
    ) -> AsyncIterator[str]:
        """Execute a streaming RAG query.

        Retrieves chunks and streams the LLM response token-by-token.

        Args:
            query: The user question.
            collection_id: The collection to search.
            config: Optional query configuration.

        Yields:
            Token strings as they arrive from the LLM.
        """
        cfg = config or QueryConfig()

        retrieved_chunks = await self._retriever.retrieve(
            query=query,
            collection_id=collection_id,
            top_k=cfg.top_k,
            filters=cfg.filters,
            min_score=cfg.min_score,
        )

        if not retrieved_chunks:
            yield _NO_RESULTS_ANSWER
            return

        context_config = ContextConfig(
            max_context_tokens=cfg.max_context_tokens,
            include_sources=cfg.include_sources,
        )
        context_payload = self._context_builder.build(
            query=query,
            chunks=retrieved_chunks,
            config=context_config,
        )

        llm_provider = (
            self._llm_registry.get(cfg.llm_provider)
            if cfg.llm_provider
            else self._llm_registry.get_default()
        )

        messages = [
            {"role": "system", "content": context_payload.system_prompt},
            {"role": "user", "content": context_payload.user_message},
        ]

        llm_config = LLMRequestConfig(
            model=cfg.llm_model or llm_provider.default_model(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        async for token in llm_provider.generate_stream(messages, llm_config):
            yield token

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _log_query(
        self,
        query: str,
        collection_id: uuid.UUID,
        result: QueryResult,
    ) -> None:
        """Persist a query log record. Failures are swallowed with a warning.

        Args:
            query: The original query text.
            collection_id: The target collection.
            result: The query result.
        """
        try:
            chunk_ids = [str(c.chunk.id) for c in result.retrieved_chunks]
            dto = CreateQueryLogDTO(
                collection_id=collection_id,
                query_text=query,
                retrieved_chunk_ids={"chunk_ids": chunk_ids},
                generated_response=result.answer,
                latency_ms=result.total_latency_ms,
            )
            await self._query_log_repo.create(dto)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Failed to log query: %s", exc)

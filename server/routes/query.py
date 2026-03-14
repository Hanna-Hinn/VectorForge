"""Query endpoints — synchronous and SSE streaming."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from server.dependencies import ApiKey, QueryServiceDep
from server.schemas import QueryRequest, QueryResponse, SourceCitationResponse
from vectorforge.pipeline.types import QueryConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


def _to_query_config(body: QueryRequest) -> QueryConfig:
    """Convert an API QueryRequest to the pipeline QueryConfig."""
    return QueryConfig(
        top_k=body.top_k,
        min_score=body.min_score,
        filters=body.filters,
        llm_provider=body.llm_provider,
        llm_model=body.llm_model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        include_sources=body.include_sources,
        max_context_tokens=body.max_context_tokens,
    )


@router.post("", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    service: QueryServiceDep,
    _key: ApiKey,
) -> QueryResponse:
    """Execute a synchronous RAG query."""
    config = _to_query_config(body)

    result = await service.query(
        query=body.query,
        collection_id=body.collection_id,
        config=config,
    )

    sources = [
        SourceCitationResponse(
            document_source=s.document_source,
            chunk_index=s.chunk_index,
            score=s.score,
            snippet=s.snippet,
        )
        for s in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_latency_ms=result.total_latency_ms,
    )


@router.post("/stream")
async def query_stream(
    body: QueryRequest,
    service: QueryServiceDep,
    _key: ApiKey,
) -> EventSourceResponse:
    """Execute a streaming RAG query via Server-Sent Events.

    Event types:
      - ``metadata``: Contains source citations (sent first).
      - ``token``: A single token from the LLM stream.
      - ``done``: Signals completion.
      - ``error``: Signals an error.
    """
    config = _to_query_config(body)

    async def event_generator() -> AsyncIterator[str]:
        try:
            # Stream tokens from the query service
            async for token in service.query_stream(
                query=body.query,
                collection_id=body.collection_id,
                config=config,
            ):
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"

            # Completion signal
            done_payload = json.dumps({"type": "done"})
            yield f"data: {done_payload}\n\n"

        except Exception as exc:
            logger.error("Streaming query error: %s", exc, exc_info=True)
            error_payload = json.dumps(
                {"type": "error", "message": "An internal error occurred."}
            )
            yield f"data: {error_payload}\n\n"

    return EventSourceResponse(event_generator(), media_type="text/event-stream")

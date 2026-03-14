"""Pipeline request/response types for the RAG query service."""

from __future__ import annotations

from pydantic import BaseModel, Field

from vectorforge.llm.types import LLMResponse
from vectorforge.models.domain import RetrievedChunk
from vectorforge.pipeline.context import SourceCitation


class QueryConfig(BaseModel):
    """Configuration for a single RAG query.

    Args:
        top_k: Maximum chunks to retrieve.
        min_score: Minimum similarity score threshold.
        filters: Optional metadata filters for retrieval.
        llm_provider: Override LLM provider name.
        llm_model: Override LLM model name.
        embedding_provider: Override embedding provider name.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens for LLM response.
        include_sources: Whether to include source citations.
        max_context_tokens: Token budget for context assembly.
    """

    top_k: int = 10
    min_score: float = 0.0
    filters: dict[str, object] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    embedding_provider: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    include_sources: bool = True
    max_context_tokens: int = 4096


class QueryResult(BaseModel):
    """The result of a complete RAG pipeline query."""

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    llm_response: LLMResponse | None = None

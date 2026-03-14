"""Pydantic request/response schemas for the REST API.

These models are used exclusively at the API boundary — they are
never passed into the core package.  Route handlers convert between
these schemas and the domain DTOs/models defined in ``vectorforge.models``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    message: str
    details: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class CreateCollectionRequest(BaseModel):
    """Body for ``POST /api/collections``."""

    name: str
    description: str = ""
    metric: str = "cosine"
    embedding_provider: str | None = None
    embedding_model: str | None = None
    chunking_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class CollectionResponse(BaseModel):
    """Single collection in API responses."""

    id: UUID
    name: str
    description: str
    embedding_config: dict[str, object] | None = None
    chunking_config: dict[str, object] | None = None
    created_at: datetime
    updated_at: datetime | None = None


class CollectionListResponse(BaseModel):
    """Response for ``GET /api/collections``."""

    collections: list[CollectionResponse]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class IngestDocumentRequest(BaseModel):
    """Body for ``POST /api/collections/{id}/documents``."""

    source: str
    metadata: dict[str, object] = Field(default_factory=dict)
    chunking_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class DocumentResponse(BaseModel):
    """Single document in API responses."""

    id: UUID
    collection_id: UUID
    source_uri: str
    content_type: str
    status: str
    content_size_bytes: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime | None = None


class DocumentListResponse(BaseModel):
    """Response for ``GET /api/collections/{id}/documents``."""

    documents: list[DocumentResponse]
    total: int


class DocumentDetailResponse(DocumentResponse):
    """Detailed document response (includes raw content)."""

    raw_content: str | None = None


class BatchIngestResult(BaseModel):
    """Result of a single document in a batch ingest."""

    source: str
    document_id: UUID | None = None
    error: str | None = None


class BatchIngestResponse(BaseModel):
    """Response for ``POST /api/collections/{id}/documents/batch``."""

    results: list[BatchIngestResult]
    succeeded: int
    failed: int


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body for ``POST /api/query`` and ``POST /api/query/stream``."""

    query: str
    collection_id: UUID
    top_k: int = 10
    min_score: float = 0.0
    filters: dict[str, object] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    include_sources: bool = True
    max_context_tokens: int = 4096


class SourceCitationResponse(BaseModel):
    """A source citation in a query response."""

    document_source: str
    chunk_index: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    """Response for ``POST /api/query``."""

    answer: str
    sources: list[SourceCitationResponse] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


class LatencyStatsResponse(BaseModel):
    """Latency statistics for analytics endpoints."""

    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    sample_count: int


class QueryFrequencyResponse(BaseModel):
    """A query and its frequency count."""

    query_text: str
    count: int


class VolumeDataPointResponse(BaseModel):
    """Daily query volume point."""

    date: datetime
    count: int


class AnalyticsSummaryResponse(BaseModel):
    """Full analytics summary for a collection."""

    total_queries: int
    unique_queries: int
    latency: LatencyStatsResponse | None = None
    top_queries: list[QueryFrequencyResponse] = Field(default_factory=list)
    volume: list[VolumeDataPointResponse] = Field(default_factory=list)


class TopQueriesResponse(BaseModel):
    """Response for top queries endpoint."""

    queries: list[QueryFrequencyResponse]


# ---------------------------------------------------------------------------
# Status / Health
# ---------------------------------------------------------------------------


class ComponentHealthResponse(BaseModel):
    """Health of a single system component."""

    name: str
    status: str
    latency_ms: float | None = None
    message: str | None = None


class SystemHealthResponse(BaseModel):
    """Aggregated system health response."""

    status: str
    components: list[ComponentHealthResponse]
    checked_at: datetime


class ProviderInfo(BaseModel):
    """Information about a registered provider."""

    name: str
    type: str


class ProvidersResponse(BaseModel):
    """Response for ``GET /api/status/providers``."""

    embedding_providers: list[str]
    llm_providers: list[str]


class MessageResponse(BaseModel):
    """Generic success message."""

    message: str

"""Pydantic domain models, DTOs, and enums for VectorForge.

Domain models are used at the service and API boundary.
They are never passed directly to the database layer.
"""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentStatus(enum.StrEnum):
    """Processing status of a document in the pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ERROR = "error"
    DELETED = "deleted"


class DistanceMetric(enum.StrEnum):
    """Vector similarity distance metric."""

    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


# ---------------------------------------------------------------------------
# Domain Models (read representations)
# ---------------------------------------------------------------------------


class Collection(BaseModel):
    """A named grouping of documents with shared embedding/chunking config."""

    id: UUID
    name: str
    description: str = ""
    embedding_config: dict[str, object] | None = None
    chunking_config: dict[str, object] | None = None
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class Document(BaseModel):
    """A document ingested into a collection."""

    id: UUID
    collection_id: UUID
    source_uri: str
    content_type: str
    raw_content: str | None = None
    storage_backend: str = "pg"
    s3_key: str | None = None
    content_size_bytes: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class Chunk(BaseModel):
    """A segment of a document, ready for embedding."""

    id: UUID
    document_id: UUID
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}


class Embedding(BaseModel):
    """A vector embedding associated with a chunk."""

    id: UUID
    chunk_id: UUID
    model_name: str
    dimensions: int
    vector: list[float]
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryLog(BaseModel):
    """A record of a RAG query for analytics."""

    id: UUID
    collection_id: UUID
    query_text: str
    retrieved_chunk_ids: dict[str, object] | None = None
    generated_response: str | None = None
    latency_ms: float | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Query / Result Models
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A chunk returned from a similarity search with its score."""

    chunk: Chunk
    score: float
    document_source: str


class QueryResult(BaseModel):
    """The result of a RAG query."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    generated_answer: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# DTOs (Create / Update)
# ---------------------------------------------------------------------------


class CreateCollectionDTO(BaseModel):
    """Data required to create a new collection."""

    name: str
    description: str = ""
    metric: DistanceMetric = DistanceMetric.COSINE
    embedding_config: dict[str, object] | None = None
    chunking_config: dict[str, object] | None = None


class CreateDocumentDTO(BaseModel):
    """Data required to ingest a new document."""

    collection_id: UUID
    source_uri: str
    content_type: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)


class CreateChunkDTO(BaseModel):
    """Data required to create a chunk from a document."""

    document_id: UUID
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, object] = Field(default_factory=dict)


class CreateEmbeddingDTO(BaseModel):
    """Data required to store an embedding for a chunk."""

    chunk_id: UUID
    model_name: str
    dimensions: int
    vector: list[float]


class UpdateCollectionDTO(BaseModel):
    """Fields that can be updated on a collection."""

    description: str | None = None
    embedding_config: dict[str, object] | None = None
    chunking_config: dict[str, object] | None = None


class CreateQueryLogDTO(BaseModel):
    """Data required to record a RAG query."""

    collection_id: UUID
    query_text: str
    retrieved_chunk_ids: dict[str, object] | None = None
    generated_response: str | None = None
    latency_ms: float | None = None

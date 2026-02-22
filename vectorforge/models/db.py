"""SQLAlchemy ORM models for VectorForge.

ORM models map directly to PostgreSQL tables. They are used exclusively
in the repository / database layer. Service code works with Pydantic
domain models; conversion happens at the repository boundary.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all VectorForge models."""


class CollectionModel(Base):
    """Collections table — groups of documents with shared config."""

    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("name", name="uq_collections_name"),
        Index("ix_collections_name", "name", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    embedding_config: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    chunking_config: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, onupdate=_utcnow
    )

    # Relationships
    documents: Mapped[list[DocumentModel]] = relationship(
        "DocumentModel", back_populates="collection", cascade="all, delete-orphan"
    )
    query_logs: Mapped[list[QueryLogModel]] = relationship(
        "QueryLogModel", back_populates="collection", cascade="all, delete-orphan"
    )


class DocumentModel(Base):
    """Documents table — ingested files/content belonging to a collection."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_collection_id", "collection_id"),
        Index("ix_documents_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    source_uri: Mapped[str] = mapped_column(String(1024), default="")
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_backend: Mapped[str] = mapped_column(String(50), default="pg", server_default="pg")
    s3_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    doc_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}",
    )
    status: Mapped[str] = mapped_column(String(50), default="pending", server_default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, onupdate=_utcnow
    )

    # Relationships
    collection: Mapped[CollectionModel] = relationship(
        "CollectionModel", back_populates="documents"
    )
    chunks: Mapped[list[ChunkModel]] = relationship(
        "ChunkModel", back_populates="document", cascade="all, delete-orphan"
    )


class ChunkModel(Base):
    """Chunks table — segments of a document ready for embedding."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_document_id_index", "document_id", "chunk_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, default=0)
    end_char: Mapped[int] = mapped_column(Integer, default=0)
    chunk_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata", JSONB, default=dict, server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    # Relationships
    document: Mapped[DocumentModel] = relationship("DocumentModel", back_populates="chunks")
    embedding: Mapped[EmbeddingModel | None] = relationship(
        "EmbeddingModel",
        back_populates="chunk",
        uselist=False,
        cascade="all, delete-orphan",
    )


class EmbeddingModel(Base):
    """Embeddings table — vector representations of chunks."""

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_embeddings_chunk_id"),
        Index("ix_embeddings_chunk_id", "chunk_id", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    # Relationships
    chunk: Mapped[ChunkModel] = relationship("ChunkModel", back_populates="embedding")


class QueryLogModel(Base):
    """Query logs table — records of RAG queries for analytics."""

    __tablename__ = "query_logs"
    __table_args__ = (
        Index("ix_query_logs_collection_id", "collection_id"),
        Index("ix_query_logs_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    generated_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    # Relationships
    collection: Mapped[CollectionModel] = relationship(
        "CollectionModel", back_populates="query_logs"
    )

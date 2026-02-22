"""Initial schema — create all foundation tables.

Revision ID: 001
Revises: None
Create Date: 2025-01-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    _uuid_col = postgresql.UUID(as_uuid=True)
    _uuid_default = sa.text("gen_random_uuid()")
    _now_default = sa.func.now()

    # --- collections ---
    op.create_table(
        "collections",
        sa.Column(
            "id",
            _uuid_col,
            server_default=_uuid_default,
            primary_key=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("embedding_config", postgresql.JSONB(), nullable=True),
        sa.Column("chunking_config", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_now_default,
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("name", name="uq_collections_name"),
    )
    op.create_index("ix_collections_name", "collections", ["name"], unique=True)

    # --- documents ---
    op.create_table(
        "documents",
        sa.Column(
            "id",
            _uuid_col,
            server_default=_uuid_default,
            primary_key=True,
        ),
        sa.Column(
            "collection_id",
            _uuid_col,
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_uri", sa.String(1024), server_default="", nullable=False),
        sa.Column("content_type", sa.String(255), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=True),
        sa.Column(
            "storage_backend",
            sa.String(50),
            server_default="pg",
            nullable=False,
        ),
        sa.Column("s3_key", sa.String(1024), nullable=True),
        sa.Column(
            "content_size_bytes",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(50),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_now_default,
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_documents_collection_id", "documents", ["collection_id"])
    op.create_index("ix_documents_status", "documents", ["status"])

    # --- chunks ---
    op.create_table(
        "chunks",
        sa.Column(
            "id",
            _uuid_col,
            server_default=_uuid_default,
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            _uuid_col,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), server_default="0", nullable=False),
        sa.Column("end_char", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_now_default,
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index(
        "ix_chunks_document_id_index",
        "chunks",
        ["document_id", "chunk_index"],
    )

    # --- embeddings ---
    op.create_table(
        "embeddings",
        sa.Column(
            "id",
            _uuid_col,
            server_default=_uuid_default,
            primary_key=True,
        ),
        sa.Column(
            "chunk_id",
            _uuid_col,
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_now_default,
            nullable=False,
        ),
        sa.UniqueConstraint("chunk_id", name="uq_embeddings_chunk_id"),
    )
    op.create_index(
        "ix_embeddings_chunk_id",
        "embeddings",
        ["chunk_id"],
        unique=True,
    )

    # --- query_logs ---
    op.create_table(
        "query_logs",
        sa.Column(
            "id",
            _uuid_col,
            server_default=_uuid_default,
            primary_key=True,
        ),
        sa.Column(
            "collection_id",
            _uuid_col,
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("retrieved_chunk_ids", postgresql.JSONB(), nullable=True),
        sa.Column("generated_response", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_now_default,
            nullable=False,
        ),
    )
    op.create_index(
        "ix_query_logs_collection_id",
        "query_logs",
        ["collection_id"],
    )
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("query_logs")
    op.drop_table("embeddings")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("collections")
    op.execute("DROP EXTENSION IF EXISTS vector")

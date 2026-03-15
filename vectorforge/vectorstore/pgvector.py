"""pgvector-backed vector store implementation."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from sqlalchemy import delete, func, select, text, type_coerce
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from vectorforge.models.db import ChunkModel, DocumentModel, EmbeddingModel
from vectorforge.models.domain import (
    Chunk,
    DistanceMetric,
    DocumentStatus,
    RetrievedChunk,
)
from vectorforge.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

_VALID_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str) -> None:
    """Validate a SQL identifier to prevent injection.

    Args:
        name: The identifier to validate.

    Raises:
        ValueError: If the identifier contains invalid characters.
    """
    if not _VALID_IDENTIFIER_RE.match(name):
        msg = f"Invalid SQL identifier: {name!r}"
        raise ValueError(msg)


def _get_distance_expression(
    metric: DistanceMetric, query_vector: list[float]
) -> Any:
    """Build the SQLAlchemy distance expression for the given metric.

    Args:
        metric: The distance metric to use.
        query_vector: The query embedding vector.

    Returns:
        A SQLAlchemy column expression for distance.
    """
    if metric == DistanceMetric.COSINE:
        return EmbeddingModel.embedding.cosine_distance(query_vector)
    if metric == DistanceMetric.L2:
        return EmbeddingModel.embedding.l2_distance(query_vector)
    # INNER_PRODUCT
    return EmbeddingModel.embedding.max_inner_product(query_vector)


def _distance_to_similarity(metric: DistanceMetric, distance: float) -> float:
    """Convert a raw distance value to a 0-1 similarity score.

    Args:
        metric: The distance metric used.
        distance: The raw distance value.

    Returns:
        A similarity score between 0.0 and 1.0.
    """
    if metric == DistanceMetric.COSINE:
        return max(0.0, 1.0 - distance)
    if metric == DistanceMetric.L2:
        return 1.0 / (1.0 + distance)
    # INNER_PRODUCT
    return max(0.0, min(1.0, -distance))


_METRIC_TO_INDEX_OPS: dict[DistanceMetric, str] = {
    DistanceMetric.COSINE: "vector_cosine_ops",
    DistanceMetric.L2: "vector_l2_ops",
    DistanceMetric.INNER_PRODUCT: "vector_ip_ops",
}


class PgVectorStore(BaseVectorStore):
    """Vector store backed by PostgreSQL + pgvector.

    Supports upsert (INSERT ON CONFLICT), similarity search with
    3 distance metrics, and HNSW index management.

    Args:
        session_factory: An async_sessionmaker for creating DB sessions.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert(
        self,
        chunk_ids: list[uuid.UUID],
        embeddings: list[list[float]],
        model_name: str = "",
    ) -> None:
        """Insert or update embedding vectors using ON CONFLICT.

        Args:
            chunk_ids: List of chunk UUIDs.
            embeddings: List of embedding vectors.
            model_name: The embedding model name.
        """
        if len(chunk_ids) != len(embeddings):
            msg = f"chunk_ids ({len(chunk_ids)}) and embeddings ({len(embeddings)}) length mismatch"
            raise ValueError(msg)

        batch_size = 500
        async with self._session_factory() as session:
            for start in range(0, len(chunk_ids), batch_size):
                batch_chunk_ids = chunk_ids[start : start + batch_size]
                batch_embeddings = embeddings[start : start + batch_size]
                rows = [
                    {
                        "id": uuid.uuid4(),
                        "chunk_id": cid,
                        "model_name": model_name,
                        "dimensions": len(emb),
                        "embedding": emb,
                    }
                    for cid, emb in zip(batch_chunk_ids, batch_embeddings, strict=True)
                ]
                stmt = pg_insert(EmbeddingModel).values(rows)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_embeddings_chunk_id",
                    set_={
                        "embedding": stmt.excluded.embedding,
                        "model_name": stmt.excluded.model_name,
                        "dimensions": stmt.excluded.dimensions,
                    },
                )
                await session.execute(stmt)
            await session.commit()

        logger.info("Upserted %d embeddings", len(chunk_ids))

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        collection_id: uuid.UUID | None = None,
    ) -> list[RetrievedChunk]:
        """Search for similar vectors using the specified distance metric.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results.
            filters: Optional JSONB metadata filters on chunks.
            metric: Distance metric to use.
            collection_id: Optional collection scope.

        Returns:
            List of RetrievedChunk results ordered by similarity (best first).
        """
        distance_expr = _get_distance_expression(metric, query_vector)

        stmt = (
            select(
                ChunkModel,
                DocumentModel.source_uri,
                distance_expr.label("distance"),
            )
            .join(EmbeddingModel, EmbeddingModel.chunk_id == ChunkModel.id)
            .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
        )

        if collection_id is not None:
            stmt = stmt.where(DocumentModel.collection_id == collection_id)

        stmt = stmt.where(DocumentModel.status == DocumentStatus.INDEXED.value)

        if filters:
            for key, value in filters.items():
                json_str = f'{{"{key}": "{value}"}}'
                stmt = stmt.where(
                    ChunkModel.chunk_metadata.op("@>")(
                        type_coerce(json_str, JSONB)
                    )
                )

        stmt = stmt.order_by(text("distance")).limit(top_k)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.all()

        retrieved: list[RetrievedChunk] = []
        for row in rows:
            chunk_model = row[0]
            source_uri = row[1]
            distance = float(row[2])
            similarity = _distance_to_similarity(metric, distance)
            chunk = Chunk(
                id=chunk_model.id,
                document_id=chunk_model.document_id,
                text=chunk_model.content,
                index=chunk_model.chunk_index,
                start_char=chunk_model.start_char,
                end_char=chunk_model.end_char,
                metadata=chunk_model.chunk_metadata,
                created_at=chunk_model.created_at,
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=similarity,
                    document_source=source_uri,
                )
            )

        return retrieved

    async def delete_by_ids(self, chunk_ids: list[uuid.UUID]) -> None:
        """Delete embeddings by chunk IDs.

        Args:
            chunk_ids: List of chunk UUIDs.
        """
        if not chunk_ids:
            return
        async with self._session_factory() as session:
            await session.execute(
                delete(EmbeddingModel).where(EmbeddingModel.chunk_id.in_(chunk_ids))
            )
            await session.commit()

    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        """Delete all embeddings for chunks belonging to a document.

        Args:
            document_id: The document UUID.
        """
        chunk_ids_subq = (
            select(ChunkModel.id)
            .where(ChunkModel.document_id == document_id)
            .scalar_subquery()
        )
        async with self._session_factory() as session:
            await session.execute(
                delete(EmbeddingModel).where(EmbeddingModel.chunk_id.in_(chunk_ids_subq))
            )
            await session.commit()

    async def count(self) -> int:
        """Count total stored embeddings.

        Returns:
            Number of embedding records.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count()).select_from(EmbeddingModel)
            )
            count: int = result.scalar_one()
            return count

    async def create_index(
        self,
        collection_id: uuid.UUID,
        metric: DistanceMetric = DistanceMetric.COSINE,
        m: int = 16,
        ef_construction: int = 64,
    ) -> None:
        """Create an HNSW index for a collection.

        Args:
            collection_id: The collection UUID.
            metric: Distance metric for the index.
            m: HNSW max connections per node.
            ef_construction: HNSW search depth during build.
        """
        ops = _METRIC_TO_INDEX_OPS[metric]
        index_name = f"ix_embeddings_{str(collection_id).replace('-', '_')}_{metric.value}"
        _validate_identifier(index_name)

        sql = text(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON embeddings USING hnsw (embedding {ops}) "
            f"WITH (m = :m, ef_construction = :ef_construction)"
        )

        async with self._session_factory() as session:
            await session.execute(sql, {"m": m, "ef_construction": ef_construction})
            await session.commit()

        logger.info(
            "Created HNSW index %s for collection %s (%s)",
            index_name,
            collection_id,
            metric.value,
        )

    async def drop_index(self, collection_id: uuid.UUID) -> None:
        """Drop all HNSW indexes for a collection.

        Args:
            collection_id: The collection UUID.
        """
        prefix = f"ix_embeddings_{str(collection_id).replace('-', '_')}_"
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'embeddings' AND indexname LIKE :prefix"
                ),
                {"prefix": f"{prefix}%"},
            )
            for row in result.all():
                index_name = row[0]
                _validate_identifier(index_name)
                await session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
            await session.commit()

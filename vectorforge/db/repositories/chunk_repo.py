"""Chunk repository — data access for the chunks table."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select

from vectorforge.db.repositories.base import BaseRepository
from vectorforge.models.db import ChunkModel
from vectorforge.models.domain import Chunk, CreateChunkDTO


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for managing chunk records."""

    _model_class = ChunkModel

    def _to_domain(self, instance: ChunkModel) -> Chunk:
        """Convert a ChunkModel ORM instance to a Chunk domain model."""
        return Chunk(
            id=instance.id,
            document_id=instance.document_id,
            text=instance.content,
            index=instance.chunk_index,
            start_char=instance.start_char,
            end_char=instance.end_char,
            metadata=instance.chunk_metadata,
            created_at=instance.created_at,
        )

    async def create(self, data: CreateChunkDTO) -> Chunk:
        """Insert a single chunk, mapping DTO fields to ORM attributes.

        Args:
            data: A CreateChunkDTO with the chunk fields.

        Returns:
            The newly created Chunk domain model.
        """
        instance = ChunkModel(
            document_id=data.document_id,
            content=data.text,
            chunk_index=data.index,
            start_char=data.start_char,
            end_char=data.end_char,
            chunk_metadata=data.metadata,
        )
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def find_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        """Find all chunks belonging to a document, ordered by index.

        Args:
            document_id: The parent document's UUID.

        Returns:
            List of Chunks ordered by chunk_index.
        """
        result = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.chunk_index)
        )
        return [self._to_domain(row) for row in result.scalars().all()]

    async def find_by_index_range(
        self,
        document_id: uuid.UUID,
        start: int,
        end: int,
    ) -> list[Chunk]:
        """Find chunks within an index range for a document.

        Args:
            document_id: The parent document's UUID.
            start: Start index (inclusive).
            end: End index (inclusive).

        Returns:
            List of Chunks within the index range.
        """
        result = await self._session.execute(
            select(ChunkModel)
            .where(
                ChunkModel.document_id == document_id,
                ChunkModel.chunk_index >= start,
                ChunkModel.chunk_index <= end,
            )
            .order_by(ChunkModel.chunk_index)
        )
        return [self._to_domain(row) for row in result.scalars().all()]

    async def bulk_create(self, chunks: list[CreateChunkDTO]) -> list[Chunk]:
        """Insert multiple chunks in a single batch.

        Uses a single flush followed by a batch re-query to avoid
        N+1 refresh calls.

        Args:
            chunks: List of CreateChunkDTO objects.

        Returns:
            List of created Chunk domain models.
        """
        instances = [
            ChunkModel(
                document_id=dto.document_id,
                content=dto.text,
                chunk_index=dto.index,
                start_char=dto.start_char,
                end_char=dto.end_char,
                chunk_metadata=dto.metadata,
            )
            for dto in chunks
        ]
        self._session.add_all(instances)
        await self._session.flush()

        ids = [inst.id for inst in instances]
        result = await self._session.execute(
            select(ChunkModel).where(ChunkModel.id.in_(ids))
        )
        refreshed = result.scalars().all()
        return [self._to_domain(inst) for inst in refreshed]

    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The parent document's UUID.
        """
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
        await self._session.flush()

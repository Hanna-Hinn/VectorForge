"""Embedding repository — data access for the embeddings table."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select

from vectorforge.db.repositories.base import BaseRepository
from vectorforge.models.db import ChunkModel, EmbeddingModel
from vectorforge.models.domain import CreateEmbeddingDTO, Embedding


class EmbeddingRepository(BaseRepository[Embedding]):
    """Repository for managing embedding records."""

    _model_class = EmbeddingModel

    def _to_domain(self, instance: EmbeddingModel) -> Embedding:
        """Convert an EmbeddingModel ORM instance to an Embedding domain model."""
        return Embedding(
            id=instance.id,
            chunk_id=instance.chunk_id,
            model_name=instance.model_name,
            dimensions=instance.dimensions,
            vector=list(instance.embedding),
            created_at=instance.created_at,
        )

    @staticmethod
    def _validate_dimensions(dto: CreateEmbeddingDTO) -> None:
        """Verify that the vector length matches the declared dimensions.

        Args:
            dto: The embedding DTO to validate.

        Raises:
            ValueError: If vector length does not match dimensions.
        """
        if len(dto.vector) != dto.dimensions:
            msg = (
                f"Vector length ({len(dto.vector)}) does not match "
                f"declared dimensions ({dto.dimensions})"
            )
            raise ValueError(msg)

    async def find_by_chunk(self, chunk_id: uuid.UUID) -> Embedding | None:
        """Find the embedding for a specific chunk.

        Args:
            chunk_id: The chunk's UUID.

        Returns:
            The Embedding if found, otherwise None.
        """
        result = await self._session.execute(
            select(EmbeddingModel).where(EmbeddingModel.chunk_id == chunk_id)
        )
        instance = result.scalar_one_or_none()
        return self._to_domain(instance) if instance else None

    async def create(self, data: CreateEmbeddingDTO) -> Embedding:
        """Insert a single embedding, validating dimension consistency.

        Args:
            data: A CreateEmbeddingDTO with the embedding fields.

        Returns:
            The newly created Embedding domain model.

        Raises:
            ValueError: If vector length does not match dimensions.
        """
        self._validate_dimensions(data)
        instance = EmbeddingModel(
            chunk_id=data.chunk_id,
            model_name=data.model_name,
            dimensions=data.dimensions,
            embedding=data.vector,
        )
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def bulk_create(self, embeddings: list[CreateEmbeddingDTO]) -> list[Embedding]:
        """Insert multiple embeddings in a single batch.

        Validates dimension consistency for all embeddings before
        inserting. Uses a single flush followed by a batch re-query
        to avoid N+1 refresh calls.

        Args:
            embeddings: List of CreateEmbeddingDTO objects.

        Returns:
            List of created Embedding domain models.

        Raises:
            ValueError: If any vector length does not match its dimensions.
        """
        for dto in embeddings:
            self._validate_dimensions(dto)

        instances = [
            EmbeddingModel(
                chunk_id=dto.chunk_id,
                model_name=dto.model_name,
                dimensions=dto.dimensions,
                embedding=dto.vector,
            )
            for dto in embeddings
        ]
        self._session.add_all(instances)
        await self._session.flush()

        ids = [inst.id for inst in instances]
        result = await self._session.execute(
            select(EmbeddingModel).where(EmbeddingModel.id.in_(ids))
        )
        refreshed = result.scalars().all()
        return [self._to_domain(inst) for inst in refreshed]

    async def delete_by_document(self, document_id: uuid.UUID) -> None:
        """Delete all embeddings for chunks belonging to a document.

        Args:
            document_id: The parent document's UUID.
        """
        chunk_ids_subquery = (
            select(ChunkModel.id).where(ChunkModel.document_id == document_id).scalar_subquery()
        )
        await self._session.execute(
            delete(EmbeddingModel).where(EmbeddingModel.chunk_id.in_(chunk_ids_subquery))
        )
        await self._session.flush()

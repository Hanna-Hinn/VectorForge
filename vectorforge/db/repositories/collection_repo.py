"""Collection repository — data access for the collections table."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from vectorforge.db.repositories.base import BaseRepository
from vectorforge.exceptions import DuplicateError
from vectorforge.models.db import CollectionModel
from vectorforge.models.domain import Collection, CreateCollectionDTO


class CollectionRepository(BaseRepository[Collection]):
    """Repository for managing collection records.

    Handles unique-name enforcement and cascading delete.
    """

    _model_class = CollectionModel

    def _to_domain(self, instance: CollectionModel) -> Collection:
        """Convert a CollectionModel ORM instance to a Collection domain model."""
        return Collection.model_validate(instance)

    async def find_by_name(self, name: str) -> Collection | None:
        """Find a collection by its unique name.

        Args:
            name: The collection name to search for.

        Returns:
            The Collection if found, otherwise None.
        """
        result = await self._session.execute(
            select(CollectionModel).where(CollectionModel.name == name)
        )
        instance = result.scalar_one_or_none()
        return self._to_domain(instance) if instance else None

    async def create(self, data: CreateCollectionDTO) -> Collection:
        """Create a new collection, enforcing name uniqueness.

        Uses an application-level check followed by a database-level
        ``IntegrityError`` catch to prevent TOCTOU races.

        Args:
            data: DTO with collection creation fields.

        Returns:
            The newly created Collection.

        Raises:
            DuplicateError: If a collection with the same name already exists.
        """
        existing = await self.find_by_name(data.name)
        if existing is not None:
            msg = f"Collection '{data.name}' already exists"
            raise DuplicateError(msg)

        instance = CollectionModel(
            name=data.name,
            description=data.description,
            embedding_config=data.embedding_config,
            chunking_config=data.chunking_config,
        )
        self._session.add(instance)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            msg = f"Collection '{data.name}' already exists"
            raise DuplicateError(msg) from exc
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def delete(self, id: uuid.UUID) -> None:
        """Delete a collection and all its dependent records.

        Cascading delete is handled by SQLAlchemy relationship config
        (cascade='all, delete-orphan').

        Args:
            id: The UUID of the collection to delete.

        Raises:
            NotFoundError: If no collection exists with the given id.
        """
        await super().delete(id)

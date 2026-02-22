"""Abstract base repository providing generic CRUD operations.

All concrete repositories inherit from BaseRepository and work with
SQLAlchemy ORM models internally, converting to/from Pydantic domain
models at the boundary.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import class_mapper

from vectorforge.exceptions import NotFoundError

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Generic repository ABC for standard CRUD operations.

    Subclasses must define ``_model_class`` pointing to the SQLAlchemy
    ORM model and implement ``_to_domain`` for conversion.

    Type Parameters:
        T: The Pydantic domain model type.

    Args:
        session: An active async database session.
    """

    _model_class: type[Any]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @abstractmethod
    def _to_domain(self, instance: Any) -> T:
        """Convert an ORM model instance to a Pydantic domain model."""

    async def find_by_id(self, id: uuid.UUID) -> T | None:
        """Find a single record by primary key.

        Args:
            id: The UUID primary key.

        Returns:
            The domain model if found, otherwise None.
        """
        result = await self._session.execute(
            select(self._model_class).where(self._model_class.id == id)
        )
        instance = result.scalar_one_or_none()
        return self._to_domain(instance) if instance else None

    async def find_all(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[T]:
        """Retrieve multiple records with optional filtering and pagination.

        Args:
            filters: Column-value pairs to filter by.
            limit: Maximum records to return.
            offset: Number of records to skip.

        Returns:
            List of domain model instances.
        """
        stmt = select(self._model_class)
        if filters:
            valid_columns = {
                c.key for c in class_mapper(self._model_class).column_attrs
            }
            for key, value in filters.items():
                if key not in valid_columns:
                    msg = (
                        f"Invalid filter key '{key}' for "
                        f"{self._model_class.__name__}"
                    )
                    raise ValueError(msg)
                stmt = stmt.where(getattr(self._model_class, key) == value)
        stmt = stmt.order_by(self._model_class.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self._session.execute(stmt)
        return [self._to_domain(row) for row in result.scalars().all()]

    async def create(self, data: Any) -> T:
        """Insert a new record from a DTO.

        Args:
            data: A Pydantic DTO with creation fields.

        Returns:
            The newly created domain model with server-generated fields.
        """
        instance = self._model_class(**data.model_dump())
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def update(self, id: uuid.UUID, data: Any) -> T:
        """Update an existing record with partial data.

        Args:
            id: The UUID of the record to update.
            data: A Pydantic DTO with fields to update (only set fields applied).

        Returns:
            The updated domain model.

        Raises:
            NotFoundError: If no record exists with the given id.
        """
        result = await self._session.execute(
            select(self._model_class).where(self._model_class.id == id)
        )
        instance = result.scalar_one_or_none()
        if instance is None:
            msg = f"{self._model_class.__name__} with id={id} not found"
            raise NotFoundError(msg)

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(instance, field, value)

        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def delete(self, id: uuid.UUID) -> None:
        """Delete a record by primary key.

        Args:
            id: The UUID of the record to delete.

        Raises:
            NotFoundError: If no record exists with the given id.
        """
        result = await self._session.execute(
            select(self._model_class).where(self._model_class.id == id)
        )
        instance = result.scalar_one_or_none()
        if instance is None:
            msg = f"{self._model_class.__name__} with id={id} not found"
            raise NotFoundError(msg)

        await self._session.delete(instance)
        await self._session.flush()

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count records matching optional filters.

        Args:
            filters: Column-value pairs to filter by.

        Returns:
            Number of matching records.
        """
        stmt = select(func.count()).select_from(self._model_class)
        if filters:
            valid_columns = {
                c.key for c in class_mapper(self._model_class).column_attrs
            }
            for key, value in filters.items():
                if key not in valid_columns:
                    msg = (
                        f"Invalid filter key '{key}' for "
                        f"{self._model_class.__name__}"
                    )
                    raise ValueError(msg)
                stmt = stmt.where(getattr(self._model_class, key) == value)

        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def exists(self, id: uuid.UUID) -> bool:
        """Check if a record exists by primary key.

        Args:
            id: The UUID to check.

        Returns:
            True if the record exists, False otherwise.
        """
        stmt = select(func.count()).select_from(self._model_class).where(self._model_class.id == id)
        result = await self._session.execute(stmt)
        return result.scalar_one() > 0

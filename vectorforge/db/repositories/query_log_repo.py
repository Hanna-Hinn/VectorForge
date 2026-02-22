"""Query log repository — data access for the query_logs table."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from vectorforge.db.repositories.base import BaseRepository
from vectorforge.models.db import QueryLogModel
from vectorforge.models.domain import CreateQueryLogDTO, QueryLog


class QueryLogRepository(BaseRepository[QueryLog]):
    """Repository for managing query log records."""

    _model_class = QueryLogModel

    def _to_domain(self, instance: QueryLogModel) -> QueryLog:
        """Convert a QueryLogModel ORM instance to a QueryLog domain model."""
        return QueryLog(
            id=instance.id,
            collection_id=instance.collection_id,
            query_text=instance.query_text,
            retrieved_chunk_ids=instance.retrieved_chunk_ids,
            generated_response=instance.generated_response,
            latency_ms=instance.latency_ms,
            created_at=instance.created_at,
        )

    async def create(self, data: CreateQueryLogDTO) -> QueryLog:
        """Insert a new query log record.

        Args:
            data: A CreateQueryLogDTO with the query log fields.

        Returns:
            The newly created QueryLog domain model.
        """
        instance = QueryLogModel(**data.model_dump())
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def find_by_collection(
        self,
        collection_id: uuid.UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[QueryLog]:
        """Find query logs for a specific collection.

        Args:
            collection_id: The collection's UUID.
            limit: Maximum records to return.
            offset: Number of records to skip.

        Returns:
            List of QueryLog items ordered by most recent first.
        """
        result = await self._session.execute(
            select(QueryLogModel)
            .where(QueryLogModel.collection_id == collection_id)
            .order_by(QueryLogModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [self._to_domain(row) for row in result.scalars().all()]

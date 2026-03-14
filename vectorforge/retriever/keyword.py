"""Keyword-based retriever using PostgreSQL full-text search.

Uses ``tsvector`` / ``tsquery`` through SQLAlchemy to find chunks
matching the query terms.  This provides a sparse (BM25-like)
complement to the dense vector retriever.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from vectorforge.models.db import ChunkModel, DocumentModel
from vectorforge.models.domain import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class BaseKeywordSearcher(ABC):
    """Abstract base class for keyword search strategies."""

    @abstractmethod
    async def search(
        self,
        query: str,
        collection_id: uuid.UUID,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Search chunks by keyword relevance.

        Args:
            query: The user query text.
            collection_id: The collection to search within.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedChunk results ordered by keyword rank.
        """


class KeywordSearcher(BaseKeywordSearcher):
    """Full-text keyword search over chunk content.

    Executes PostgreSQL ``ts_rank`` + ``plainto_tsquery`` to score
    chunks by keyword relevance.

    Args:
        session: An active async database session.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self,
        query: str,
        collection_id: uuid.UUID,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Search chunks by keyword relevance.

        Args:
            query: The user query text.
            collection_id: The collection to search within.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedChunk results ordered by keyword rank.
        """
        if not query.strip():
            return []

        ts_query = func.plainto_tsquery("english", query)
        ts_vector = func.to_tsvector("english", ChunkModel.content)
        rank = func.ts_rank(ts_vector, ts_query).label("rank")

        stmt = (
            select(ChunkModel, rank, DocumentModel.source_uri)
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
            .where(DocumentModel.collection_id == collection_id)
            .where(ts_vector.op("@@")(ts_query))
            .order_by(literal_column("rank").desc())
            .limit(top_k)
        )

        result = await self._session.execute(stmt)
        rows = result.all()

        return [self._to_retrieved(row) for row in rows]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_retrieved(row: Any) -> RetrievedChunk:
        """Convert a result row to a RetrievedChunk.

        Args:
            row: A SQLAlchemy Row with (ChunkModel, rank, source_uri).

        Returns:
            A RetrievedChunk domain model.
        """
        chunk_model, rank, source_uri = row
        chunk = Chunk(
            id=chunk_model.id,
            document_id=chunk_model.document_id,
            text=chunk_model.content,
            index=chunk_model.chunk_index,
            start_char=chunk_model.start_char,
            end_char=chunk_model.end_char,
            metadata=chunk_model.chunk_metadata or {},
            created_at=chunk_model.created_at,
        )
        return RetrievedChunk(
            chunk=chunk,
            score=float(rank),
            document_source=source_uri or "",
        )

"""Document repository — data access for the documents table."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update

from vectorforge.db.repositories.base import BaseRepository
from vectorforge.exceptions import NotFoundError
from vectorforge.models.db import DocumentModel
from vectorforge.models.domain import CreateDocumentDTO, Document, DocumentStatus


class DocumentRepository(BaseRepository[Document]):
    """Repository for managing document records."""

    _model_class = DocumentModel

    def _to_domain(self, instance: DocumentModel) -> Document:
        """Convert a DocumentModel ORM instance to a Document domain model."""
        return Document(
            id=instance.id,
            collection_id=instance.collection_id,
            source_uri=instance.source_uri,
            content_type=instance.content_type,
            raw_content=instance.raw_content,
            storage_backend=instance.storage_backend,
            s3_key=instance.s3_key,
            content_size_bytes=instance.content_size_bytes,
            metadata=instance.doc_metadata,
            status=DocumentStatus(instance.status),
            created_at=instance.created_at,
            updated_at=instance.updated_at,
        )

    async def create(self, data: CreateDocumentDTO) -> Document:
        """Insert a new document, mapping DTO fields to ORM attributes.

        Args:
            data: A CreateDocumentDTO with the document fields.

        Returns:
            The newly created Document domain model.
        """
        dump = data.model_dump()
        dump["doc_metadata"] = dump.pop("metadata", {})
        dump["raw_content"] = dump.pop("content", None)
        instance = DocumentModel(**dump)
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return self._to_domain(instance)

    async def find_by_collection(self, collection_id: uuid.UUID) -> list[Document]:
        """Find all documents belonging to a collection.

        Args:
            collection_id: The parent collection's UUID.

        Returns:
            List of Documents in the collection.
        """
        result = await self._session.execute(
            select(DocumentModel)
            .where(DocumentModel.collection_id == collection_id)
            .order_by(DocumentModel.created_at.desc())
        )
        return [self._to_domain(row) for row in result.scalars().all()]

    async def find_by_source_uri(self, uri: str) -> Document | None:
        """Find a document by its source URI.

        Args:
            uri: The source URI to search for.

        Returns:
            The Document if found, otherwise None.
        """
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.source_uri == uri)
        )
        instance = result.scalar_one_or_none()
        return self._to_domain(instance) if instance else None

    async def find_by_status(self, status: DocumentStatus) -> list[Document]:
        """Find all documents with a given processing status.

        Args:
            status: The DocumentStatus to filter by.

        Returns:
            List of Documents matching the status.
        """
        result = await self._session.execute(
            select(DocumentModel)
            .where(DocumentModel.status == status.value)
            .order_by(DocumentModel.created_at.desc())
        )
        return [self._to_domain(row) for row in result.scalars().all()]

    async def update_status(self, id: uuid.UUID, status: DocumentStatus) -> None:
        """Update the processing status of a document.

        Args:
            id: The document UUID.
            status: The new DocumentStatus.

        Raises:
            NotFoundError: If no document exists with the given id.
        """
        result = await self._session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == id)
            .values(status=status.value)
            .returning(DocumentModel.id)
        )
        if result.scalar_one_or_none() is None:
            msg = f"DocumentModel with id={id} not found"
            raise NotFoundError(msg)
        await self._session.flush()

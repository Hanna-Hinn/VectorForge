"""Document management and ingestion endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Query, status

from server.dependencies import ApiKey, DbSession, IngestionDep
from server.schemas import (
    BatchIngestResponse,
    BatchIngestResult,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    IngestDocumentRequest,
    MessageResponse,
)
from vectorforge.config.settings import ChunkingConfig
from vectorforge.db.repositories.collection_repo import CollectionRepository
from vectorforge.db.repositories.document_repo import DocumentRepository
from vectorforge.exceptions import NotFoundError, VectorForgeError
from vectorforge.models.domain import Document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


def _to_response(doc: Document) -> DocumentResponse:
    """Convert a domain Document to an API response schema."""
    return DocumentResponse(
        id=doc.id,
        collection_id=doc.collection_id,
        source_uri=doc.source_uri,
        content_type=doc.content_type,
        status=doc.status.value,
        content_size_bytes=doc.content_size_bytes,
        metadata=doc.metadata,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def _to_detail_response(doc: Document) -> DocumentDetailResponse:
    """Convert a domain Document to a detailed API response."""
    return DocumentDetailResponse(
        id=doc.id,
        collection_id=doc.collection_id,
        source_uri=doc.source_uri,
        content_type=doc.content_type,
        status=doc.status.value,
        content_size_bytes=doc.content_size_bytes,
        metadata=doc.metadata,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        raw_content=doc.raw_content,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/collections/{collection_id}/documents",
    response_model=DocumentListResponse,
)
async def list_documents(
    collection_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    """List documents in a collection."""
    col_repo = CollectionRepository(session)
    collection = await col_repo.find_by_id(collection_id)
    if collection is None:
        msg = f"Collection {collection_id} not found"
        raise NotFoundError(msg)

    doc_repo = DocumentRepository(session)
    total = await doc_repo.count_by_collection(collection_id)
    documents = await doc_repo.find_by_collection(
        collection_id, limit=limit, offset=offset,
    )
    return DocumentListResponse(
        documents=[_to_response(d) for d in documents],
        total=total,
    )


@router.post(
    "/collections/{collection_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    collection_id: uuid.UUID,
    body: IngestDocumentRequest,
    session: DbSession,
    ingestion: IngestionDep,
    _key: ApiKey,
) -> DocumentResponse:
    """Ingest a single document into a collection."""
    col_repo = CollectionRepository(session)
    collection = await col_repo.find_by_id(collection_id)
    if collection is None:
        msg = f"Collection {collection_id} not found"
        raise NotFoundError(msg)

    chunking_config: ChunkingConfig | None = None
    if body.chunking_strategy or body.chunk_size or body.chunk_overlap:
        overrides: dict[str, object] = {}
        if body.chunking_strategy:
            overrides["strategy"] = body.chunking_strategy
        if body.chunk_size is not None:
            overrides["chunk_size"] = body.chunk_size
        if body.chunk_overlap is not None:
            overrides["chunk_overlap"] = body.chunk_overlap
        chunking_config = ChunkingConfig(**overrides)  # type: ignore[arg-type]

    document = await ingestion.ingest(
        source=body.source,
        collection_id=collection_id,
        session=session,
        metadata=body.metadata,
        chunking_config=chunking_config,
    )
    return _to_response(document)


@router.post(
    "/collections/{collection_id}/documents/batch",
    response_model=BatchIngestResponse,
)
async def batch_ingest_documents(
    collection_id: uuid.UUID,
    bodies: list[IngestDocumentRequest],
    session: DbSession,
    ingestion: IngestionDep,
    _key: ApiKey,
) -> BatchIngestResponse:
    """Batch ingest multiple documents into a collection."""
    col_repo = CollectionRepository(session)
    collection = await col_repo.find_by_id(collection_id)
    if collection is None:
        msg = f"Collection {collection_id} not found"
        raise NotFoundError(msg)

    results: list[BatchIngestResult] = []
    succeeded = 0
    failed = 0

    for body in bodies:
        try:
            doc = await ingestion.ingest(
                source=body.source,
                collection_id=collection_id,
                session=session,
                metadata=body.metadata,
            )
            results.append(BatchIngestResult(source=body.source, document_id=doc.id))
            succeeded += 1
        except VectorForgeError as exc:
            results.append(BatchIngestResult(source=body.source, error=str(exc)))
            failed += 1
            logger.warning("Batch ingest failed for %s: %s", body.source, exc)

    return BatchIngestResponse(results=results, succeeded=succeeded, failed=failed)


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
) -> DocumentDetailResponse:
    """Get a single document by ID."""
    repo = DocumentRepository(session)
    document = await repo.find_by_id(document_id)
    if document is None:
        msg = f"Document {document_id} not found"
        raise NotFoundError(msg)
    return _to_detail_response(document)


@router.delete("/documents/{document_id}", response_model=MessageResponse)
async def delete_document(
    document_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
) -> MessageResponse:
    """Delete a document and all its chunks/embeddings."""
    repo = DocumentRepository(session)
    await repo.delete(document_id)
    return MessageResponse(message="Document deleted")

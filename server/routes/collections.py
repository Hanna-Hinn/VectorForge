"""Collection CRUD endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, status

from server.dependencies import ApiKey, DbSession
from server.schemas import (
    CollectionListResponse,
    CollectionResponse,
    CreateCollectionRequest,
    MessageResponse,
)
from vectorforge.db.repositories.collection_repo import CollectionRepository
from vectorforge.exceptions import NotFoundError
from vectorforge.models.domain import CreateCollectionDTO, DistanceMetric

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])


def _to_response(col: object) -> CollectionResponse:
    """Convert a domain Collection to an API response schema."""
    from vectorforge.models.domain import Collection

    assert isinstance(col, Collection)
    return CollectionResponse(
        id=col.id,
        name=col.name,
        description=col.description,
        embedding_config=col.embedding_config,
        chunking_config=col.chunking_config,
        created_at=col.created_at,
        updated_at=col.updated_at,
    )


@router.get("", response_model=CollectionListResponse)
async def list_collections(
    session: DbSession,
    _key: ApiKey,
) -> CollectionListResponse:
    """List all collections."""
    repo = CollectionRepository(session)
    collections = await repo.find_all()
    logger.info("Listed %d collections", len(collections))
    return CollectionListResponse(
        collections=[_to_response(c) for c in collections],
    )


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    body: CreateCollectionRequest,
    session: DbSession,
    _key: ApiKey,
) -> CollectionResponse:
    """Create a new collection."""
    embedding_config: dict[str, object] = {}
    if body.embedding_provider:
        embedding_config["default_provider"] = body.embedding_provider
    if body.embedding_model:
        embedding_config["default_model"] = body.embedding_model
    if body.metric:
        embedding_config["metric"] = body.metric

    chunking_config: dict[str, object] = {}
    if body.chunking_strategy:
        chunking_config["strategy"] = body.chunking_strategy
    if body.chunk_size is not None:
        chunking_config["chunk_size"] = body.chunk_size
    if body.chunk_overlap is not None:
        chunking_config["chunk_overlap"] = body.chunk_overlap

    dto = CreateCollectionDTO(
        name=body.name,
        description=body.description,
        metric=DistanceMetric(body.metric),
        embedding_config=embedding_config or None,
        chunking_config=chunking_config or None,
    )

    repo = CollectionRepository(session)
    collection = await repo.create(dto)
    logger.info("Created collection %s (%s)", collection.name, collection.id)
    return _to_response(collection)


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
) -> CollectionResponse:
    """Get a single collection by ID."""
    repo = CollectionRepository(session)
    collection = await repo.find_by_id(collection_id)
    if collection is None:
        msg = f"Collection {collection_id} not found"
        raise NotFoundError(msg)
    return _to_response(collection)


@router.delete("/{collection_id}", response_model=MessageResponse)
async def delete_collection(
    collection_id: uuid.UUID,
    session: DbSession,
    _key: ApiKey,
) -> MessageResponse:
    """Delete a collection and all its data."""
    repo = CollectionRepository(session)
    await repo.delete(collection_id)
    logger.info("Deleted collection %s", collection_id)
    return MessageResponse(message="Collection deleted")

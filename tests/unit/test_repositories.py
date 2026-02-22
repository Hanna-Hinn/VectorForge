"""Unit tests for VectorForge repositories (mocked database)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from vectorforge.db.repositories.chunk_repo import ChunkRepository
from vectorforge.db.repositories.collection_repo import CollectionRepository
from vectorforge.db.repositories.document_repo import DocumentRepository
from vectorforge.db.repositories.embedding_repo import EmbeddingRepository
from vectorforge.db.repositories.query_log_repo import QueryLogRepository
from vectorforge.exceptions import DuplicateError, NotFoundError
from vectorforge.models.db import (
    ChunkModel,
    CollectionModel,
    DocumentModel,
    EmbeddingModel,
    QueryLogModel,
)
from vectorforge.models.domain import (
    CreateChunkDTO,
    CreateCollectionDTO,
    CreateEmbeddingDTO,
    CreateQueryLogDTO,
    DocumentStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(UTC)


def _make_collection_model(
    name: str = "test-col",
    id: uuid.UUID | None = None,
) -> CollectionModel:
    """Create a mock CollectionModel instance."""
    model = MagicMock(spec=CollectionModel)
    model.id = id or uuid.uuid4()
    model.name = name
    model.description = "desc"
    model.embedding_config = None
    model.chunking_config = None
    model.created_at = NOW
    model.updated_at = None
    return model


def _make_document_model(
    collection_id: uuid.UUID | None = None,
    status: str = "pending",
) -> DocumentModel:
    """Create a mock DocumentModel instance."""
    model = MagicMock(spec=DocumentModel)
    model.id = uuid.uuid4()
    model.collection_id = collection_id or uuid.uuid4()
    model.source_uri = "file:///test.txt"
    model.content_type = "text/plain"
    model.raw_content = "content"
    model.storage_backend = "pg"
    model.s3_key = None
    model.content_size_bytes = 100
    model.doc_metadata = {}
    model.status = status
    model.created_at = NOW
    model.updated_at = None
    return model


def _make_chunk_model(document_id: uuid.UUID | None = None) -> ChunkModel:
    """Create a mock ChunkModel instance."""
    model = MagicMock(spec=ChunkModel)
    model.id = uuid.uuid4()
    model.document_id = document_id or uuid.uuid4()
    model.content = "chunk text"
    model.chunk_index = 0
    model.start_char = 0
    model.end_char = 10
    model.chunk_metadata = {}
    model.created_at = NOW
    return model


def _make_embedding_model(chunk_id: uuid.UUID | None = None) -> EmbeddingModel:
    """Create a mock EmbeddingModel instance."""
    model = MagicMock(spec=EmbeddingModel)
    model.id = uuid.uuid4()
    model.chunk_id = chunk_id or uuid.uuid4()
    model.model_name = "voyage-3"
    model.dimensions = 4
    model.embedding = [0.1, 0.2, 0.3, 0.4]
    model.created_at = NOW
    return model


# ---------------------------------------------------------------------------
# Collection Repository Tests
# ---------------------------------------------------------------------------


class TestCollectionRepository:
    """Tests for CollectionRepository."""

    @pytest.fixture()
    def repo(self, mock_async_session: AsyncMock) -> CollectionRepository:
        """Provide a CollectionRepository with a mocked session."""
        return CollectionRepository(mock_async_session)

    async def test_find_by_name_found(
        self, repo: CollectionRepository, mock_async_session: AsyncMock
    ) -> None:
        """find_by_name returns a Collection when found."""
        model = _make_collection_model(name="found")
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = model
        mock_async_session.execute.return_value = result_mock

        result = await repo.find_by_name("found")
        assert result is not None
        assert result.name == "found"

    async def test_find_by_name_not_found(
        self, repo: CollectionRepository, mock_async_session: AsyncMock
    ) -> None:
        """find_by_name returns None when no match."""
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_async_session.execute.return_value = result_mock

        result = await repo.find_by_name("missing")
        assert result is None

    async def test_create_duplicate_raises(
        self, repo: CollectionRepository, mock_async_session: AsyncMock
    ) -> None:
        """create raises DuplicateError when name already exists."""
        existing = _make_collection_model(name="dup")
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = existing
        mock_async_session.execute.return_value = result_mock

        dto = CreateCollectionDTO(name="dup")
        with pytest.raises(DuplicateError, match="already exists"):
            await repo.create(dto)

    async def test_create_success(
        self, repo: CollectionRepository, mock_async_session: AsyncMock
    ) -> None:
        """create succeeds when name is unique."""
        # First call returns None (no existing), subsequent calls return the model
        no_result = MagicMock()
        no_result.scalar_one_or_none.return_value = None
        mock_async_session.execute.return_value = no_result

        new_model = _make_collection_model(name="new-col")

        async def fake_refresh(inst: CollectionModel) -> None:
            inst.id = new_model.id
            inst.name = new_model.name
            inst.description = new_model.description
            inst.embedding_config = new_model.embedding_config
            inst.chunking_config = new_model.chunking_config
            inst.created_at = new_model.created_at
            inst.updated_at = new_model.updated_at

        mock_async_session.refresh = fake_refresh

        dto = CreateCollectionDTO(name="new-col")
        result = await repo.create(dto)
        assert result.name == "new-col"


# ---------------------------------------------------------------------------
# Document Repository Tests
# ---------------------------------------------------------------------------


class TestDocumentRepository:
    """Tests for DocumentRepository."""

    @pytest.fixture()
    def repo(self, mock_async_session: AsyncMock) -> DocumentRepository:
        """Provide a DocumentRepository with a mocked session."""
        return DocumentRepository(mock_async_session)

    async def test_find_by_collection(
        self, repo: DocumentRepository, mock_async_session: AsyncMock
    ) -> None:
        """find_by_collection returns documents for a collection."""
        col_id = uuid.uuid4()
        models = [_make_document_model(collection_id=col_id) for _ in range(2)]

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = models
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        mock_async_session.execute.return_value = result_mock

        results = await repo.find_by_collection(col_id)
        assert len(results) == 2

    async def test_find_by_status(
        self, repo: DocumentRepository, mock_async_session: AsyncMock
    ) -> None:
        """find_by_status filters by DocumentStatus."""
        models = [_make_document_model(status="indexed")]

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = models
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        mock_async_session.execute.return_value = result_mock

        results = await repo.find_by_status(DocumentStatus.INDEXED)
        assert len(results) == 1

    async def test_update_status_not_found(
        self, repo: DocumentRepository, mock_async_session: AsyncMock
    ) -> None:
        """update_status raises NotFoundError for missing document."""
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_async_session.execute.return_value = result_mock

        with pytest.raises(NotFoundError):
            await repo.update_status(uuid.uuid4(), DocumentStatus.INDEXED)


# ---------------------------------------------------------------------------
# Chunk Repository Tests
# ---------------------------------------------------------------------------


class TestChunkRepository:
    """Tests for ChunkRepository."""

    @pytest.fixture()
    def repo(self, mock_async_session: AsyncMock) -> ChunkRepository:
        """Provide a ChunkRepository with a mocked session."""
        return ChunkRepository(mock_async_session)

    async def test_find_by_document(
        self, repo: ChunkRepository, mock_async_session: AsyncMock
    ) -> None:
        """find_by_document returns ordered chunks."""
        doc_id = uuid.uuid4()
        models = [_make_chunk_model(document_id=doc_id) for _ in range(3)]

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = models
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        mock_async_session.execute.return_value = result_mock

        results = await repo.find_by_document(doc_id)
        assert len(results) == 3

    async def test_bulk_create(
        self,
        repo: ChunkRepository,
        mock_async_session: AsyncMock,
        sample_chunks: list[CreateChunkDTO],
    ) -> None:
        """bulk_create inserts multiple chunks."""
        # After flush, the batch re-query returns refreshed models
        refreshed_models = [_make_chunk_model() for _ in sample_chunks]
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = refreshed_models
        requery_result = MagicMock()
        requery_result.scalars.return_value = scalars_mock

        # First execute is the re-query after flush
        mock_async_session.execute.return_value = requery_result

        results = await repo.bulk_create(sample_chunks)
        assert len(results) == 3
        mock_async_session.add_all.assert_called_once()

    async def test_delete_by_document(
        self, repo: ChunkRepository, mock_async_session: AsyncMock
    ) -> None:
        """delete_by_document executes delete statement."""
        doc_id = uuid.uuid4()
        await repo.delete_by_document(doc_id)
        mock_async_session.execute.assert_called_once()
        mock_async_session.flush.assert_called_once()


# ---------------------------------------------------------------------------
# Embedding Repository Tests
# ---------------------------------------------------------------------------


class TestEmbeddingRepository:
    """Tests for EmbeddingRepository."""

    @pytest.fixture()
    def repo(self, mock_async_session: AsyncMock) -> EmbeddingRepository:
        """Provide an EmbeddingRepository with a mocked session."""
        return EmbeddingRepository(mock_async_session)

    async def test_find_by_chunk_found(
        self, repo: EmbeddingRepository, mock_async_session: AsyncMock
    ) -> None:
        """find_by_chunk returns an Embedding when found."""
        chunk_id = uuid.uuid4()
        model = _make_embedding_model(chunk_id=chunk_id)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = model
        mock_async_session.execute.return_value = result_mock

        result = await repo.find_by_chunk(chunk_id)
        assert result is not None
        assert result.chunk_id == chunk_id

    async def test_find_by_chunk_not_found(
        self, repo: EmbeddingRepository, mock_async_session: AsyncMock
    ) -> None:
        """find_by_chunk returns None when no match."""
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_async_session.execute.return_value = result_mock

        result = await repo.find_by_chunk(uuid.uuid4())
        assert result is None

    async def test_bulk_create(
        self, repo: EmbeddingRepository, mock_async_session: AsyncMock
    ) -> None:
        """bulk_create inserts multiple embeddings."""
        dtos = [
            CreateEmbeddingDTO(
                chunk_id=uuid.uuid4(),
                model_name="voyage-3",
                dimensions=4,
                vector=[0.1, 0.2, 0.3, 0.4],
            )
            for _ in range(2)
        ]

        # After flush, the batch re-query returns refreshed models
        refreshed_models = [_make_embedding_model() for _ in dtos]
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = refreshed_models
        requery_result = MagicMock()
        requery_result.scalars.return_value = scalars_mock
        mock_async_session.execute.return_value = requery_result

        results = await repo.bulk_create(dtos)
        assert len(results) == 2

    async def test_delete_by_document(
        self, repo: EmbeddingRepository, mock_async_session: AsyncMock
    ) -> None:
        """delete_by_document removes embeddings via subquery."""
        doc_id = uuid.uuid4()
        await repo.delete_by_document(doc_id)
        mock_async_session.execute.assert_called_once()
        mock_async_session.flush.assert_called_once()

    async def test_bulk_create_dimension_mismatch_raises(
        self, repo: EmbeddingRepository, mock_async_session: AsyncMock
    ) -> None:
        """bulk_create raises ValueError when vector length mismatches dimensions."""
        dtos = [
            CreateEmbeddingDTO(
                chunk_id=uuid.uuid4(),
                model_name="voyage-3",
                dimensions=4,
                vector=[0.1, 0.2, 0.3],  # Only 3 values, dimensions says 4
            )
        ]
        with pytest.raises(ValueError, match="does not match"):
            await repo.bulk_create(dtos)


# ---------------------------------------------------------------------------
# Query Log Repository Tests
# ---------------------------------------------------------------------------


def _make_query_log_model(
    collection_id: uuid.UUID | None = None,
) -> QueryLogModel:
    """Create a mock QueryLogModel instance."""
    model = MagicMock(spec=QueryLogModel)
    model.id = uuid.uuid4()
    model.collection_id = collection_id or uuid.uuid4()
    model.query_text = "test query"
    model.retrieved_chunk_ids = None
    model.generated_response = "answer"
    model.latency_ms = 42.5
    model.created_at = NOW
    return model


class TestQueryLogRepository:
    """Tests for QueryLogRepository."""

    @pytest.fixture()
    def repo(self, mock_async_session: AsyncMock) -> QueryLogRepository:
        """Provide a QueryLogRepository with a mocked session."""
        return QueryLogRepository(mock_async_session)

    async def test_create(
        self, repo: QueryLogRepository, mock_async_session: AsyncMock
    ) -> None:
        """create inserts a query log record."""
        model = _make_query_log_model()

        async def fake_refresh(inst: QueryLogModel) -> None:
            inst.id = model.id
            inst.collection_id = model.collection_id
            inst.query_text = model.query_text
            inst.retrieved_chunk_ids = model.retrieved_chunk_ids
            inst.generated_response = model.generated_response
            inst.latency_ms = model.latency_ms
            inst.created_at = model.created_at

        mock_async_session.refresh = fake_refresh

        dto = CreateQueryLogDTO(
            collection_id=model.collection_id,
            query_text="test query",
            latency_ms=42.5,
        )
        result = await repo.create(dto)
        assert result.query_text == "test query"
        assert result.latency_ms == 42.5

    async def test_find_by_collection(
        self, repo: QueryLogRepository, mock_async_session: AsyncMock
    ) -> None:
        """find_by_collection returns logs for a collection."""
        col_id = uuid.uuid4()
        models = [_make_query_log_model(collection_id=col_id) for _ in range(3)]

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = models
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        mock_async_session.execute.return_value = result_mock

        results = await repo.find_by_collection(col_id)
        assert len(results) == 3

"""Unit tests for VectorForge domain models, DTOs, and enums."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vectorforge.models.domain import (
    Chunk,
    Collection,
    CreateChunkDTO,
    CreateCollectionDTO,
    CreateDocumentDTO,
    CreateEmbeddingDTO,
    CreateQueryLogDTO,
    DistanceMetric,
    Document,
    DocumentStatus,
    Embedding,
    QueryLog,
    QueryResult,
    RetrievedChunk,
    UpdateCollectionDTO,
)


class TestEnums:
    """Tests for DocumentStatus and DistanceMetric enums."""

    def test_document_status_values(self) -> None:
        """DocumentStatus contains all expected members."""
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.PROCESSING == "processing"
        assert DocumentStatus.INDEXED == "indexed"
        assert DocumentStatus.ERROR == "error"
        assert DocumentStatus.DELETED == "deleted"

    def test_distance_metric_values(self) -> None:
        """DistanceMetric contains all expected members."""
        assert DistanceMetric.COSINE == "cosine"
        assert DistanceMetric.L2 == "l2"
        assert DistanceMetric.INNER_PRODUCT == "inner_product"


class TestCollection:
    """Tests for the Collection domain model."""

    def test_creation(self) -> None:
        """Collection can be created with required fields."""
        now = datetime.now(UTC)
        col = Collection(
            id=uuid.uuid4(),
            name="test",
            description="A test collection",
            created_at=now,
        )
        assert col.name == "test"
        assert col.embedding_config is None
        assert col.updated_at is None

    def test_serialization_roundtrip(self) -> None:
        """Collection survives serialization/deserialization."""
        now = datetime.now(UTC)
        original = Collection(
            id=uuid.uuid4(),
            name="roundtrip",
            description="test",
            embedding_config={"provider": "voyage"},
            created_at=now,
            updated_at=now,
        )
        data = original.model_dump()
        restored = Collection.model_validate(data)
        assert original == restored


class TestDocument:
    """Tests for the Document domain model."""

    def test_creation_with_defaults(self) -> None:
        """Document defaults are applied correctly."""
        now = datetime.now(UTC)
        doc = Document(
            id=uuid.uuid4(),
            collection_id=uuid.uuid4(),
            source_uri="file:///test.txt",
            content_type="text/plain",
            created_at=now,
        )
        assert doc.status == DocumentStatus.PENDING
        assert doc.storage_backend == "pg"
        assert doc.metadata == {}
        assert doc.content_size_bytes == 0


class TestChunk:
    """Tests for the Chunk domain model."""

    def test_creation(self) -> None:
        """Chunk is created with all fields."""
        now = datetime.now(UTC)
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="Hello world",
            index=0,
            start_char=0,
            end_char=11,
            created_at=now,
        )
        assert chunk.text == "Hello world"
        assert chunk.index == 0


class TestEmbedding:
    """Tests for the Embedding domain model."""

    def test_creation(self) -> None:
        """Embedding is created with a vector."""
        now = datetime.now(UTC)
        emb = Embedding(
            id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            model_name="voyage-3",
            dimensions=4,
            vector=[0.1, 0.2, 0.3, 0.4],
            created_at=now,
        )
        assert len(emb.vector) == 4
        assert emb.model_name == "voyage-3"


class TestQueryResult:
    """Tests for QueryResult and RetrievedChunk."""

    def test_empty_result(self) -> None:
        """QueryResult can be created with no chunks."""
        result = QueryResult(query="test query")
        assert result.chunks == []
        assert result.generated_answer is None
        assert result.latency_ms == 0.0

    def test_with_chunks(self) -> None:
        """QueryResult holds retrieved chunks with scores."""
        now = datetime.now(UTC)
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="relevant text",
            index=0,
            start_char=0,
            end_char=13,
            created_at=now,
        )
        retrieved = RetrievedChunk(chunk=chunk, score=0.95, document_source="test.txt")
        result = QueryResult(
            query="find relevant text",
            chunks=[retrieved],
            generated_answer="Here is the answer.",
            latency_ms=42.5,
        )
        assert len(result.chunks) == 1
        assert result.chunks[0].score == 0.95


class TestCreateCollectionDTO:
    """Tests for CreateCollectionDTO."""

    def test_minimal_creation(self) -> None:
        """DTO can be created with just a name."""
        dto = CreateCollectionDTO(name="my-collection")
        assert dto.name == "my-collection"
        assert dto.metric == DistanceMetric.COSINE
        assert dto.description == ""

    def test_missing_name_raises(self) -> None:
        """DTO without name raises ValidationError."""
        with pytest.raises(ValidationError):
            CreateCollectionDTO()  # type: ignore[call-arg]


class TestCreateDocumentDTO:
    """Tests for CreateDocumentDTO."""

    def test_creation(self) -> None:
        """DTO is created with required fields."""
        dto = CreateDocumentDTO(
            collection_id=uuid.uuid4(),
            source_uri="s3://bucket/key",
            content_type="application/pdf",
            content="document text content",
        )
        assert dto.content_type == "application/pdf"
        assert dto.metadata == {}

    def test_missing_required_raises(self) -> None:
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            CreateDocumentDTO(collection_id=uuid.uuid4())  # type: ignore[call-arg]


class TestCreateChunkDTO:
    """Tests for CreateChunkDTO."""

    def test_creation(self, sample_chunks: list[CreateChunkDTO]) -> None:
        """DTOs from fixture are valid."""
        assert len(sample_chunks) == 3
        assert sample_chunks[0].index == 0
        assert sample_chunks[2].index == 2


class TestCreateEmbeddingDTO:
    """Tests for CreateEmbeddingDTO."""

    def test_creation(self, sample_embedding_dto: CreateEmbeddingDTO) -> None:
        """DTO from fixture is valid."""
        assert sample_embedding_dto.dimensions == 4
        assert len(sample_embedding_dto.vector) == 4


class TestUpdateCollectionDTO:
    """Tests for UpdateCollectionDTO."""

    def test_partial_update(self) -> None:
        """Only set fields are present in model_dump(exclude_unset=True)."""
        dto = UpdateCollectionDTO(description="updated")
        dumped = dto.model_dump(exclude_unset=True)
        assert "description" in dumped
        assert "embedding_config" not in dumped

    def test_empty_update(self) -> None:
        """No fields set results in empty dump."""
        dto = UpdateCollectionDTO()
        dumped = dto.model_dump(exclude_unset=True)
        assert dumped == {}


class TestQueryLog:
    """Tests for the QueryLog domain model."""

    def test_creation(self) -> None:
        """QueryLog can be created with all fields."""
        now = datetime.now(UTC)
        log = QueryLog(
            id=uuid.uuid4(),
            collection_id=uuid.uuid4(),
            query_text="search query",
            latency_ms=50.0,
            created_at=now,
        )
        assert log.query_text == "search query"
        assert log.retrieved_chunk_ids is None
        assert log.generated_response is None


class TestCreateQueryLogDTO:
    """Tests for CreateQueryLogDTO."""

    def test_creation(self) -> None:
        """DTO is created with required fields."""
        dto = CreateQueryLogDTO(
            collection_id=uuid.uuid4(),
            query_text="test query",
            latency_ms=42.5,
        )
        assert dto.query_text == "test query"
        assert dto.retrieved_chunk_ids is None

    def test_missing_required_raises(self) -> None:
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            CreateQueryLogDTO(collection_id=uuid.uuid4())  # type: ignore[call-arg]

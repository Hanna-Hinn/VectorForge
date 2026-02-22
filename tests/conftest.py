"""Shared test fixtures for VectorForge test suite."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vectorforge.config.settings import (
    ChunkingConfig,
    DatabaseConfig,
    EmbeddingConfig,
    LLMConfig,
    MonitoringConfig,
    StorageConfig,
    VectorForgeConfig,
)
from vectorforge.models.domain import (
    CreateChunkDTO,
    CreateCollectionDTO,
    CreateDocumentDTO,
    CreateEmbeddingDTO,
    DistanceMetric,
)
from vectorforge.monitoring.metrics import reset_metrics_collector

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_config(monkeypatch: pytest.MonkeyPatch) -> VectorForgeConfig:
    """Provide a VectorForgeConfig with test-specific values."""
    monkeypatch.setenv("VECTORFORGE_DB_HOST", "localhost")
    monkeypatch.setenv("VECTORFORGE_DB_PORT", "5432")
    monkeypatch.setenv("VECTORFORGE_DB_DATABASE", "vectorforge_test")
    monkeypatch.setenv("VECTORFORGE_DB_USER", "test_user")
    monkeypatch.setenv("VECTORFORGE_DB_PASSWORD", "test_password")
    return VectorForgeConfig(
        database=DatabaseConfig(),
        storage=StorageConfig(),
        embedding=EmbeddingConfig(),
        chunking=ChunkingConfig(),
        llm=LLMConfig(),
        monitoring=MonitoringConfig(),
    )


# ---------------------------------------------------------------------------
# Domain model fixtures
# ---------------------------------------------------------------------------

SAMPLE_COLLECTION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SAMPLE_DOCUMENT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
SAMPLE_CHUNK_IDS = [
    uuid.UUID("33333333-3333-3333-3333-333333333331"),
    uuid.UUID("33333333-3333-3333-3333-333333333332"),
    uuid.UUID("33333333-3333-3333-3333-333333333333"),
]


@pytest.fixture()
def sample_collection() -> CreateCollectionDTO:
    """Provide a CreateCollectionDTO for testing."""
    return CreateCollectionDTO(
        name="test-collection",
        description="Test collection for unit tests",
        metric=DistanceMetric.COSINE,
    )


@pytest.fixture()
def sample_document() -> CreateDocumentDTO:
    """Provide a CreateDocumentDTO for testing."""
    return CreateDocumentDTO(
        collection_id=SAMPLE_COLLECTION_ID,
        source_uri="file:///test/sample.txt",
        content_type="text/plain",
        content="Sample document content for testing.",
        metadata={"author": "test"},
    )


@pytest.fixture()
def sample_chunks() -> list[CreateChunkDTO]:
    """Provide a list of CreateChunkDTO for testing."""
    return [
        CreateChunkDTO(
            document_id=SAMPLE_DOCUMENT_ID,
            text="First chunk of the document.",
            index=0,
            start_char=0,
            end_char=28,
            metadata={"position": "start"},
        ),
        CreateChunkDTO(
            document_id=SAMPLE_DOCUMENT_ID,
            text="Second chunk continues here.",
            index=1,
            start_char=28,
            end_char=56,
        ),
        CreateChunkDTO(
            document_id=SAMPLE_DOCUMENT_ID,
            text="Third and final chunk.",
            index=2,
            start_char=56,
            end_char=78,
        ),
    ]


@pytest.fixture()
def sample_embedding_dto() -> CreateEmbeddingDTO:
    """Provide a CreateEmbeddingDTO for testing."""
    return CreateEmbeddingDTO(
        chunk_id=SAMPLE_CHUNK_IDS[0],
        model_name="voyage-3",
        dimensions=4,
        vector=[0.1, 0.2, 0.3, 0.4],
    )


# ---------------------------------------------------------------------------
# Mock database session
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_async_session() -> AsyncMock:
    """Provide a mocked AsyncSession for repository tests."""
    session = AsyncMock()

    # Mock execute to return a result proxy
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    mock_result.scalar_one = MagicMock(return_value=0)
    session.execute = AsyncMock(return_value=mock_result)

    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.delete = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()

    return session


# ---------------------------------------------------------------------------
# Cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_metrics() -> Any:
    """Reset the metrics collector before each test."""
    reset_metrics_collector()
    yield
    reset_metrics_collector()

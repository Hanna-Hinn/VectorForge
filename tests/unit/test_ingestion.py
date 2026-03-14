"""Unit tests for VectorForge ingestion service (all deps mocked)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vectorforge.chunking.base import BaseChunker
from vectorforge.chunking.registry import ChunkerRegistry
from vectorforge.config.settings import ChunkingConfig
from vectorforge.ingestion.loaders.base import DocumentLoaderRegistry
from vectorforge.ingestion.loaders.text_loader import TextLoader
from vectorforge.ingestion.service import IngestionService
from vectorforge.models.domain import Chunk, Document, DocumentStatus

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "sample_documents"
NOW = datetime.now(UTC)
COLLECTION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _make_fake_embedding_provider() -> MagicMock:
    """Create a mock embedding provider."""
    provider = MagicMock()
    provider.provider_name.return_value = "fake"
    provider.model_name.return_value = "fake-model"
    provider.dimensions.return_value = 4
    provider.max_batch_size.return_value = 128
    provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    return provider


def _make_fake_vector_store() -> MagicMock:
    """Create a mock vector store."""
    store = MagicMock()
    store.upsert = AsyncMock()
    return store


def _make_fake_storage_router() -> MagicMock:
    """Create a mock storage router."""
    router = MagicMock()
    router.store = AsyncMock(return_value=("storage-key", "pg"))
    return router


class TestIngestionService:
    """Tests for the ingestion orchestrator."""

    @pytest.mark.asyncio
    async def test_ingest_full_pipeline(self) -> None:
        """Full pipeline: load → store → persist doc → chunk → embed → upsert."""
        # Arrange
        loader_registry = DocumentLoaderRegistry(loaders=[TextLoader()])
        chunker = MagicMock(spec=BaseChunker)
        chunker.strategy_name.return_value = "recursive"
        chunker.chunk.return_value = [
            Chunk(
                id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="Sample chunk",
                index=0,
                start_char=0,
                end_char=12,
                metadata={},
                created_at=NOW,
            )
        ]

        chunker_registry = ChunkerRegistry()
        chunker_registry.register(chunker)
        chunker_registry.set_default("recursive")

        embedder = _make_fake_embedding_provider()
        vector_store = _make_fake_vector_store()
        storage_router = _make_fake_storage_router()

        service = IngestionService(
            loader_registry=loader_registry,
            chunker_registry=chunker_registry,
            embedding_provider=embedder,
            vector_store=vector_store,
            storage_router=storage_router,
            chunking_config=ChunkingConfig(chunk_size=500, chunk_overlap=50),
        )

        # Mock session and repositories
        session = AsyncMock()

        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        mock_doc = Document(
            id=doc_id,
            collection_id=COLLECTION_ID,
            source_uri="file:///test/sample.txt",
            content_type="text/plain",
            raw_content="content",
            status=DocumentStatus.PENDING,
            created_at=NOW,
        )
        mock_indexed_doc = Document(
            id=doc_id,
            collection_id=COLLECTION_ID,
            source_uri="file:///test/sample.txt",
            content_type="text/plain",
            raw_content="content",
            status=DocumentStatus.INDEXED,
            created_at=NOW,
        )
        mock_chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            text="Sample chunk",
            index=0,
            start_char=0,
            end_char=12,
            metadata={},
            created_at=NOW,
        )

        with (
            patch(
                "vectorforge.ingestion.service.DocumentRepository"
            ) as MockDocRepo,
            patch(
                "vectorforge.ingestion.service.ChunkRepository"
            ) as MockChunkRepo,
            patch(
                "vectorforge.ingestion.service.EmbeddingRepository"
            ) as MockEmbeddingRepo,
        ):
            doc_repo = MockDocRepo.return_value
            doc_repo.create = AsyncMock(return_value=mock_doc)
            doc_repo.update_status = AsyncMock()
            doc_repo.find_by_id = AsyncMock(return_value=mock_indexed_doc)

            chunk_repo = MockChunkRepo.return_value
            chunk_repo.bulk_create = AsyncMock(return_value=[mock_chunk])

            embedding_repo = MockEmbeddingRepo.return_value
            embedding_repo.bulk_create = AsyncMock(return_value=[])

            # Act
            result = await service.ingest(
                source=str(FIXTURES_DIR / "sample.txt"),
                collection_id=COLLECTION_ID,
                session=session,
            )

        # Assert
        assert result.status == DocumentStatus.INDEXED
        doc_repo.create.assert_called_once()
        chunk_repo.bulk_create.assert_called_once()
        embedder.embed.assert_called_once()
        vector_store.upsert.assert_called_once()
        storage_router.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_with_custom_metadata(self) -> None:
        """Extra metadata is merged into the document."""
        loader_registry = DocumentLoaderRegistry(loaders=[TextLoader()])
        chunker = MagicMock(spec=BaseChunker)
        chunker.strategy_name.return_value = "recursive"
        chunker.chunk.return_value = []

        chunker_registry = ChunkerRegistry()
        chunker_registry.register(chunker)
        chunker_registry.set_default("recursive")

        embedder = _make_fake_embedding_provider()
        embedder.embed = AsyncMock(return_value=[])
        vector_store = _make_fake_vector_store()
        storage_router = _make_fake_storage_router()

        service = IngestionService(
            loader_registry=loader_registry,
            chunker_registry=chunker_registry,
            embedding_provider=embedder,
            vector_store=vector_store,
            storage_router=storage_router,
        )

        session = AsyncMock()
        doc_id = uuid.uuid4()
        mock_doc = Document(
            id=doc_id,
            collection_id=COLLECTION_ID,
            source_uri="test",
            content_type="text/plain",
            status=DocumentStatus.PENDING,
            created_at=NOW,
        )

        with (
            patch(
                "vectorforge.ingestion.service.DocumentRepository"
            ) as MockDocRepo,
            patch(
                "vectorforge.ingestion.service.ChunkRepository"
            ) as MockChunkRepo,
            patch(
                "vectorforge.ingestion.service.EmbeddingRepository"
            ) as MockEmbeddingRepo,
        ):
            doc_repo = MockDocRepo.return_value
            doc_repo.create = AsyncMock(return_value=mock_doc)
            doc_repo.update_status = AsyncMock()
            doc_repo.find_by_id = AsyncMock(return_value=mock_doc)

            chunk_repo = MockChunkRepo.return_value
            chunk_repo.bulk_create = AsyncMock(return_value=[])

            embedding_repo = MockEmbeddingRepo.return_value
            embedding_repo.bulk_create = AsyncMock(return_value=[])

            await service.ingest(
                source=str(FIXTURES_DIR / "sample.txt"),
                collection_id=COLLECTION_ID,
                session=session,
                metadata={"custom_key": "custom_value"},
            )

            # Verify the create call included the custom metadata
            call_args = doc_repo.create.call_args[0][0]
            assert "custom_key" in call_args.metadata

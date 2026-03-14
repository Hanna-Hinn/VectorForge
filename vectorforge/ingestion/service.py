"""Ingestion orchestrator — coordinates the full document ingest pipeline.

Pipeline flow:
  1. Load document (via DocumentLoaderRegistry)
  2. Store raw content (via StorageRouter)
  3. Persist document record (via DocumentRepository)
  4. Chunk the text (via ChunkerRegistry)
  5. Persist chunk records (via ChunkRepository)
  6. Embed chunks (via EmbeddingProvider)
  7. Upsert embeddings (via VectorStore)
  8. Update document status to INDEXED
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from vectorforge.chunking.registry import ChunkerRegistry
from vectorforge.config.settings import ChunkingConfig
from vectorforge.db.repositories.chunk_repo import ChunkRepository
from vectorforge.db.repositories.document_repo import DocumentRepository
from vectorforge.db.repositories.embedding_repo import EmbeddingRepository
from vectorforge.embedding.base import BaseEmbeddingProvider
from vectorforge.exceptions import VectorForgeError
from vectorforge.ingestion.loaders.base import DocumentLoaderRegistry
from vectorforge.models.domain import (
    CreateChunkDTO,
    CreateDocumentDTO,
    CreateEmbeddingDTO,
    Document,
    DocumentStatus,
)
from vectorforge.storage.router import StorageRouter
from vectorforge.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    """Orchestrates the full document ingestion pipeline.

    All database operations share the caller-provided session so that
    the entire ingest is wrapped in a single transaction.

    Args:
        loader_registry: Registry of document loaders.
        chunker_registry: Registry of chunking strategies.
        embedding_provider: The embedding provider to use.
        vector_store: The vector store for upserting embeddings.
        storage_router: Router for raw content storage.
        chunking_config: Default chunking configuration.
    """

    def __init__(
        self,
        loader_registry: DocumentLoaderRegistry,
        chunker_registry: ChunkerRegistry,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        storage_router: StorageRouter,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self._loaders = loader_registry
        self._chunkers = chunker_registry
        self._embedder = embedding_provider
        self._vector_store = vector_store
        self._storage = storage_router
        self._chunking_config = chunking_config or ChunkingConfig()

    async def ingest(
        self,
        source: str,
        collection_id: uuid.UUID,
        session: AsyncSession,
        metadata: dict[str, object] | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> Document:
        """Ingest a document through the full pipeline.

        Args:
            source: File path to ingest.
            collection_id: Target collection UUID.
            session: Active async database session (caller manages transaction).
            metadata: Optional extra metadata to attach.
            chunking_config: Override chunking config for this ingest.

        Returns:
            The persisted Document domain model with INDEXED status.

        Raises:
            VectorForgeError: If any pipeline stage fails.
        """
        config = chunking_config or self._chunking_config
        doc_id: uuid.UUID | None = None

        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)
        embedding_repo = EmbeddingRepository(session)

        try:
            # 1. Load document
            loaded = self._loaders.load(source)
            logger.info("Loaded document from %s (%s)", source, loaded.content_type)

            # 2. Store raw content
            raw_bytes = (loaded.raw_content or "").encode("utf-8")
            _storage_key, _backend_name = await self._storage.store(
                str(uuid.uuid4()), raw_bytes
            )

            # 3. Persist document record
            extra_meta = metadata or {}
            combined_metadata = {**loaded.metadata, **extra_meta}

            doc = await doc_repo.create(
                CreateDocumentDTO(
                    collection_id=collection_id,
                    source_uri=loaded.source_uri,
                    content_type=loaded.content_type,
                    content=loaded.raw_content or "",
                    metadata=combined_metadata,
                )
            )
            doc_id = doc.id
            logger.info("Created document record %s", doc.id)

            # 4. Update status to PROCESSING
            await doc_repo.update_status(doc.id, DocumentStatus.PROCESSING)

            # 5. Chunk the text
            chunker = self._chunkers.get_for_content_type(loaded.content_type)
            domain_chunks = chunker.chunk(loaded.raw_content or "", config)
            logger.info("Split into %d chunks", len(domain_chunks))

            # 6. Persist chunk records
            chunk_dtos = [
                CreateChunkDTO(
                    document_id=doc.id,
                    text=c.text,
                    index=c.index,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    metadata=c.metadata,
                )
                for c in domain_chunks
            ]
            persisted_chunks = await chunk_repo.bulk_create(chunk_dtos)
            logger.info("Persisted %d chunks", len(persisted_chunks))

            # 7. Embed chunks
            chunk_texts = [c.text for c in persisted_chunks]
            embeddings = await self._embedder.embed(chunk_texts)

            # 8. Persist embedding records
            embedding_dtos = [
                CreateEmbeddingDTO(
                    chunk_id=chunk.id,
                    model_name=self._embedder.model_name(),
                    dimensions=self._embedder.dimensions(),
                    vector=vec,
                )
                for chunk, vec in zip(persisted_chunks, embeddings, strict=True)
            ]
            await embedding_repo.bulk_create(embedding_dtos)

            # 9. Upsert into vector store
            chunk_ids = [c.id for c in persisted_chunks]
            await self._vector_store.upsert(
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                model_name=self._embedder.model_name(),
            )
            logger.info("Upserted %d embeddings into vector store", len(embeddings))

            # 10. Update status to INDEXED
            await doc_repo.update_status(doc.id, DocumentStatus.INDEXED)
            logger.info("Document %s ingestion complete (INDEXED)", doc.id)

            # Return the final document state
            final_doc = await doc_repo.find_by_id(doc.id)
            return final_doc if final_doc is not None else doc

        except VectorForgeError:
            await self._rollback_status(doc_repo, doc_id)
            raise
        except Exception as exc:
            await self._rollback_status(doc_repo, doc_id)
            msg = f"Ingestion failed for {source}: {exc}"
            logger.error(msg, exc_info=True)
            raise VectorForgeError(msg) from exc

    @staticmethod
    async def _rollback_status(
        doc_repo: DocumentRepository,
        doc_id: uuid.UUID | None,
    ) -> None:
        """Best-effort rollback of document status to ERROR."""
        if doc_id is None:
            return
        try:
            await doc_repo.update_status(doc_id, DocumentStatus.ERROR)
        except Exception:
            logger.warning("Failed to rollback document %s to ERROR", doc_id, exc_info=True)

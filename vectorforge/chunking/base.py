"""Base chunker ABC for VectorForge chunking strategies.

All chunkers wrap LangChain text splitters and produce
domain ``Chunk`` models.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from uuid import uuid4

from vectorforge.config.settings import ChunkingConfig
from vectorforge.models.domain import Chunk

logger = logging.getLogger(__name__)


def _sub_chunk_sections(
    sections: list[tuple[str, dict[str, object]]],
    config: ChunkingConfig,
) -> list[tuple[str, dict[str, object]]]:
    """Sub-chunk sections exceeding *chunk_size* with recursive splitting.

    Sections within the limit pass through unchanged.  Oversized sections
    are split while propagating their metadata to every sub-chunk.

    Args:
        sections: List of (text, metadata) tuples.
        config: Chunking configuration with size and overlap.

    Returns:
        Flat list of (text, metadata) tuples, each within chunk_size.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        strip_whitespace=True,
    )

    results: list[tuple[str, dict[str, object]]] = []
    for text, metadata in sections:
        if len(text) > config.chunk_size:
            sub_docs = sub_splitter.create_documents([text], metadatas=[metadata])
            results.extend((sd.page_content, dict(sd.metadata)) for sd in sub_docs)
        else:
            results.append((text, metadata))
    return results


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies.

    Subclasses implement ``_build_splitter`` to create the appropriate
    LangChain text splitter for their strategy.
    """

    @abstractmethod
    def strategy_name(self) -> str:
        """Return the unique strategy name."""

    @abstractmethod
    def _split_text(self, text: str, config: ChunkingConfig) -> list[tuple[str, dict[str, object]]]:
        """Split text into (content, metadata) tuples.

        Args:
            text: The full document text.
            config: Chunking configuration.

        Returns:
            List of (chunk_text, chunk_metadata) tuples.
        """

    def chunk(self, text: str, config: ChunkingConfig) -> list[Chunk]:
        """Split text into Chunk domain models.

        Args:
            text: The full document text.
            config: Chunking configuration with size, overlap, etc.

        Returns:
            List of Chunk domain models with assigned indices.
        """
        if not text:
            return []

        from datetime import UTC, datetime

        splits = self._split_text(text, config)
        chunks: list[Chunk] = []
        running_offset = 0
        for i, (content, metadata) in enumerate(splits):
            start_char = metadata.get("start_index", running_offset)
            if not isinstance(start_char, int):
                start_char = running_offset
            end_char = start_char + len(content)
            chunks.append(
                Chunk(
                    id=uuid4(),
                    document_id=uuid4(),  # placeholder
                    text=content,
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={k: v for k, v in metadata.items() if k != "start_index"},
                    created_at=datetime.now(UTC),
                )
            )
            running_offset = end_char

        logger.info(
            "Split into %d chunks (strategy=%s)",
            len(chunks),
            self.strategy_name(),
        )
        return chunks

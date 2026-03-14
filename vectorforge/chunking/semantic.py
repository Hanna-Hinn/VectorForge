"""Semantic chunker using embedding similarity breakpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorforge.chunking.base import BaseChunker
from vectorforge.config.settings import ChunkingConfig

if TYPE_CHECKING:
    from vectorforge.embedding.base import BaseEmbeddingProvider


class SemanticChunker(BaseChunker):
    """Chunker that uses embedding similarity to find natural breakpoints.

    Falls back to recursive splitting since LangChain's SemanticChunker
    requires a specific embeddings interface. This implementation splits
    recursively but is designed as the extension point for true semantic
    chunking when an embedding provider is wired in.

    Args:
        embedding_provider: An optional embedding provider for future
            semantic breakpoint detection.
    """

    def __init__(self, embedding_provider: BaseEmbeddingProvider | None = None) -> None:
        self._embedding_provider = embedding_provider

    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "semantic"

    def _split_text(
        self, text: str, config: ChunkingConfig
    ) -> list[tuple[str, dict[str, object]]]:
        """Split text using recursive character splitting as a baseline.

        Args:
            text: The full document text.
            config: Chunking configuration.

        Returns:
            List of (chunk_text, metadata) tuples.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
        )
        docs = splitter.create_documents([text])
        return [(doc.page_content, dict(doc.metadata)) for doc in docs]

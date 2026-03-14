"""Recursive character text splitter chunker."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorforge.chunking.base import BaseChunker
from vectorforge.config.settings import ChunkingConfig


class RecursiveChunker(BaseChunker):
    """Chunker using LangChain RecursiveCharacterTextSplitter.

    Splits text hierarchically using a list of separators,
    falling through to the next separator when chunks exceed size.
    """

    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "recursive"

    def _split_text(
        self, text: str, config: ChunkingConfig
    ) -> list[tuple[str, dict[str, object]]]:
        """Split text using recursive character splitting.

        Args:
            text: The full document text.
            config: Chunking configuration.

        Returns:
            List of (chunk_text, metadata) tuples.
        """
        separators = config.separators or ["\n\n", "\n", ". ", " ", ""]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=separators,
            add_start_index=True,
            strip_whitespace=True,
        )
        docs = splitter.create_documents([text])
        return [(doc.page_content, dict(doc.metadata)) for doc in docs]

"""Token-based text splitter chunker."""

from __future__ import annotations

from langchain_text_splitters import TokenTextSplitter

from vectorforge.chunking.base import BaseChunker
from vectorforge.config.settings import ChunkingConfig


class TokenChunker(BaseChunker):
    """Chunker using LangChain TokenTextSplitter.

    Splits text by token count using tiktoken encoding,
    guaranteeing each chunk fits within a model's context window.
    """

    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "token"

    def _split_text(
        self, text: str, config: ChunkingConfig
    ) -> list[tuple[str, dict[str, object]]]:
        """Split text by token count.

        Args:
            text: The full document text.
            config: Chunking configuration.

        Returns:
            List of (chunk_text, metadata) tuples.
        """
        model_name = config.model_name or "gpt-4"
        splitter = TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            model_name=model_name,
            add_start_index=True,
        )
        docs = splitter.create_documents([text])
        return [(doc.page_content, dict(doc.metadata)) for doc in docs]

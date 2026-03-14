"""HTML-aware chunker with two-pass splitting.

Pass 1: Splits on HTML heading boundaries for semantic sections.
Pass 2: Sub-chunks oversized sections with recursive character splitting.
"""

from __future__ import annotations

from typing import ClassVar

from langchain_text_splitters import HTMLHeaderTextSplitter

from vectorforge.chunking.base import BaseChunker, _sub_chunk_sections
from vectorforge.config.settings import ChunkingConfig


class HTMLChunker(BaseChunker):
    """Two-pass HTML chunker for semantically accurate splits.

    First splits on heading tags (h1-h6) to preserve document
    structure, then sub-chunks any oversized sections using recursive
    character splitting to honour chunk_size / chunk_overlap.
    """

    _HEADERS_TO_SPLIT_ON: ClassVar[list[tuple[str, str]]] = [
        ("h1", "h1"),
        ("h2", "h2"),
        ("h3", "h3"),
        ("h4", "h4"),
        ("h5", "h5"),
        ("h6", "h6"),
    ]

    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "html"

    def _split_text(
        self, text: str, config: ChunkingConfig
    ) -> list[tuple[str, dict[str, object]]]:
        """Split HTML by headings, then sub-chunk oversized sections.

        Args:
            text: The full HTML text.
            config: Chunking configuration.

        Returns:
            List of (chunk_text, metadata) tuples with heading hierarchy.
        """
        header_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=self._HEADERS_TO_SPLIT_ON,
        )
        header_docs = header_splitter.split_text(text)
        sections = [(doc.page_content, dict(doc.metadata)) for doc in header_docs]
        return _sub_chunk_sections(sections, config)

"""XML-aware chunker with structure-preserving splitting.

Parses the XML tree and extracts text sections at the most granular
level that fits within chunk_size, preserving tag-path metadata.
Falls back to recursive character splitting for malformed XML.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from vectorforge.chunking.base import BaseChunker, _sub_chunk_sections
from vectorforge.config.settings import ChunkingConfig

logger = logging.getLogger(__name__)


def _local_name(tag: str) -> str:
    """Strip namespace URI from an XML tag, returning only the local name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


class XMLChunker(BaseChunker):
    """Structure-aware XML chunker.

    Walks the XML tree depth-first, emitting the deepest elements
    whose full text fits within ``chunk_size``.  Oversized leaf
    sections are further split with a recursive character splitter.
    Malformed XML falls back to plain recursive splitting.
    """

    def strategy_name(self) -> str:
        """Return the strategy name."""
        return "xml"

    def _split_text(
        self, text: str, config: ChunkingConfig
    ) -> list[tuple[str, dict[str, object]]]:
        """Split XML text by document structure.

        Args:
            text: The full XML text.
            config: Chunking configuration.

        Returns:
            List of (chunk_text, metadata) tuples with xml_path / xml_tag.
        """
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            logger.warning("Malformed XML — falling back to recursive splitting")
            return _sub_chunk_sections([(text, {})], config)

        sections = self._extract_sections(root, "", config.chunk_size)
        if not sections:
            return _sub_chunk_sections([(text, {})], config)

        return _sub_chunk_sections(sections, config)

    # ------------------------------------------------------------------

    def _extract_sections(
        self,
        element: ET.Element,
        parent_path: str,
        chunk_size: int,
    ) -> list[tuple[str, dict[str, object]]]:
        """Recursively extract text sections from the XML tree.

        Emits the deepest element whose full text fits within
        *chunk_size*.  If it does not fit, recurse into children.
        Direct text of a parent (``element.text``) and tail text
        after each child (``child.tail``) are captured to avoid
        losing interleaved content.

        Args:
            element: Current XML element.
            parent_path: Slash-separated ancestor tag path.
            chunk_size: Target maximum section size.

        Returns:
            List of (text, metadata) tuples.
        """
        tag = _local_name(element.tag)
        current_path = f"{parent_path}/{tag}" if parent_path else tag
        full_text = "".join(element.itertext()).strip()

        if not full_text:
            return []

        metadata: dict[str, object] = {"xml_path": current_path, "xml_tag": tag}

        # Fits in one chunk — emit directly
        if len(full_text) <= chunk_size:
            return [(full_text, metadata)]

        # Too large — try splitting into children + interleaved text
        if len(element) > 0:
            child_sections: list[tuple[str, dict[str, object]]] = []

            # Direct text before the first child
            if element.text and element.text.strip():
                child_sections.append((element.text.strip(), metadata))

            for child in element:
                child_sections.extend(
                    self._extract_sections(child, current_path, chunk_size)
                )
                # Tail text after child's closing tag
                if child.tail and child.tail.strip():
                    child_sections.append((child.tail.strip(), metadata))

            if child_sections:
                return child_sections

        # Leaf that exceeds chunk_size, or no usable children
        return [(full_text, metadata)]

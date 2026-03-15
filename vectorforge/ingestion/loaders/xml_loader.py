"""XML document loader using stdlib xml.etree.ElementTree."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from vectorforge.exceptions import DocumentLoadError
from vectorforge.ingestion.loaders.base import BaseDocumentLoader

logger = logging.getLogger(__name__)

_MULTI_BLANK = re.compile(r"\n{3,}")


def _local_name(tag: str) -> str:
    """Strip namespace URI from an XML tag, returning only the local name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


class XMLLoader(BaseDocumentLoader):
    """Loader for XML files (.xml).

    Parses the XML tree and concatenates all text content,
    preserving the reading order.  Returns ``application/xml``
    as content type so the downstream XMLChunker is selected
    automatically by the chunking registry.
    """

    def content_type(self) -> str:
        """Return the MIME type for XML."""
        return "application/xml"

    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return {".xml"}

    def _extract_text(self, raw_bytes: bytes) -> str:
        """Extract concatenated text from all XML elements.

        Args:
            raw_bytes: The raw XML content.

        Returns:
            Plain text extracted from XML elements.

        Raises:
            DocumentLoadError: If the XML cannot be parsed.
        """
        try:
            root = ET.fromstring(raw_bytes)
        except ET.ParseError as exc:
            msg = f"Failed to parse XML: {exc}"
            raise DocumentLoadError(msg) from exc

        text = "\n".join(
            segment.strip()
            for segment in root.itertext()
            if segment.strip()
        )
        return _MULTI_BLANK.sub("\n\n", text)

    def _extract_metadata(self, source: str, raw_bytes: bytes) -> dict[str, Any]:
        """Extract metadata including root element name and element count.

        Args:
            source: The source file path.
            raw_bytes: The raw XML content.

        Returns:
            Metadata dictionary with filename, root_tag, and element_count.
        """
        metadata: dict[str, Any] = {"filename": Path(source).name}
        try:
            root = ET.fromstring(raw_bytes)
            metadata["root_tag"] = _local_name(root.tag)
            metadata["element_count"] = sum(1 for _ in root.iter())
        except ET.ParseError:
            logger.warning("Could not extract XML metadata from %s", source)
        return metadata

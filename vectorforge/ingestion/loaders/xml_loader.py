"""XML document loader using stdlib xml.etree.ElementTree."""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from vectorforge.exceptions import DocumentLoadError
from vectorforge.ingestion.loaders.base import BaseDocumentLoader
from vectorforge.models.domain import Document, DocumentStatus

logger = logging.getLogger(__name__)


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
        """Validate the XML and return the original XML text.

        Unlike other loaders, the XML loader preserves the original markup
        so the downstream XMLChunker can parse the structure. Validation
        ensures we fail fast on malformed files.

        Args:
            raw_bytes: The raw XML content.

        Returns:
            The original XML as a UTF-8 string.

        Raises:
            DocumentLoadError: If the XML cannot be parsed.
        """
        try:
            root = ET.fromstring(raw_bytes)
        except ET.ParseError as exc:
            msg = f"Failed to parse XML: {exc}"
            raise DocumentLoadError(msg) from exc

        # Verify there is actual text content
        text_content = "".join(root.itertext()).strip()
        if not text_content:
            msg = "XML contains no text content"
            raise DocumentLoadError(msg)

        return raw_bytes.decode("utf-8", errors="replace")

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

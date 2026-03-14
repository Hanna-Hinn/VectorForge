"""Plain text document loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import charset_normalizer

from vectorforge.exceptions import DocumentLoadError
from vectorforge.ingestion.loaders.base import BaseDocumentLoader

logger = logging.getLogger(__name__)


def _detect_encoding(raw_bytes: bytes) -> str:
    """Detect encoding for raw bytes.

    Tries UTF-8, then Latin-1, then charset-normalizer.

    Args:
        raw_bytes: The raw file content.

    Returns:
        Detected encoding name.
    """
    try:
        raw_bytes.decode("utf-8")
        return "utf-8"
    except (UnicodeDecodeError, ValueError):
        pass

    try:
        raw_bytes.decode("latin-1")
        return "latin-1"
    except (UnicodeDecodeError, ValueError):
        pass

    result = charset_normalizer.from_bytes(raw_bytes).best()
    return result.encoding if result else "unknown"


class TextLoader(BaseDocumentLoader):
    """Loader for plain text files (.txt)."""

    def content_type(self) -> str:
        """Return the MIME type for plain text."""
        return "text/plain"

    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return {".txt"}

    def _extract_text(self, raw_bytes: bytes) -> str:
        """Decode text from raw bytes with encoding detection.

        Args:
            raw_bytes: The raw file content.

        Returns:
            Decoded text string.

        Raises:
            DocumentLoadError: If encoding cannot be detected.
        """
        encoding = _detect_encoding(raw_bytes)
        if encoding == "unknown":
            msg = "Cannot detect encoding for text file"
            raise DocumentLoadError(msg)
        return raw_bytes.decode(encoding)

    def _extract_metadata(self, source: str, raw_bytes: bytes) -> dict[str, Any]:
        """Extract metadata including detected encoding.

        Args:
            source: The source file path.
            raw_bytes: The raw file content.

        Returns:
            Metadata dictionary with filename and encoding.
        """
        return {
            "filename": Path(source).name,
            "encoding": _detect_encoding(raw_bytes),
        }

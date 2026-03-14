"""PDF document loader using pdfplumber."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pdfplumber

from vectorforge.exceptions import DocumentLoadError
from vectorforge.ingestion.loaders.base import BaseDocumentLoader

logger = logging.getLogger(__name__)


class PDFLoader(BaseDocumentLoader):
    """Loader for PDF files (.pdf) using pdfplumber."""

    def content_type(self) -> str:
        """Return the MIME type for PDF."""
        return "application/pdf"

    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return {".pdf"}

    def _extract_text(self, raw_bytes: bytes) -> str:
        """Extract text from all pages of a PDF.

        Args:
            raw_bytes: The raw PDF content.

        Returns:
            Concatenated text from all pages.

        Raises:
            DocumentLoadError: If the PDF cannot be parsed.
        """
        try:
            pages = self._extract_pages(raw_bytes)
        except Exception as exc:
            msg = f"Failed to parse PDF: {exc}"
            raise DocumentLoadError(msg) from exc
        return self._merge_pages(pages)

    def _extract_metadata(self, source: str, raw_bytes: bytes) -> dict[str, Any]:
        """Extract metadata from PDF including page count and info.

        Args:
            source: The source file path.
            raw_bytes: The raw PDF content.

        Returns:
            Metadata dictionary with filename, page count, title, author.
        """
        metadata: dict[str, Any] = {"filename": Path(source).name}
        try:
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                metadata["page_count"] = len(pdf.pages)
                info = pdf.metadata or {}
                metadata["title"] = info.get("Title", "")
                metadata["author"] = info.get("Author", "")
        except Exception:
            logger.warning("Could not extract PDF metadata from %s", source)
        return metadata

    @staticmethod
    def _extract_pages(raw_bytes: bytes) -> list[str]:
        """Extract text from each page of a PDF.

        Args:
            raw_bytes: The raw PDF content.

        Returns:
            List of page text strings (non-empty pages only).
        """
        pages: list[str] = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return pages

    @staticmethod
    def _merge_pages(pages: list[str]) -> str:
        """Merge page texts with double newlines.

        Args:
            pages: List of page text strings.

        Returns:
            Concatenated text with double newline separators.
        """
        return "\n\n".join(pages)

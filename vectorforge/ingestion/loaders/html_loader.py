"""HTML document loader using BeautifulSoup."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from bs4 import BeautifulSoup

from vectorforge.ingestion.loaders.base import BaseDocumentLoader

_MULTI_BLANK = re.compile(r"\n{3,}")


class HTMLLoader(BaseDocumentLoader):
    """Loader for HTML files (.html, .htm).

    Strips script/style/nav/footer/header tags and extracts
    visible text content.
    """

    _REMOVE_TAGS: ClassVar[set[str]] = {"script", "style", "nav", "footer", "header"}

    def content_type(self) -> str:
        """Return the MIME type for HTML."""
        return "text/html"

    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return {".html", ".htm"}

    def _extract_text(self, raw_bytes: bytes) -> str:
        """Extract visible text from HTML, removing boilerplate tags.

        Args:
            raw_bytes: The raw HTML content.

        Returns:
            Plain text extracted from the HTML.
        """
        html = raw_bytes.decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(self._REMOVE_TAGS):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return _MULTI_BLANK.sub("\n\n", text)

    def _extract_metadata(self, source: str, raw_bytes: bytes) -> dict[str, Any]:
        """Extract metadata including the HTML title.

        Args:
            source: The source file path.
            raw_bytes: The raw HTML content.

        Returns:
            Metadata dictionary with filename and title.
        """
        html = raw_bytes.decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.string if title_tag and title_tag.string else ""
        return {
            "filename": Path(source).name,
            "title": title,
        }

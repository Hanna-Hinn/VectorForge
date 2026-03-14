"""Markdown document loader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from vectorforge.ingestion.loaders.base import BaseDocumentLoader

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class MarkdownLoader(BaseDocumentLoader):
    """Loader for Markdown files (.md, .markdown).

    Strips optional YAML frontmatter but preserves Markdown structure
    so downstream chunkers (e.g. MarkdownChunker) can use headings.
    """

    def content_type(self) -> str:
        """Return the MIME type for Markdown."""
        return "text/markdown"

    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return {".md", ".markdown"}

    def _extract_text(self, raw_bytes: bytes) -> str:
        """Decode Markdown and strip frontmatter.

        Args:
            raw_bytes: The raw file content.

        Returns:
            Markdown text with frontmatter removed.
        """
        text = raw_bytes.decode("utf-8")
        return self._strip_frontmatter(text)

    def _extract_metadata(self, source: str, raw_bytes: bytes) -> dict[str, Any]:
        """Extract metadata including parsed frontmatter.

        Args:
            source: The source file path.
            raw_bytes: The raw file content.

        Returns:
            Metadata dictionary with filename and frontmatter dict.
        """
        text = raw_bytes.decode("utf-8")
        frontmatter = self._parse_frontmatter(text)
        return {
            "filename": Path(source).name,
            "frontmatter": frontmatter,
        }

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Remove YAML frontmatter block from the beginning of text.

        Args:
            text: The full Markdown text.

        Returns:
            Text with frontmatter removed.
        """
        return _FRONTMATTER_RE.sub("", text)

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, str]:
        """Parse simple key: value pairs from YAML frontmatter.

        Args:
            text: The full Markdown text.

        Returns:
            Dictionary of frontmatter key-value pairs (empty if none).
        """
        match = _FRONTMATTER_RE.match(text)
        if not match:
            return {}
        result: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result

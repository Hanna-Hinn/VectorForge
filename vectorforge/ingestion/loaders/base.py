"""Base document loader ABC and registry.

Defines the interface for all document loaders and a registry
that auto-selects the correct loader by file extension.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from vectorforge.exceptions import DocumentLoadError, UnsupportedFormatError
from vectorforge.models.domain import Document, DocumentStatus

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders.

    Subclasses implement ``_extract_text`` and ``_extract_metadata``
    for their specific file format.
    """

    @abstractmethod
    def content_type(self) -> str:
        """Return the MIME content type this loader handles."""

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return the set of file extensions this loader handles (with leading dot)."""

    @abstractmethod
    def _extract_text(self, raw_bytes: bytes) -> str:
        """Extract text content from raw bytes.

        Args:
            raw_bytes: The raw file content.

        Returns:
            Extracted text as a string.
        """

    def _extract_metadata(self, source: str, raw_bytes: bytes) -> dict[str, Any]:
        """Extract metadata from the source and raw bytes.

        Args:
            source: The original source path or identifier.
            raw_bytes: The raw file content.

        Returns:
            Dictionary of metadata key-value pairs.
        """
        return {"filename": Path(source).name}

    def supports(self, source: str) -> bool:
        """Check if this loader supports the given source.

        Args:
            source: File path or identifier.

        Returns:
            True if the source extension matches this loader.
        """
        ext = Path(source).suffix.lower()
        return ext in self.supported_extensions()

    def load(self, source: str | Path | bytes) -> Document:
        """Load a document from a file path or raw bytes.

        Args:
            source: A file path (str or Path) or raw bytes.

        Returns:
            A Document domain model with extracted text and metadata.

        Raises:
            FileNotFoundError: If the source file does not exist.
            DocumentLoadError: If no text content can be extracted.
        """
        if isinstance(source, bytes):
            raw_bytes = source
            sha = hashlib.sha256(raw_bytes).hexdigest()
            source_uri = f"bytes://{sha}"
        else:
            path = Path(source)
            if not path.exists():
                msg = f"File not found: {path}"
                raise FileNotFoundError(msg)
            raw_bytes = path.read_bytes()
            source_uri = str(path.resolve())

        text = self._extract_text(raw_bytes)
        if not text or not text.strip():
            msg = f"No text content extracted from {source_uri}"
            raise DocumentLoadError(msg)

        source_str = source_uri if isinstance(source, bytes) else str(source)
        metadata = self._extract_metadata(source_str, raw_bytes)
        metadata["content_length"] = len(raw_bytes)
        metadata["text_length"] = len(text)

        return Document(
            id=uuid4(),
            collection_id=uuid4(),  # placeholder — set by caller
            source_uri=source_uri,
            content_type=self.content_type(),
            raw_content=text,
            content_size_bytes=len(raw_bytes),
            metadata=metadata,
            status=DocumentStatus.PENDING,
            created_at=datetime.now(UTC),
        )


class DocumentLoaderRegistry:
    """Registry of document loaders, selects loader by file extension.

    Args:
        loaders: Optional list of loaders to register at init.
    """

    def __init__(self, loaders: list[BaseDocumentLoader] | None = None) -> None:
        self._loaders: list[BaseDocumentLoader] = []
        for loader in loaders or []:
            self.register(loader)

    def register(self, loader: BaseDocumentLoader) -> None:
        """Register a document loader.

        Args:
            loader: The loader instance to register.
        """
        self._loaders.append(loader)
        logger.info(
            "Registered document loader: %s (%s)",
            type(loader).__name__,
            loader.content_type(),
        )

    def get_loader(self, source: str) -> BaseDocumentLoader:
        """Find the first loader that supports the given source.

        Args:
            source: File path or identifier.

        Returns:
            A matching BaseDocumentLoader.

        Raises:
            UnsupportedFormatError: If no loader supports the source.
        """
        for loader in self._loaders:
            if loader.supports(source):
                return loader
        ext = Path(source).suffix.lower()
        msg = f"No loader registered for extension '{ext}'"
        raise UnsupportedFormatError(msg)

    def load(self, source: str) -> Document:
        """Load a document using the appropriate loader.

        Args:
            source: File path or identifier.

        Returns:
            A Document domain model.

        Raises:
            UnsupportedFormatError: If no loader supports the source.
        """
        loader = self.get_loader(source)
        return loader.load(source)

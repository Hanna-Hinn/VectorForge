"""Unit tests for VectorForge document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from vectorforge.exceptions import DocumentLoadError, UnsupportedFormatError
from vectorforge.ingestion.loaders.base import DocumentLoaderRegistry
from vectorforge.ingestion.loaders.html_loader import HTMLLoader
from vectorforge.ingestion.loaders.markdown_loader import MarkdownLoader
from vectorforge.ingestion.loaders.text_loader import TextLoader
from vectorforge.models.domain import DocumentStatus

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "sample_documents"


# ---------------------------------------------------------------------------
# TextLoader tests
# ---------------------------------------------------------------------------


class TestTextLoader:
    """Tests for the plain-text document loader."""

    def test_supported_extensions(self) -> None:
        loader = TextLoader()
        assert ".txt" in loader.supported_extensions()

    def test_content_type(self) -> None:
        loader = TextLoader()
        assert loader.content_type() == "text/plain"

    def test_supports_txt_file(self) -> None:
        loader = TextLoader()
        assert loader.supports("document.txt") is True
        assert loader.supports("document.md") is False

    def test_load_from_file(self) -> None:
        loader = TextLoader()
        doc = loader.load(str(FIXTURES_DIR / "sample.txt"))
        assert doc.raw_content is not None
        assert "plain text sample" in doc.raw_content
        assert doc.content_type == "text/plain"
        assert doc.status == DocumentStatus.PENDING

    def test_load_from_bytes(self) -> None:
        loader = TextLoader()
        content = b"Hello, world!"
        doc = loader.load(content)
        assert doc.raw_content == "Hello, world!"
        assert doc.source_uri.startswith("bytes://")

    def test_load_empty_content_raises(self) -> None:
        loader = TextLoader()
        with pytest.raises(DocumentLoadError, match="No text content"):
            loader.load(b"   ")

    def test_load_missing_file_raises(self) -> None:
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.txt")


# ---------------------------------------------------------------------------
# MarkdownLoader tests
# ---------------------------------------------------------------------------


class TestMarkdownLoader:
    """Tests for the Markdown document loader."""

    def test_supported_extensions(self) -> None:
        loader = MarkdownLoader()
        exts = loader.supported_extensions()
        assert ".md" in exts
        assert ".markdown" in exts

    def test_content_type(self) -> None:
        loader = MarkdownLoader()
        assert loader.content_type() == "text/markdown"

    def test_load_strips_frontmatter(self) -> None:
        content = b"---\ntitle: Test\n---\n# Heading\nBody text."
        loader = MarkdownLoader()
        doc = loader.load(content)
        assert doc.raw_content is not None
        assert "---" not in doc.raw_content
        assert "Heading" in doc.raw_content

    def test_load_from_file(self) -> None:
        loader = MarkdownLoader()
        doc = loader.load(str(FIXTURES_DIR / "sample.md"))
        assert doc.raw_content is not None
        assert "Sample Markdown Document" in doc.raw_content


# ---------------------------------------------------------------------------
# HTMLLoader tests
# ---------------------------------------------------------------------------


class TestHTMLLoader:
    """Tests for the HTML document loader."""

    def test_supported_extensions(self) -> None:
        loader = HTMLLoader()
        exts = loader.supported_extensions()
        assert ".html" in exts
        assert ".htm" in exts

    def test_content_type(self) -> None:
        loader = HTMLLoader()
        assert loader.content_type() == "text/html"

    def test_load_strips_scripts_and_styles(self) -> None:
        loader = HTMLLoader()
        doc = loader.load(str(FIXTURES_DIR / "sample.html"))
        assert doc.raw_content is not None
        assert "var x" not in doc.raw_content
        assert "margin" not in doc.raw_content
        assert "Sample HTML Document" in doc.raw_content

    def test_load_from_bytes(self) -> None:
        html = b"<html><body><p>Hello</p><script>bad()</script></body></html>"
        loader = HTMLLoader()
        doc = loader.load(html)
        assert doc.raw_content is not None
        assert "Hello" in doc.raw_content
        assert "bad()" not in doc.raw_content


# ---------------------------------------------------------------------------
# PDFLoader tests (only test supports, since PDF bytes are complex)
# ---------------------------------------------------------------------------


class TestPDFLoader:
    """Tests for the PDF document loader (extension matching only)."""

    def test_supported_extensions(self) -> None:
        from vectorforge.ingestion.loaders.pdf_loader import PDFLoader

        loader = PDFLoader()
        assert ".pdf" in loader.supported_extensions()

    def test_content_type(self) -> None:
        from vectorforge.ingestion.loaders.pdf_loader import PDFLoader

        loader = PDFLoader()
        assert loader.content_type() == "application/pdf"


# ---------------------------------------------------------------------------
# DocumentLoaderRegistry tests
# ---------------------------------------------------------------------------


class TestDocumentLoaderRegistry:
    """Tests for the loader registry."""

    def test_register_and_get_loader(self) -> None:
        registry = DocumentLoaderRegistry()
        loader = TextLoader()
        registry.register(loader)
        found = registry.get_loader("document.txt")
        assert found is loader

    def test_get_loader_unsupported_raises(self) -> None:
        registry = DocumentLoaderRegistry()
        registry.register(TextLoader())
        with pytest.raises(UnsupportedFormatError, match="No loader registered"):
            registry.get_loader("file.xyz")

    def test_load_through_registry(self) -> None:
        registry = DocumentLoaderRegistry(loaders=[TextLoader(), MarkdownLoader()])
        doc = registry.load(str(FIXTURES_DIR / "sample.txt"))
        assert doc.raw_content is not None
        assert "plain text sample" in doc.raw_content

    def test_registry_selects_correct_loader(self) -> None:
        registry = DocumentLoaderRegistry(loaders=[TextLoader(), MarkdownLoader()])
        loader = registry.get_loader("readme.md")
        assert isinstance(loader, MarkdownLoader)

    def test_empty_registry_raises(self) -> None:
        registry = DocumentLoaderRegistry()
        with pytest.raises(UnsupportedFormatError):
            registry.get_loader("anything.txt")

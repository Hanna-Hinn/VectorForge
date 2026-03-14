"""Unit tests for VectorForge chunking strategies."""

from __future__ import annotations

import pytest

from vectorforge.chunking.html import HTMLChunker
from vectorforge.chunking.markdown import MarkdownChunker
from vectorforge.chunking.recursive import RecursiveChunker
from vectorforge.chunking.registry import ChunkerRegistry
from vectorforge.chunking.semantic import SemanticChunker
from vectorforge.chunking.token import TokenChunker
from vectorforge.chunking.xml import XMLChunker
from vectorforge.config.settings import ChunkingConfig
from vectorforge.exceptions import ChunkerNotFoundError

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

LONG_TEXT = (
    "This is a paragraph about vector databases. "
    "They store high-dimensional embeddings for similarity search.\n\n"
    "Another paragraph discusses chunking strategies. "
    "Recursive splitting is the most common approach.\n\n"
    "A third paragraph covers embedding models. "
    "Models like Voyage and Cohere produce dense vectors.\n\n"
    "Finally, retrieval augmented generation combines search with LLMs."
)


def _default_config(**overrides: object) -> ChunkingConfig:
    """Build a ChunkingConfig with small sizes for testing."""
    defaults: dict[str, object] = {
        "chunk_size": 80,
        "chunk_overlap": 10,
        "strategy": "recursive",
    }
    defaults.update(overrides)
    return ChunkingConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RecursiveChunker tests
# ---------------------------------------------------------------------------


class TestRecursiveChunker:
    """Tests for the recursive character text splitter."""

    def test_strategy_name(self) -> None:
        assert RecursiveChunker().strategy_name() == "recursive"

    def test_empty_text_returns_empty(self) -> None:
        config = _default_config()
        chunks = RecursiveChunker().chunk("", config)
        assert chunks == []

    def test_short_text_single_chunk(self) -> None:
        config = _default_config(chunk_size=500)
        chunks = RecursiveChunker().chunk("Short text.", config)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_long_text_produces_multiple_chunks(self) -> None:
        config = _default_config(chunk_size=80, chunk_overlap=10)
        chunks = RecursiveChunker().chunk(LONG_TEXT, config)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.index >= 0
            assert len(chunk.text) > 0

    def test_chunk_indices_are_sequential(self) -> None:
        config = _default_config(chunk_size=80, chunk_overlap=10)
        chunks = RecursiveChunker().chunk(LONG_TEXT, config)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# TokenChunker tests
# ---------------------------------------------------------------------------


class TestTokenChunker:
    """Tests for the token-based text splitter."""

    def test_strategy_name(self) -> None:
        assert TokenChunker().strategy_name() == "token"

    def test_empty_text(self) -> None:
        config = _default_config(strategy="token")
        assert TokenChunker().chunk("", config) == []

    def test_produces_chunks(self) -> None:
        config = _default_config(chunk_size=20, chunk_overlap=5, strategy="token")
        chunks = TokenChunker().chunk(LONG_TEXT, config)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# MarkdownChunker tests
# ---------------------------------------------------------------------------


class TestMarkdownChunker:
    """Tests for the markdown header splitter."""

    def test_strategy_name(self) -> None:
        assert MarkdownChunker().strategy_name() == "markdown"

    def test_splits_by_headers(self) -> None:
        md_text = "# Heading 1\nContent one.\n## Heading 2\nContent two."
        config = _default_config(strategy="markdown")
        chunks = MarkdownChunker().chunk(md_text, config)
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Content one" in all_text
        assert "Content two" in all_text

    def test_h5_h6_headers(self) -> None:
        md_text = "##### Heading 5\nFive.\n###### Heading 6\nSix."
        config = _default_config(chunk_size=500, strategy="markdown")
        chunks = MarkdownChunker().chunk(md_text, config)
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Five" in all_text
        assert "Six" in all_text

    def test_long_section_sub_chunked(self) -> None:
        """Oversized sections under a heading get sub-chunked."""
        long_body = "Word " * 200  # ~1000 chars
        md_text = f"# Title\n{long_body}"
        config = _default_config(chunk_size=80, chunk_overlap=10, strategy="markdown")
        chunks = MarkdownChunker().chunk(md_text, config)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= config.chunk_size + config.chunk_overlap


# ---------------------------------------------------------------------------
# HTMLChunker tests
# ---------------------------------------------------------------------------


class TestHTMLChunker:
    """Tests for the HTML header splitter."""

    def test_strategy_name(self) -> None:
        assert HTMLChunker().strategy_name() == "html"

    def test_splits_html(self) -> None:
        html = "<h1>Title</h1><p>Paragraph one.</p><h2>Sub</h2><p>Paragraph two.</p>"
        config = _default_config(strategy="html")
        chunks = HTMLChunker().chunk(html, config)
        assert len(chunks) >= 1

    def test_long_section_sub_chunked(self) -> None:
        """Oversized HTML sections under a heading get sub-chunked."""
        long_body = "Word " * 200  # ~1000 chars
        html = f"<h1>Title</h1><p>{long_body}</p>"
        config = _default_config(chunk_size=80, chunk_overlap=10, strategy="html")
        chunks = HTMLChunker().chunk(html, config)
        assert len(chunks) > 1

    def test_heading_metadata_preserved(self) -> None:
        """Sub-chunks inherit the heading hierarchy metadata."""
        long_body = "Word " * 200
        html = f"<h1>Main</h1><p>{long_body}</p>"
        config = _default_config(chunk_size=80, chunk_overlap=10)
        chunks = HTMLChunker().chunk(html, config)
        for chunk in chunks:
            assert "h1" in chunk.metadata


# ---------------------------------------------------------------------------
# SemanticChunker tests
# ---------------------------------------------------------------------------


class TestSemanticChunker:
    """Tests for the semantic chunker (falls back to recursive)."""

    def test_strategy_name(self) -> None:
        assert SemanticChunker().strategy_name() == "semantic"

    def test_produces_chunks(self) -> None:
        config = _default_config(chunk_size=80, chunk_overlap=10)
        chunks = SemanticChunker().chunk(LONG_TEXT, config)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# ChunkerRegistry tests
# ---------------------------------------------------------------------------


class TestChunkerRegistry:
    """Tests for the chunker registry."""

    def test_register_and_get(self) -> None:
        registry = ChunkerRegistry()
        chunker = RecursiveChunker()
        registry.register(chunker)
        assert registry.get("recursive") is chunker

    def test_get_unknown_raises(self) -> None:
        registry = ChunkerRegistry()
        with pytest.raises(ChunkerNotFoundError, match="Unknown chunking strategy"):
            registry.get("nonexistent")

    def test_list_strategies(self) -> None:
        registry = ChunkerRegistry()
        registry.register(RecursiveChunker())
        registry.register(TokenChunker())
        strategies = registry.list_strategies()
        assert "recursive" in strategies
        assert "token" in strategies

    def test_get_default(self) -> None:
        registry = ChunkerRegistry()
        registry.register(RecursiveChunker())
        registry.set_default("recursive")
        default = registry.get_default()
        assert default.strategy_name() == "recursive"

    def test_get_for_content_type(self) -> None:
        registry = ChunkerRegistry()
        registry.register(RecursiveChunker())
        registry.register(MarkdownChunker())
        chunker = registry.get_for_content_type("text/markdown")
        assert chunker.strategy_name() == "markdown"

    def test_get_for_unknown_content_type_returns_default(self) -> None:
        registry = ChunkerRegistry()
        registry.register(RecursiveChunker())
        registry.set_default("recursive")
        chunker = registry.get_for_content_type("application/octet-stream")
        assert chunker.strategy_name() == "recursive"

    def test_get_for_xml_content_type(self) -> None:
        registry = ChunkerRegistry()
        registry.register(RecursiveChunker())
        registry.register(XMLChunker())
        chunker = registry.get_for_content_type("application/xml")
        assert chunker.strategy_name() == "xml"

    def test_get_for_text_xml_content_type(self) -> None:
        registry = ChunkerRegistry()
        registry.register(RecursiveChunker())
        registry.register(XMLChunker())
        chunker = registry.get_for_content_type("text/xml")
        assert chunker.strategy_name() == "xml"


# ---------------------------------------------------------------------------
# XMLChunker tests
# ---------------------------------------------------------------------------


class TestXMLChunker:
    """Tests for the XML tree structure chunker."""

    def test_strategy_name(self) -> None:
        assert XMLChunker().strategy_name() == "xml"

    def test_empty_text(self) -> None:
        config = _default_config()
        assert XMLChunker().chunk("", config) == []

    def test_simple_xml(self) -> None:
        xml = "<root><title>Hello</title><body>World</body></root>"
        config = _default_config(chunk_size=500)
        chunks = XMLChunker().chunk(xml, config)
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Hello" in all_text
        assert "World" in all_text

    def test_nested_xml_structure(self) -> None:
        xml = (
            "<book>"
            "<chapter><title>Ch1</title><p>Para one.</p></chapter>"
            "<chapter><title>Ch2</title><p>Para two.</p></chapter>"
            "</book>"
        )
        config = _default_config(chunk_size=500)
        chunks = XMLChunker().chunk(xml, config)
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Ch1" in all_text
        assert "Ch2" in all_text

    def test_xml_metadata_has_path(self) -> None:
        xml = "<root><section><p>Content here.</p></section></root>"
        config = _default_config(chunk_size=500)
        chunks = XMLChunker().chunk(xml, config)
        assert len(chunks) >= 1
        # At least one chunk should carry xml_path metadata
        paths = [c.metadata.get("xml_path") for c in chunks]
        assert any(p for p in paths)

    def test_oversized_leaf_sub_chunked(self) -> None:
        long_body = "Word " * 200  # ~1000 chars
        xml = f"<root><section>{long_body}</section></root>"
        config = _default_config(chunk_size=80, chunk_overlap=10)
        chunks = XMLChunker().chunk(xml, config)
        assert len(chunks) > 1

    def test_malformed_xml_falls_back(self) -> None:
        """Malformed XML should fall back to recursive splitting."""
        bad_xml = "<root><unclosed>Content without closing tag"
        config = _default_config(chunk_size=500)
        chunks = XMLChunker().chunk(bad_xml, config)
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Content" in all_text

    def test_namespace_stripped(self) -> None:
        xml = '<root xmlns:ns="http://example.com"><ns:item>Data</ns:item></root>'
        config = _default_config(chunk_size=500)
        chunks = XMLChunker().chunk(xml, config)
        assert len(chunks) >= 1
        # Tag should be stripped of namespace
        tags = [c.metadata.get("xml_tag") for c in chunks]
        assert all("}" not in str(t) for t in tags if t)

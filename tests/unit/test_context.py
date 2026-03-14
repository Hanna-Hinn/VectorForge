"""Unit tests for the context builder module."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from vectorforge.models.domain import Chunk, RetrievedChunk
from vectorforge.pipeline.context import (
    ContextBuilder,
    ContextConfig,
    ContextPayload,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "chunk text", index: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
        metadata={},
        created_at=datetime.now(UTC),
    )


def _make_retrieved(
    text: str = "chunk text",
    score: float = 0.9,
    index: int = 0,
    source: str = "doc.txt",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(text, index),
        score=score,
        document_source=source,
    )


# ---------------------------------------------------------------------------
# estimate_tokens tests
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Tests for the token estimation function."""

    def test_returns_positive_for_nonempty(self) -> None:
        assert estimate_tokens("Hello world") > 0

    def test_returns_zero_for_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_longer_text_more_tokens(self) -> None:
        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 100)
        assert long > short


# ---------------------------------------------------------------------------
# ContextBuilder tests
# ---------------------------------------------------------------------------


class TestContextBuilder:
    """Tests for the ContextBuilder."""

    def test_build_basic(self) -> None:
        builder = ContextBuilder()
        chunks = [_make_retrieved("Chunk A", 0.9, 0), _make_retrieved("Chunk B", 0.8, 1)]
        payload = builder.build("What is RAG?", chunks)

        assert isinstance(payload, ContextPayload)
        assert payload.user_message == "What is RAG?"
        assert "[Source 1]" in payload.assembled_context
        assert "[Source 2]" in payload.assembled_context
        assert "Chunk A" in payload.assembled_context
        assert payload.context_token_count > 0

    def test_build_uses_default_system_prompt(self) -> None:
        builder = ContextBuilder()
        chunks = [_make_retrieved("Some text")]
        payload = builder.build("query", chunks)

        assert "helpful assistant" in payload.system_prompt
        assert "Some text" in payload.system_prompt

    def test_build_custom_system_prompt(self) -> None:
        builder = ContextBuilder()
        config = ContextConfig(
            system_prompt_template="Custom prompt. Context: {context}",
        )
        chunks = [_make_retrieved("Data")]
        payload = builder.build("q", chunks, config)

        assert payload.system_prompt.startswith("Custom prompt.")
        assert "Data" in payload.system_prompt

    def test_build_includes_sources(self) -> None:
        builder = ContextBuilder()
        chunks = [_make_retrieved("text", 0.95, 0, "file.pdf")]
        payload = builder.build("q", chunks)

        assert len(payload.sources) == 1
        assert payload.sources[0].document_source == "file.pdf"
        assert payload.sources[0].score == 0.95

    def test_build_no_sources_when_disabled(self) -> None:
        builder = ContextBuilder()
        config = ContextConfig(include_sources=False)
        chunks = [_make_retrieved()]
        payload = builder.build("q", chunks, config)

        assert payload.sources == []

    def test_build_includes_scores_in_context(self) -> None:
        builder = ContextBuilder()
        config = ContextConfig(include_scores=True)
        chunks = [_make_retrieved("text", 0.85)]
        payload = builder.build("q", chunks, config)

        assert "(relevance: 0.85)" in payload.assembled_context

    def test_build_no_scores_by_default(self) -> None:
        builder = ContextBuilder()
        chunks = [_make_retrieved("text", 0.85)]
        payload = builder.build("q", chunks)

        assert "relevance:" not in payload.assembled_context

    def test_build_empty_chunks(self) -> None:
        builder = ContextBuilder()
        payload = builder.build("q", [])

        assert payload.assembled_context == ""
        assert payload.sources == []

    def test_build_truncates_to_token_budget(self) -> None:
        builder = ContextBuilder()
        # Use a very small token budget to force truncation
        config = ContextConfig(max_context_tokens=10)
        chunks = [
            _make_retrieved("A " * 50, 0.9, 0),
            _make_retrieved("B " * 50, 0.8, 1),
            _make_retrieved("C " * 50, 0.7, 2),
        ]
        payload = builder.build("q", chunks, config)

        # Should have dropped some chunks
        assert len(payload.sources) < 3

    def test_build_custom_separator(self) -> None:
        builder = ContextBuilder()
        config = ContextConfig(chunk_separator="\n\n")
        chunks = [_make_retrieved("A", 0.9, 0), _make_retrieved("B", 0.8, 1)]
        payload = builder.build("q", chunks, config)

        # Default separator "---" should not appear
        assert "---" not in payload.assembled_context

    def test_source_snippet_truncation(self) -> None:
        builder = ContextBuilder()
        long_text = "A" * 300
        chunks = [_make_retrieved(long_text)]
        payload = builder.build("q", chunks)

        assert len(payload.sources) == 1
        assert payload.sources[0].snippet.endswith("...")
        assert len(payload.sources[0].snippet) <= 204  # 200 + "..."

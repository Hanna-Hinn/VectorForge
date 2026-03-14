"""Context builder for RAG pipeline.

Assembles retrieved chunks into a structured prompt payload
ready for LLM generation, with token budgeting and source citations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import tiktoken
from pydantic import BaseModel, Field

from vectorforge.models.domain import RetrievedChunk
from vectorforge.pipeline.prompts import DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SourceCitation(BaseModel):
    """A citation referencing the source of a retrieved chunk."""

    document_source: str
    chunk_index: int
    score: float
    snippet: str


class ContextConfig(BaseModel):
    """Configuration for context assembly.

    Args:
        system_prompt_template: Prompt template with ``{context}`` placeholder.
        max_context_tokens: Maximum tokens for the context section.
        include_sources: Whether to build source citations.
        chunk_separator: Separator between formatted chunks.
        include_scores: Whether to show relevance scores in the context.
    """

    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT
    max_context_tokens: int = 4096
    include_sources: bool = True
    chunk_separator: str = "\n---\n"
    include_scores: bool = False


class ContextPayload(BaseModel):
    """The assembled context ready for LLM consumption."""

    system_prompt: str
    user_message: str
    assembled_context: str
    sources: list[SourceCitation] = Field(default_factory=list)
    context_token_count: int = 0


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in *text*.

    Uses tiktoken when available, falling back to a character-based heuristic.

    Args:
        text: The input text.
        model: Model name for tiktoken lookup.

    Returns:
        Estimated token count.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        return len(text) // 4


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class BaseContextBuilder(ABC):
    """Abstract interface for context builders.

    High-level modules (e.g. QueryService) depend on this
    abstraction instead of the concrete ContextBuilder.
    """

    @abstractmethod
    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        config: ContextConfig | None = None,
    ) -> ContextPayload:
        """Assemble a context payload from chunks and query."""


class ContextBuilder(BaseContextBuilder):
    """Builds a structured LLM prompt from retrieved chunks.

    Formats chunk text, enforces a token budget, and generates
    source citations for the response.
    """

    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        config: ContextConfig | None = None,
    ) -> ContextPayload:
        """Assemble a context payload from chunks and query.

        Args:
            query: The original user query.
            chunks: Retrieved chunks ordered by relevance (best first).
            config: Optional context configuration overrides.

        Returns:
            A ContextPayload ready for LLM generation.
        """
        cfg = config or ContextConfig()

        # Format and truncate
        used_chunks, context_text = self._format_and_truncate(chunks, cfg)

        token_count = estimate_tokens(context_text)

        # Build system prompt
        system_prompt = cfg.system_prompt_template.format(context=context_text)

        # Build source citations
        sources: list[SourceCitation] = []
        if cfg.include_sources:
            sources = self._build_sources(used_chunks)

        return ContextPayload(
            system_prompt=system_prompt,
            user_message=query,
            assembled_context=context_text,
            sources=sources,
            context_token_count=token_count,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chunk(
        index: int,
        chunk: RetrievedChunk,
        include_scores: bool,
    ) -> str:
        """Format a single chunk with a source header.

        Args:
            index: 1-based source index.
            chunk: The retrieved chunk.
            include_scores: Whether to append the relevance score.

        Returns:
            Formatted chunk string.
        """
        header = f"[Source {index}]"
        if include_scores:
            header += f" (relevance: {chunk.score:.2f})"
        return f"{header}\n{chunk.chunk.text}"

    def _format_and_truncate(
        self,
        chunks: list[RetrievedChunk],
        config: ContextConfig,
    ) -> tuple[list[RetrievedChunk], str]:
        """Format chunks and drop lowest-scored ones to fit the token budget.

        Args:
            chunks: All retrieved chunks.
            config: Context configuration.

        Returns:
            Tuple of (chunks actually used, assembled context string).
        """
        used: list[RetrievedChunk] = list(chunks)
        context_text = self._join_chunks(used, config)
        token_count = estimate_tokens(context_text)

        original_count = len(used)
        while token_count > config.max_context_tokens and len(used) > 1:
            used.pop()  # drop lowest-scored (last)
            context_text = self._join_chunks(used, config)
            token_count = estimate_tokens(context_text)

        if len(used) < original_count:
            logger.warning(
                "Context truncated: dropped %d chunks to fit token budget (%d tokens)",
                original_count - len(used),
                token_count,
            )

        return used, context_text

    def _join_chunks(
        self,
        chunks: list[RetrievedChunk],
        config: ContextConfig,
    ) -> str:
        """Join formatted chunks with the configured separator.

        Args:
            chunks: Chunks to join.
            config: Context configuration.

        Returns:
            The joined context string.
        """
        parts = [
            self._format_chunk(i + 1, c, config.include_scores)
            for i, c in enumerate(chunks)
        ]
        return config.chunk_separator.join(parts)

    @staticmethod
    def _build_sources(
        chunks: list[RetrievedChunk],
    ) -> list[SourceCitation]:
        """Build source citation objects from the used chunks.

        Args:
            chunks: The chunks that made it into the context.

        Returns:
            List of SourceCitation objects.
        """
        snippet_limit = 200
        sources: list[SourceCitation] = []
        for chunk in chunks:
            text = chunk.chunk.text
            snippet = text[:snippet_limit] + "..." if len(text) > snippet_limit else text
            sources.append(
                SourceCitation(
                    document_source=chunk.document_source,
                    chunk_index=chunk.chunk.index,
                    score=chunk.score,
                    snippet=snippet,
                )
            )
        return sources

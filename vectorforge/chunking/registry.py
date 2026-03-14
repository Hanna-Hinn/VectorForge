"""Chunker registry for strategy selection."""

from __future__ import annotations

import logging

from vectorforge.chunking.base import BaseChunker
from vectorforge.exceptions import ChunkerNotFoundError, DuplicateError

logger = logging.getLogger(__name__)

_DEFAULT_CONTENT_TYPE_MAP: dict[str, str] = {
    "text/markdown": "markdown",
    "text/html": "html",
    "application/xml": "xml",
    "text/xml": "xml",
}


class ChunkerRegistry:
    """Registry of chunking strategies.

    Maps strategy names to chunker instances and provides
    content-type-based auto-selection.

    Args:
        default_strategy: The default strategy name (default: "recursive").
    """

    def __init__(self, default_strategy: str = "recursive") -> None:
        self._chunkers: dict[str, BaseChunker] = {}
        self._default_strategy = default_strategy
        self._content_type_map: dict[str, str] = dict(_DEFAULT_CONTENT_TYPE_MAP)

    def register(self, chunker: BaseChunker) -> None:
        """Register a chunker strategy.

        Args:
            chunker: The chunker instance to register.

        Raises:
            DuplicateError: If the strategy name is already registered.
        """
        name = chunker.strategy_name()
        if name in self._chunkers:
            msg = f"Chunker strategy '{name}' is already registered"
            raise DuplicateError(msg)
        self._chunkers[name] = chunker
        logger.info("Registered chunker strategy: %s", name)

    def get(self, strategy: str) -> BaseChunker:
        """Get a chunker by strategy name.

        Args:
            strategy: The strategy name.

        Returns:
            The registered BaseChunker instance.

        Raises:
            ChunkerNotFoundError: If the strategy is not registered.
        """
        if strategy not in self._chunkers:
            msg = f"Unknown chunking strategy: {strategy}"
            raise ChunkerNotFoundError(msg)
        return self._chunkers[strategy]

    def get_default(self) -> BaseChunker:
        """Get the default chunker.

        Returns:
            The default BaseChunker instance.

        Raises:
            ChunkerNotFoundError: If the default strategy is not registered.
        """
        return self.get(self._default_strategy)

    def get_for_content_type(self, content_type: str) -> BaseChunker:
        """Get the best chunker for a given content type.

        Uses a content-type-to-strategy mapping. Falls back to the
        default strategy for unmapped content types.

        Args:
            content_type: MIME content type string.

        Returns:
            The best matching BaseChunker.
        """
        strategy = self._content_type_map.get(content_type, self._default_strategy)
        try:
            return self.get(strategy)
        except ChunkerNotFoundError:
            return self.get(self._default_strategy)

    def list_strategies(self) -> list[str]:
        """List all registered strategy names.

        Returns:
            Sorted list of strategy name strings.
        """
        return sorted(self._chunkers.keys())

    def register_content_type(self, content_type: str, strategy: str) -> None:
        """Map a MIME content type to a chunking strategy.

        Args:
            content_type: MIME content type string.
            strategy: The strategy name to use for this content type.
        """
        self._content_type_map[content_type] = strategy

    def set_default(self, strategy: str) -> None:
        """Change the default strategy.

        Args:
            strategy: The new default strategy name.

        Raises:
            ChunkerNotFoundError: If the strategy is not registered.
        """
        if strategy not in self._chunkers:
            msg = f"Cannot set default: strategy '{strategy}' not registered"
            raise ChunkerNotFoundError(msg)
        self._default_strategy = strategy

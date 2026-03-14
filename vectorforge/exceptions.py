"""Custom exception hierarchy for VectorForge.

All VectorForge exceptions inherit from VectorForgeError,
enabling catch-all handling while allowing specific catches.
"""


class VectorForgeError(Exception):
    """Base exception for all VectorForge errors."""


class ConfigurationError(VectorForgeError):
    """Invalid or missing configuration."""


class DatabaseError(VectorForgeError):
    """Database operation failed."""


class NotFoundError(VectorForgeError):
    """Requested resource was not found."""


class DuplicateError(VectorForgeError):
    """Resource already exists (unique constraint violation)."""


class EmbeddingError(VectorForgeError):
    """Failed to generate embeddings."""


class RetrievalError(VectorForgeError):
    """Failed to retrieve documents from vector store."""


class LLMError(VectorForgeError):
    """Failed to generate LLM response."""


class StorageError(VectorForgeError):
    """Document storage operation failed."""


class DocumentLoadError(VectorForgeError):
    """Failed to load or parse a document."""


class UnsupportedFormatError(VectorForgeError):
    """Document format is not supported by any registered loader."""


class EvaluationError(VectorForgeError):
    """Evaluation pipeline operation failed."""


class ChunkerNotFoundError(VectorForgeError):
    """Requested chunking strategy is not registered."""

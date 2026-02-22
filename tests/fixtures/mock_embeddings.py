"""Mock embedding vectors for testing."""

from __future__ import annotations


def make_mock_vector(dimensions: int = 1024, seed: float = 0.1) -> list[float]:
    """Generate a deterministic mock embedding vector.

    Args:
        dimensions: Number of dimensions.
        seed: Base value for generating sequential floats.

    Returns:
        A list of floats representing a mock embedding.
    """
    return [seed + (i * 0.001) for i in range(dimensions)]


MOCK_VECTOR_4D = [0.1, 0.2, 0.3, 0.4]
MOCK_VECTOR_1024D = make_mock_vector(1024)

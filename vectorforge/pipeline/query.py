"""Query preprocessing utilities.

Provides text cleaning and validation for user queries before
they enter the retrieval pipeline.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 10_000


def preprocess_query(query: str) -> str:
    """Clean and validate a user query string.

    Steps:
      1. Strip leading / trailing whitespace.
      2. Collapse multiple spaces into a single space.
      3. Reject empty queries.
      4. Truncate to ``MAX_QUERY_LENGTH`` with a warning.

    Args:
        query: The raw query from the user.

    Returns:
        The cleaned query string.

    Raises:
        ValueError: If the query is empty after stripping.
    """
    query = " ".join(query.split())
    if not query:
        msg = "Query cannot be empty"
        raise ValueError(msg)
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(
            "Query truncated from %d to %d chars",
            len(query),
            MAX_QUERY_LENGTH,
        )
        query = query[:MAX_QUERY_LENGTH]
    return query

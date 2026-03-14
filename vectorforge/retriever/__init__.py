"""Retriever module — strategies for finding relevant chunks."""

from vectorforge.retriever.base import BaseRetriever
from vectorforge.retriever.dense import DenseRetriever

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
]

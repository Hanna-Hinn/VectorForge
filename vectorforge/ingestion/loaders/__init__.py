"""Document loaders for VectorForge ingestion pipeline."""

from vectorforge.ingestion.loaders.base import BaseDocumentLoader, DocumentLoaderRegistry
from vectorforge.ingestion.loaders.html_loader import HTMLLoader
from vectorforge.ingestion.loaders.markdown_loader import MarkdownLoader
from vectorforge.ingestion.loaders.pdf_loader import PDFLoader
from vectorforge.ingestion.loaders.text_loader import TextLoader
from vectorforge.ingestion.loaders.xml_loader import XMLLoader

__all__ = [
    "BaseDocumentLoader",
    "DocumentLoaderRegistry",
    "HTMLLoader",
    "MarkdownLoader",
    "PDFLoader",
    "TextLoader",
    "XMLLoader",
]

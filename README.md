# VectorForge

**High-performance, standalone Retrieval-Augmented Generation (RAG) engine.**

VectorForge is a modular Python RAG engine built on PostgreSQL + pgvector. It handles the full document pipeline — loading, chunking, embedding, vector storage, and retrieval — in a single deployable unit.

## Features

- **Document Loaders** — Text, HTML, PDF with extensible loader registry
- **Chunking Strategies** — Recursive, Token, Markdown-aware, HTML-aware, XML-aware, Semantic (planned)
- **Embedding Providers** — Voyage AI, Cohere, LiteLLM (100+ providers via proxy)
- **Vector Store** — pgvector with HNSW indexing and cosine / L2 / inner-product similarity
- **Storage Backends** — PostgreSQL + S3-compatible with automatic size-based routing
- **Ingestion Pipeline** — End-to-end orchestrator from raw document to indexed embeddings

## Architecture

```
vectorforge/
├── ingestion/          # Document loading and ingestion orchestration
│   └── loaders/        # Text, HTML, PDF loaders
├── chunking/           # Pluggable chunking strategies
├── embedding/          # Embedding provider integrations
│   └── providers/      # Voyage, Cohere, LiteLLM
├── vectorstore/        # pgvector-backed vector operations
├── storage/            # Document storage (PostgreSQL, S3)
├── db/                 # Database layer (SQLAlchemy async, repositories)
├── models/             # Pydantic domain models and DTOs
├── config/             # Settings management (pydantic-settings)
└── exceptions.py       # Custom exception hierarchy
```

## Installation

```bash
# Core install
pip install -e .

# With development tools
pip install -e ".[dev]"

# With LiteLLM support (generic proxy for 100+ embedding providers)
pip install -e ".[litellm]"
```

## Configuration

VectorForge reads environment variables with the `VECTORFORGE_` prefix. Create a `.env` file in the project root:

```env
# Database
VECTORFORGE_DB_HOST=localhost
VECTORFORGE_DB_PORT=5432
VECTORFORGE_DB_DATABASE=vectorforge
VECTORFORGE_DB_USER=postgres
VECTORFORGE_DB_PASSWORD=secret

# Embedding Providers (configure one or more)
VECTORFORGE_VOYAGE_API_KEY=your-voyage-key
VECTORFORGE_COHERE_API_KEY=your-cohere-key
VECTORFORGE_LITELLM_API_KEY=your-openai-or-other-key

# Chunking defaults
VECTORFORGE_CHUNKING_STRATEGY=recursive
VECTORFORGE_CHUNKING_CHUNK_SIZE=512
VECTORFORGE_CHUNKING_CHUNK_OVERLAP=50

# Storage
VECTORFORGE_STORAGE_DEFAULT_BACKEND=pg
VECTORFORGE_STORAGE_THRESHOLD_MB=5

# S3 (optional — enables automatic routing of large documents)
VECTORFORGE_STORAGE_S3_BUCKET=my-bucket
VECTORFORGE_STORAGE_S3_REGION=us-east-1
VECTORFORGE_STORAGE_S3_ACCESS_KEY=AKID
VECTORFORGE_STORAGE_S3_SECRET_KEY=secret
```

## Chunking Strategies

All chunkers honour `chunk_size` and `chunk_overlap` from configuration. Structure-aware chunkers (HTML, Markdown, XML) use a **two-pass** approach: first split on semantic boundaries (headings / tags), then sub-chunk any oversized sections with recursive character splitting.

| Strategy | Auto-mapped Content Types | Description |
|------------|--------------------------|-------------|
| `recursive` | *(default for all)* | Hierarchical character splitting with configurable separators |
| `token` | — | Token-count splitting using tiktoken encoding |
| `markdown` | `text/markdown` | Two-pass: heading-aware split → recursive sub-chunking |
| `html` | `text/html` | Two-pass: heading-aware split → recursive sub-chunking |
| `xml` | `application/xml`, `text/xml` | Structure-aware XML tree walk → recursive sub-chunking |
| `semantic` | — | Embedding-based breakpoint detection *(planned — future phase)* |

## Embedding Providers

| Provider | Models | Integration |
|----------|--------|-------------|
| **Voyage AI** | voyage-3, voyage-3-lite, voyage-code-3, voyage-finance-2, voyage-law-2 | Direct API via httpx |
| **Cohere** | embed-v4.0, embed-english-v3.0, embed-multilingual-v3.0 | Direct API via httpx |
| **LiteLLM** | Any LiteLLM-supported model | Proxy to 100+ providers (OpenAI, Azure, Bedrock, etc.) |

Providers are auto-discovered from environment variables. Set the corresponding `VECTORFORGE_*_API_KEY` to enable a provider.

## Development

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -e ".[dev]"

# Run tests
python -m pytest
python -m pytest --cov=vectorforge --cov-report=term-missing

# Linting & formatting
python -m ruff check .
python -m ruff format .

# Type checking
python -m mypy vectorforge/
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Database | PostgreSQL + pgvector |
| ORM | SQLAlchemy 2.0 async + asyncpg |
| Validation | Pydantic v2 |
| Configuration | pydantic-settings |
| HTTP Client | httpx |
| Text Splitting | langchain-text-splitters |
| Testing | pytest + pytest-asyncio |
| Linting | Ruff |
| Type Checking | mypy (strict) |

## License

MIT — see [LICENSE](LICENSE) for details.

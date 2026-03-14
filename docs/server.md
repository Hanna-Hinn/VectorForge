# VectorForge — REST API Server

> Complete reference for the FastAPI server that exposes VectorForge capabilities over HTTP.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Running the Server](#running-the-server)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
  - [Collections](#collections)
  - [Documents](#documents)
  - [Query](#query)
  - [Analytics](#analytics)
  - [Status](#status)
- [Middleware](#middleware)
- [Dependency Injection](#dependency-injection)
- [Logging](#logging)
- [Error Handling](#error-handling)
- [CORS](#cors)

---

## Overview

The server is a standalone FastAPI application in the `server/` package, isolated from the core `vectorforge/` library. It provides:

- **CRUD** for collections and documents
- **Document ingestion** (single + batch) backed by the ingestion pipeline
- **RAG queries** — synchronous and SSE-streaming
- **Analytics** — query log summaries, top queries, latency stats
- **Health checks** — deep probes for database and pgvector

---

## Architecture

```
server/
├── __init__.py          # Package marker
├── __main__.py          # Entry point: python -m server
├── app.py               # FastAPI factory, lifespan, middleware stack
├── config.py            # APIConfig (pydantic-settings)
├── dependencies.py      # Dependency injection (session, auth, services)
├── middleware.py         # Error handler + request logging middleware
├── schemas.py           # Pydantic request/response schemas
└── routes/
    ├── __init__.py
    ├── analytics.py     # GET /api/analytics/:id/summary|top-queries|latency
    ├── collections.py   # CRUD /api/collections
    ├── documents.py     # CRUD + ingest /api/collections/:id/documents
    ├── query.py         # POST /api/query, POST /api/query/stream
    └── status.py        # GET /api/status, GET /api/status/providers
```

### Key Design Decisions

- **Isolated from core** — the `server/` package imports from `vectorforge/` but not vice versa.
- **Service singletons via lifespan** — heavy objects (vector store, loader/chunker registries, storage router) are created once at startup and stored on `app.state`.
- **Dependency injection** — FastAPI `Depends()` wires sessions, registries, and fully-assembled service objects into route handlers.
- **No business logic in routes** — routes are thin adapters that validate input, call a service, and format the response.

---

## Running the Server

```bash
# Requires the [server] extra
pip install -e ".[server]"

# Start the server
python -m server
```

This starts uvicorn on `127.0.0.1:8000` by default. The FastAPI interactive docs are at:

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |

---

## Configuration

All settings use the `VECTORFORGE_API_` prefix and are loaded via `pydantic-settings`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VECTORFORGE_API_HOST` | `str` | `127.0.0.1` | Bind address |
| `VECTORFORGE_API_PORT` | `int` | `8000` | Bind port |
| `VECTORFORGE_API_CORS_ORIGINS` | `list[str]` | `["*"]` | Allowed origins |
| `VECTORFORGE_API_API_KEY` | `str` | `""` | API key secret |
| `VECTORFORGE_API_AUTH_REQUIRED` | `bool` | `false` | Enforce auth |
| `VECTORFORGE_API_LOG_REQUESTS` | `bool` | `true` | Log every request |

The server also requires the core `VECTORFORGE_*` environment variables (DB, embedding, LLM) described in the [Setup Guide](setup-guide.md).

---

## Authentication

When `VECTORFORGE_API_AUTH_REQUIRED=true`, all endpoints (except `GET /api/status`) require the `X-Api-Key` header:

```bash
curl -H "X-Api-Key: your-secret" http://127.0.0.1:8000/api/collections
```

- Missing or invalid keys return `401 Unauthorized`.
- Comparison uses `hmac.compare_digest` for timing safety.
- The health check endpoint (`GET /api/status`) is always public for load balancer probes.

---

## API Endpoints

### Collections

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/api/collections` | List all collections | Yes |
| `POST` | `/api/collections` | Create a collection | Yes |
| `GET` | `/api/collections/{id}` | Get collection detail | Yes |
| `DELETE` | `/api/collections/{id}` | Delete a collection | Yes |

**Create Collection Request:**

```json
{
  "name": "my-docs",
  "description": "My document collection",
  "metric": "cosine",
  "embedding_provider": "voyage",
  "embedding_model": "voyage-3",
  "chunking_strategy": "recursive",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**Collection Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "my-docs",
  "description": "My document collection",
  "embedding_config": {"default_provider": "voyage", "metric": "cosine"},
  "chunking_config": {"strategy": "recursive", "chunk_size": 1000},
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": null
}
```

### Documents

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/api/collections/{id}/documents` | List documents (paginated) | Yes |
| `POST` | `/api/collections/{id}/documents` | Ingest a single document | Yes |
| `POST` | `/api/collections/{id}/documents/batch` | Batch ingest documents | Yes |
| `GET` | `/api/documents/{id}` | Get document detail | Yes |
| `DELETE` | `/api/documents/{id}` | Delete a document | Yes |

**Query Parameters for List:**

| Param | Type | Default | Range |
|-------|------|---------|-------|
| `limit` | `int` | `50` | 1–500 |
| `offset` | `int` | `0` | >= 0 |

**Ingest Request:**

```json
{
  "source": "/path/to/document.md",
  "metadata": {"author": "Jane"},
  "chunking_strategy": "markdown",
  "chunk_size": 800,
  "chunk_overlap": 100
}
```

**Batch Ingest:** Send a JSON array of ingest request objects.

**Batch Response:**

```json
{
  "results": [
    {"source": "doc1.md", "document_id": "...", "error": null},
    {"source": "doc2.md", "document_id": null, "error": "File not found"}
  ],
  "succeeded": 1,
  "failed": 1
}
```

### Query

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/api/query` | Synchronous RAG query | Yes |
| `POST` | `/api/query/stream` | Streaming RAG query (SSE) | Yes |

**Query Request:**

```json
{
  "query": "What is vector search?",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "top_k": 10,
  "min_score": 0.0,
  "temperature": 0.7,
  "max_tokens": 1024,
  "include_sources": true,
  "max_context_tokens": 4096,
  "llm_provider": "openai",
  "llm_model": "gpt-4o"
}
```

**Synchronous Response:**

```json
{
  "answer": "Vector search is...",
  "sources": [
    {
      "document_source": "guide.md",
      "chunk_index": 3,
      "score": 0.92,
      "snippet": "Vector search enables..."
    }
  ],
  "retrieval_latency_ms": 45.2,
  "generation_latency_ms": 820.1,
  "total_latency_ms": 865.3
}
```

**Streaming (SSE) Events:**

| Event Type | Payload | Description |
|------------|---------|-------------|
| `token` | `{"type": "token", "content": "..."}` | Single token from LLM |
| `done` | `{"type": "done"}` | Stream complete |
| `error` | `{"type": "error", "message": "..."}` | Error occurred |

### Analytics

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/api/analytics/{collection_id}/summary` | Full summary | Yes |
| `GET` | `/api/analytics/{collection_id}/top-queries` | Most frequent queries | Yes |
| `GET` | `/api/analytics/{collection_id}/latency` | Latency statistics | Yes |

All analytics endpoints accept `?from=ISO_DATETIME` for time filtering.

### Status

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `GET` | `/api/status` | Health check (public) | No |
| `GET` | `/api/status/providers` | List registered providers | Yes |

---

## Middleware

The server uses two middleware layers applied in order:

1. **`ErrorHandlerMiddleware`** — catches `VectorForgeError` subclasses and maps them to HTTP status codes:

   | Exception | HTTP Code | Error Code |
   |-----------|-----------|------------|
   | `NotFoundError` | 404 | `not_found` |
   | `DuplicateError` | 409 | `duplicate` |
   | `ConfigurationError` | 500 | `configuration_error` |
   | `DatabaseError` | 500 | `database_error` |
   | `EmbeddingError` | 502 | `embedding_error` |
   | `LLMError` | 502 | `llm_error` |

   Unhandled exceptions return `500` with a generic message (no internal details leaked).

2. **`RequestLoggingMiddleware`** — logs every request with method, path, status code, latency, and a short request ID. Attaches `X-Request-Id` response header.

---

## Dependency Injection

FastAPI dependencies are defined in `server/dependencies.py`:

| Dependency | Type Alias | Description |
|------------|------------|-------------|
| `get_session` | `DbSession` | Async SQLAlchemy session (auto-commit/rollback) |
| `verify_api_key` | `ApiKey` | API key validation |
| `get_embedding_registry` | `EmbeddingReg` | Embedding provider registry |
| `get_llm_registry` | `LLMReg` | LLM provider registry |
| `get_health_checker` | `HealthCheck` | Health checker |
| `get_ingestion_service` | `IngestionDep` | Fully assembled `IngestionService` |
| `get_query_service` | `QueryServiceDep` | Fully assembled `QueryService` |

Route handlers use annotated type aliases:

```python
from server.dependencies import DbSession, ApiKey, IngestionDep

@router.post("/documents")
async def ingest(session: DbSession, ingestion: IngestionDep, _key: ApiKey):
    ...
```

---

## Logging

The server configures structured logging in `server/__main__.py`:

- **Format**: `%(asctime)s | %(levelname)-7s | %(name)s | %(message)s`
- **Default level**: `INFO`
- All route handlers log key operations (create, delete, list, query, health check)
- Request middleware logs every HTTP request with latency
- Sensitive data (API keys, raw query content) is **never** logged

### Log Sources

| Logger | Content |
|--------|---------|
| `server.middleware` | Request/response logging, error handling |
| `server.routes.collections` | CRUD operations |
| `server.routes.documents` | Ingestion operations |
| `server.routes.query` | Query execution, streaming errors |
| `server.routes.analytics` | Analytics queries |
| `server.routes.status` | Health check results |
| `server.app` | Startup/shutdown events |

---

## Error Handling

All errors return consistent JSON:

```json
{
  "error": "not_found",
  "message": "Collection 550e8400-... not found"
}
```

Standard HTTP status codes:

| Code | Meaning |
|------|---------|
| `200` | Success |
| `201` | Created |
| `400` | Bad request / validation error |
| `401` | Missing or invalid API key |
| `404` | Resource not found |
| `409` | Duplicate resource |
| `500` | Internal server error |
| `502` | Upstream service error (embedding/LLM provider) |

---

## CORS

CORS is controlled by `VECTORFORGE_API_CORS_ORIGINS`:

- **Wildcard (`["*"]`)**: Allows all origins, disables credentials (browsers won't send cookies/auth).
- **Specific origins (`["http://localhost:5173"]`)**: Allows only listed origins with credentials enabled.

For development with the React frontend at `http://localhost:5173`:

```env
VECTORFORGE_API_CORS_ORIGINS=["http://localhost:5173"]
```

For production, list your actual domain(s):

```env
VECTORFORGE_API_CORS_ORIGINS=["https://your-domain.com"]
```

# VectorForge — Docker Local Testing Guide

Run the complete VectorForge stack (PostgreSQL + pgvector, FastAPI backend, React frontend) on your machine using Docker Compose. This guide walks you through every step from first launch to making real API calls.

---

## Prerequisites

| Tool | Minimum Version | Check |
|------|----------------|-------|
| **Docker Desktop** | 4.x | `docker --version` |
| **Docker Compose** | v2 (bundled with Desktop) | `docker compose version` |
| **curl** or **Postman** | Any | For manual API testing |

> **Windows users**: Docker Desktop for Windows includes both Docker Engine and Compose. Make sure Docker Desktop is running before proceeding.

---

## 1. Configure Environment Variables

Copy the Docker env template to `.env` in the project root:

```powershell
# PowerShell (Windows)
Copy-Item env.docker .env
```

```bash
# Bash (Linux/macOS)
cp env.docker .env
```

Open `.env` and make these edits:

### Required: Database Password

Change the placeholder password to something real:

```env
VECTORFORGE_DB_PASSWORD=my_secure_password_123
```

### Required: At Least One Provider API Key

Uncomment and fill in the providers you want to use:

```env
# For embeddings (default provider is Voyage)
VECTORFORGE_VOYAGE_API_KEY=pa-YOUR_KEY_HERE

# For LLM / generation (default provider is OpenAI)
VECTORFORGE_OPENAI_API_KEY=sk-YOUR_KEY_HERE
```

> **Tip**: If you only want to test the UI and basic CRUD (collections, documents), you can skip API keys for now. Embedding and query endpoints will fail, but everything else works.

### Optional: Switch Providers

To use OpenAI for both embeddings and LLM:

```env
VECTORFORGE_EMBEDDING_DEFAULT_PROVIDER=openai
VECTORFORGE_EMBEDDING_DEFAULT_MODEL=text-embedding-3-small
VECTORFORGE_EMBEDDING_DIMENSIONS=1536
VECTORFORGE_OPENAI_API_KEY=sk-YOUR_KEY_HERE
```

---

## 2. Build & Start

From the project root directory:

```powershell
docker compose up --build
```

This does three things in order:

1. **db** — Starts PostgreSQL 16 with pgvector extension, waits for healthy
2. **api** — Builds the Python backend image, runs Alembic migrations, starts uvicorn on port 8000
3. **frontend** — Builds the React SPA, serves it via nginx on port 3000

The first build takes a few minutes (downloading base images, installing dependencies). Subsequent builds use Docker cache and are much faster.

### What to Expect in the Logs

```
db-1        | database system is ready to accept connections
api-1       | INFO  [alembic.runtime.migration] Running upgrade -> 001, initial schema
api-1       | INFO  | VectorForge API started on 0.0.0.0:8000
frontend-1  | nginx/1.27 started
```

> **Detached mode**: Add `-d` to run in the background: `docker compose up --build -d`

---

## 3. Verify Everything Is Running

### Check Container Status

```powershell
docker compose ps
```

You should see three containers, all with status **Up** (or **Up (healthy)** for `db`):

```
NAME                  STATUS              PORTS
vectorforge-db-1      Up (healthy)        0.0.0.0:5432->5432/tcp
vectorforge-api-1     Up                  0.0.0.0:8000->8000/tcp
vectorforge-frontend-1 Up                 0.0.0.0:3000->80/tcp
```

### Health Check — Backend API

```powershell
curl http://localhost:8000/api/status
```

Expected response:

```json
{
  "status": "healthy",
  "checks": {
    "database": { "status": "healthy" },
    "pgvector": { "status": "healthy" }
  }
}
```

### Check Registered Providers

```powershell
curl http://localhost:8000/api/status/providers
```

### Open the Frontend

Open your browser and go to:

```
http://localhost:3000
```

You should see the VectorForge dashboard.

---

## 4. Manual API Testing — Step by Step

Use the commands below with `curl`. All API endpoints are prefixed with `/api`. You can also use the interactive Swagger docs at:

```
http://localhost:8000/docs
```

### 4.1 Create a Collection

```powershell
curl -X POST http://localhost:8000/api/collections `
  -H "Content-Type: application/json" `
  -d '{\"name\": \"test-collection\", \"description\": \"My first collection\"}'
```

**Bash version:**

```bash
curl -X POST http://localhost:8000/api/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "test-collection", "description": "My first collection"}'
```

Save the `id` from the response — you'll need it for the next steps. Example:

```json
{
  "id": "a1b2c3d4-...",
  "name": "test-collection",
  "description": "My first collection",
  "created_at": "2026-03-14T..."
}
```

### 4.2 List Collections

```powershell
curl http://localhost:8000/api/collections
```

### 4.3 Ingest a Document

Replace `COLLECTION_ID` with the actual ID from step 4.1:

```powershell
curl -X POST http://localhost:8000/api/collections/COLLECTION_ID/documents `
  -H "Content-Type: application/json" `
  -d '{\"content\": \"VectorForge is a high-performance RAG engine built with Python, PostgreSQL, and pgvector. It supports multiple embedding providers, chunking strategies, and LLM integrations for retrieval-augmented generation.\", \"metadata\": {\"source\": \"manual-test\", \"topic\": \"overview\"}}'
```

**Bash version:**

```bash
curl -X POST http://localhost:8000/api/collections/COLLECTION_ID/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "VectorForge is a high-performance RAG engine built with Python, PostgreSQL, and pgvector. It supports multiple embedding providers, chunking strategies, and LLM integrations for retrieval-augmented generation.", "metadata": {"source": "manual-test", "topic": "overview"}}'
```

> **Note**: This endpoint requires a working embedding provider. If you haven't set API keys, you'll get an embedding error — that's expected.

### 4.4 Batch Ingest Multiple Documents

```powershell
curl -X POST http://localhost:8000/api/collections/COLLECTION_ID/documents/batch `
  -H "Content-Type: application/json" `
  -d '{\"documents\": [{\"content\": \"Document chunking splits large texts into smaller pieces for better retrieval accuracy.\", \"metadata\": {\"topic\": \"chunking\"}}, {\"content\": \"pgvector enables fast similarity search using vector embeddings stored directly in PostgreSQL.\", \"metadata\": {\"topic\": \"pgvector\"}}]}'
```

### 4.5 List Documents in a Collection

```powershell
curl "http://localhost:8000/api/collections/COLLECTION_ID/documents?limit=10&offset=0"
```

### 4.6 Get a Single Document

Replace `DOCUMENT_ID` with an ID from the list response:

```powershell
curl http://localhost:8000/api/documents/DOCUMENT_ID
```

### 4.7 Run a RAG Query

```powershell
curl -X POST http://localhost:8000/api/query `
  -H "Content-Type: application/json" `
  -d '{\"collection_id\": \"COLLECTION_ID\", \"query\": \"What is VectorForge?\", \"top_k\": 5}'
```

> **Requires**: Both embedding provider and LLM provider API keys configured.

### 4.8 Run a Streaming RAG Query (SSE)

```powershell
curl -N -X POST http://localhost:8000/api/query/stream `
  -H "Content-Type: application/json" `
  -d '{\"collection_id\": \"COLLECTION_ID\", \"query\": \"How does chunking work?\"}'
```

The `-N` flag disables buffering so you see tokens as they stream in.

### 4.9 View Analytics

```powershell
curl http://localhost:8000/api/analytics/COLLECTION_ID/summary
curl "http://localhost:8000/api/analytics/COLLECTION_ID/top-queries?limit=5"
curl http://localhost:8000/api/analytics/COLLECTION_ID/latency
```

### 4.10 Delete a Document

```powershell
curl -X DELETE http://localhost:8000/api/documents/DOCUMENT_ID
```

### 4.11 Delete a Collection

```powershell
curl -X DELETE http://localhost:8000/api/collections/COLLECTION_ID
```

---

## 5. Test the Frontend UI

With all containers running, open **http://localhost:3000** in your browser.

| Page | What to Test |
|------|-------------|
| **Dashboard** | Overview stats — shows collection count, document count |
| **Collections** | Create, view, delete collections through the UI |
| **Documents** | Upload and browse documents within a collection |
| **Query** | Run RAG queries with the chat-like interface |
| **Analytics** | View query statistics, latency charts, top queries |
| **Evaluations** | Trigger evaluation runs, view scores and recommendations |

The frontend makes all API calls through nginx, which proxies `/api/*` to the backend container.

---

## 6. Evaluation System Testing

### Trigger an Evaluation Run

```powershell
curl -X POST http://localhost:8000/api/evaluations/run `
  -H "Content-Type: application/json" `
  -d '{\"collection_id\": \"COLLECTION_ID\"}'
```

Returns HTTP 202 with a `run_id`.

### Check Evaluation Results

```powershell
# List recent runs
curl http://localhost:8000/api/evaluations/runs

# Get details for a specific run
curl http://localhost:8000/api/evaluations/runs/RUN_ID

# Get individual evaluator results
curl "http://localhost:8000/api/evaluations/runs/RUN_ID/results?evaluator=context_precision"

# View recommendations
curl http://localhost:8000/api/evaluations/recommendations

# View score trends
curl "http://localhost:8000/api/evaluations/trends?limit=10"
```

---

## 7. View Logs

### All Services

```powershell
docker compose logs -f
```

### Single Service

```powershell
docker compose logs -f api       # Backend only
docker compose logs -f db        # Database only
docker compose logs -f frontend  # nginx only
```

### Filter Backend Logs

```powershell
docker compose logs api | Select-String "ERROR"    # PowerShell
docker compose logs api | grep ERROR               # Bash
```

---

## 8. Interactive Swagger Docs

FastAPI auto-generates interactive API documentation:

| URL | Description |
|-----|-------------|
| **http://localhost:8000/docs** | Swagger UI — try endpoints interactively |
| **http://localhost:8000/redoc** | ReDoc — clean read-only documentation |
| **http://localhost:8000/openapi.json** | Raw OpenAPI 3.x schema |

Swagger UI lets you fill in parameters and execute requests directly from the browser — very useful for exploring the API without writing curl commands.

---

## 9. Connect to the Database Directly

The PostgreSQL instance is exposed on port 5432. Connect with any client:

```powershell
# psql (if installed locally)
psql -h localhost -p 5432 -U vectorforge -d vectorforge
```

Or use a GUI tool like **pgAdmin**, **DBeaver**, or **DataGrip** with:

| Field | Value |
|-------|-------|
| Host | `localhost` |
| Port | `5432` |
| Database | `vectorforge` |
| User | `vectorforge` |
| Password | *(whatever you set in `.env`)* |

### Useful SQL Queries

```sql
-- List all tables
\dt

-- Check pgvector extension
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Count collections
SELECT COUNT(*) FROM collections;

-- Count documents
SELECT COUNT(*) FROM documents;

-- Count embeddings
SELECT COUNT(*) FROM embeddings;

-- View a sample embedding (truncated)
SELECT id, document_id, LEFT(chunk_text, 80), vector_dims(embedding) AS dims
FROM embeddings LIMIT 5;
```

---

## 10. Development Mode (Hot-Reload)

For active development, use the dev override so code changes are reflected **instantly** — no rebuild needed.

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### What Changes in Dev Mode

| Service | Production | Development |
|---------|-----------|-------------|
| **Backend** | Static code baked into image | Source mounted as volume; uvicorn `--reload` watches for changes |
| **Frontend** | nginx serves pre-built SPA | Vite dev server with HMR (Hot Module Replacement) |

### How It Works

- **Backend**: `vectorforge/` and `server/` directories are mounted into the container. Uvicorn watches these directories and automatically restarts when you save a file.
- **Frontend**: `src/`, `index.html`, and config files are mounted into the container. Vite's HMR pushes changes to the browser without a full page reload.

### Tips

- The first `--build` is still needed to install dependencies. After that, only code changes trigger hot-reload — no rebuild required.
- If you add a new Python **dependency** to `pyproject.toml`, you need to rebuild: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build api`
- If you add a new npm **package** to `package.json`, rebuild the frontend: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build frontend`

---

## 11. Common Operations

### Rebuild After Code Changes

```powershell
docker compose up --build
```

### Rebuild a Single Service

```powershell
docker compose up --build api        # Rebuild backend only
docker compose up --build frontend   # Rebuild frontend only
```

### Stop Everything

```powershell
docker compose down
```

### Stop and Delete All Data (Fresh Start)

```powershell
docker compose down -v
```

> **Warning**: The `-v` flag deletes the PostgreSQL data volume. All collections, documents, and embeddings will be lost.

### Check Resource Usage

```powershell
docker compose stats
```

### Shell into a Container

```powershell
# Backend container
docker compose exec api bash

# Database container  
docker compose exec db psql -U vectorforge

# Frontend container (Alpine — use sh)
docker compose exec frontend sh
```

### Run Alembic Migrations Manually

```powershell
docker compose exec api python -m alembic upgrade head
docker compose exec api python -m alembic history
```

---

## 12. Troubleshooting

### API container keeps restarting

Check logs: `docker compose logs api`. Common causes:

- **Missing `.env`** — Copy `env.docker` to `.env`
- **Database not ready** — The healthcheck + `depends_on` should handle this. If not, restart: `docker compose restart api`
- **Migration failure** — Check if the database password in `.env` matches

### "Connection refused" on localhost:8000

- Make sure the `api` container is running: `docker compose ps`
- Check if port 8000 is already in use: `netstat -ano | findstr :8000` (Windows)

### "Connection refused" on localhost:3000

- Make sure the `frontend` container is running
- Check nginx logs: `docker compose logs frontend`

### Embedding errors on document ingest

- Verify your API key is set in `.env`
- Check provider name matches: `VECTORFORGE_EMBEDDING_DEFAULT_PROVIDER` must be `voyage`, `openai`, or another registered provider
- Check logs: `docker compose logs api | Select-String "EmbeddingError"`

### Frontend shows "Network Error" or blank page

- The nginx proxy expects the backend at `http://api:8000` (Docker network). Make sure the `api` service is running.
- Hard refresh the browser: `Ctrl+Shift+R`

### Port conflicts

If ports 5432, 8000, or 3000 are already in use, edit `docker-compose.yml` and change the left side of the port mapping:

```yaml
ports:
  - "5433:5432"   # Use 5433 on host instead
  - "8001:8000"   # Use 8001 on host instead
  - "3001:80"     # Use 3001 on host instead
```

---

## Quick Reference

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:3000 | React dashboard |
| API | http://localhost:8000/api/status | Health check |
| Swagger | http://localhost:8000/docs | Interactive API docs |
| ReDoc | http://localhost:8000/redoc | Read-only API docs |
| PostgreSQL | localhost:5432 | Direct DB access |

| Command | What It Does |
|---------|-------------|
| `docker compose up --build` | Build & start all services |
| `docker compose up --build -d` | Same, but detached (background) |
| `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build` | Dev mode with hot-reload |
| `docker compose down` | Stop all services |
| `docker compose down -v` | Stop & delete all data |
| `docker compose logs -f` | Stream all logs |
| `docker compose logs -f api` | Stream backend logs only |
| `docker compose ps` | Show running containers |
| `docker compose exec api bash` | Shell into backend |

# VectorForge — Setup Guide

> Everything you need to install, configure, and connect before running VectorForge.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Install PostgreSQL and pgvector](#1-install-postgresql-and-pgvector)
- [2. Create the Database](#2-create-the-database)
- [3. Clone and Install VectorForge](#3-clone-and-install-vectorforge)
- [4. Configure Environment Variables](#4-configure-environment-variables)
  - [Database](#database)
  - [Embedding Providers](#embedding-providers)
  - [LLM Providers](#llm-providers)
  - [Chunking](#chunking)
  - [Storage](#storage)
  - [Monitoring](#monitoring)
- [5. Run Database Migrations](#5-run-database-migrations)
- [6. Validate the Setup](#6-validate-the-setup)
- [7. Optional Extras](#7-optional-extras)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement      | Minimum Version | Notes                                      |
|------------------|-----------------|---------------------------------------------|
| **Python**       | 3.11+           | Type hints and async features required      |
| **PostgreSQL**   | 14+             | Primary data store                          |
| **pgvector**     | 0.5+            | PostgreSQL extension for vector similarity  |
| **pip**          | 23+             | For editable installs with extras           |
| **Git**          | Any recent      | To clone the repository                     |

---

## 1. Install PostgreSQL and pgvector

### macOS (Homebrew)

```bash
brew install postgresql@16
brew services start postgresql@16

# Install pgvector
brew install pgvector
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# Install pgvector (for PostgreSQL 16)
sudo apt install -y postgresql-16-pgvector
```

### Windows

1. Download the PostgreSQL installer from [postgresql.org](https://www.postgresql.org/download/windows/).
2. Run the installer and note the port (default `5432`) and superuser password.
3. Install pgvector:
   - Download the latest pgvector release from [pgvector/pgvector](https://github.com/pgvector/pgvector/releases).
   - Copy the compiled files into your PostgreSQL `lib` and `share/extension` directories.
   - Alternatively, use `vcpkg` or build from source with Visual Studio.

### Docker (any platform)

```bash
docker run -d \
  --name vectorforge-db \
  -e POSTGRES_USER=vectorforge \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=vectorforge \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

The `pgvector/pgvector` Docker image includes PostgreSQL with pgvector pre-installed.

---

## 2. Create the Database

Connect to PostgreSQL and set up the database and extension:

```bash
# Connect as superuser (or your admin user)
psql -U postgres
```

```sql
-- Create a dedicated user (skip if using Docker defaults)
CREATE USER vectorforge WITH PASSWORD 'your_password';

-- Create the database
CREATE DATABASE vectorforge OWNER vectorforge;

-- Connect to the new database
\c vectorforge

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO vectorforge;
```

Verify pgvector is active:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

Expected output:

```
 extname | extversion
---------+------------
 vector  | 0.7.0
```

---

## 3. Clone and Install VectorForge

```bash
# Clone the repository
git clone https://github.com/Hanna-Hinn/VectorForge.git
cd VectorForge

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install VectorForge with development dependencies
pip install -e ".[dev]"
```

### Optional Extras

| Extra         | Install Command              | What It Adds                              |
|---------------|------------------------------|-------------------------------------------|
| `litellm`     | `pip install -e ".[litellm]"` | LiteLLM provider (100+ LLM/embedding backends) |
| `rerankers`   | `pip install -e ".[rerankers]"` | Cross-encoder re-ranking (sentence-transformers) |
| `dev`         | `pip install -e ".[dev]"`     | pytest, ruff, mypy, pre-commit            |

You can combine extras:

```bash
pip install -e ".[dev,litellm,rerankers]"
```

---

## 4. Configure Environment Variables

VectorForge reads all configuration from environment variables with the `VECTORFORGE_` prefix. A `.env` file in the project root is loaded automatically.

### Quick Start

Copy the example file and edit it:

```bash
# macOS / Linux:
cp .env.example .env

# Windows (PowerShell):
Copy-Item .env.example .env
```

Open `.env` in your editor and fill in the values below.

---

### Database

| Variable                      | Default          | Description                          |
|-------------------------------|------------------|--------------------------------------|
| `VECTORFORGE_DB_HOST`         | `localhost`      | PostgreSQL host                      |
| `VECTORFORGE_DB_PORT`         | `5432`           | PostgreSQL port (1–65535)            |
| `VECTORFORGE_DB_DATABASE`     | `vectorforge`    | Database name                        |
| `VECTORFORGE_DB_USER`         | `vectorforge`    | Database user                        |
| `VECTORFORGE_DB_PASSWORD`     | *(empty)*        | Database password                    |
| `VECTORFORGE_DB_POOL_SIZE`    | `10`             | Connection pool size                 |
| `VECTORFORGE_DB_MAX_OVERFLOW` | `5`              | Max overflow connections             |
| `VECTORFORGE_DB_ECHO_SQL`     | `false`          | Log all SQL statements (debug only)  |

**Example:**

```env
VECTORFORGE_DB_HOST=localhost
VECTORFORGE_DB_PORT=5432
VECTORFORGE_DB_DATABASE=vectorforge
VECTORFORGE_DB_USER=vectorforge
VECTORFORGE_DB_PASSWORD=your_secure_password
```

> **Tip:** Special characters in the password (like `@`, `/`, `%`) are URL-encoded automatically by VectorForge.

---

### Embedding Providers

VectorForge auto-discovers embedding providers from API key environment variables. Set at least one:

| Variable                       | Provider    | Notes                                  |
|--------------------------------|-------------|----------------------------------------|
| `VECTORFORGE_VOYAGE_API_KEY`   | Voyage AI   | Default provider — models: `voyage-3`, `voyage-3-lite`, `voyage-code-3` |
| `VECTORFORGE_COHERE_API_KEY`   | Cohere      | Models: `embed-v4.0`, `embed-english-v3.0`, `embed-multilingual-v3.0` |
| `VECTORFORGE_LITELLM_API_KEY`  | LiteLLM     | Proxy to 100+ providers (OpenAI, Azure, Bedrock, etc.) |

**Provider selection defaults:**

| Variable                               | Default   | Description                    |
|----------------------------------------|-----------|--------------------------------|
| `VECTORFORGE_EMBEDDING_DEFAULT_PROVIDER` | `voyage`  | Which provider to use by default |
| `VECTORFORGE_EMBEDDING_DEFAULT_MODEL`    | `voyage-3` | Default model name             |
| `VECTORFORGE_EMBEDDING_DIMENSIONS`       | `1024`    | Embedding vector dimensions    |
| `VECTORFORGE_EMBEDDING_BATCH_SIZE`       | `100`     | Texts per batch API call       |

**Example (Voyage AI):**

```env
VECTORFORGE_VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxx
VECTORFORGE_EMBEDDING_DEFAULT_PROVIDER=voyage
VECTORFORGE_EMBEDDING_DEFAULT_MODEL=voyage-3
VECTORFORGE_EMBEDDING_DIMENSIONS=1024
```

**Example (OpenAI via LiteLLM):**

```env
VECTORFORGE_LITELLM_API_KEY=sk-xxxxxxxxxxxxxxxx
VECTORFORGE_EMBEDDING_DEFAULT_PROVIDER=litellm
VECTORFORGE_EMBEDDING_DEFAULT_MODEL=text-embedding-3-small
VECTORFORGE_EMBEDDING_DIMENSIONS=1536
```

> **Important:** The `VECTORFORGE_EMBEDDING_DIMENSIONS` value must match the output dimensions of your chosen model. Mismatched dimensions will cause pgvector errors at ingestion time.

---

### LLM Providers

LLM providers are optional — VectorForge can run ingestion and retrieval without them. For RAG queries that include answer generation, set at least one:

| Variable                          | Provider   | Notes                                  |
|-----------------------------------|------------|----------------------------------------|
| `VECTORFORGE_OPENAI_API_KEY`      | OpenAI     | Default — models: `gpt-4o`, `gpt-4o-mini`, etc. |
| `VECTORFORGE_ANTHROPIC_API_KEY`   | Anthropic  | Models: `claude-sonnet-4-20250514`, etc. |
| `VECTORFORGE_LITELLM_API_KEY`     | LiteLLM    | Shared with embedding if both use LiteLLM |

**LLM defaults:**

| Variable                          | Default    | Description                    |
|-----------------------------------|------------|--------------------------------|
| `VECTORFORGE_LLM_DEFAULT_PROVIDER` | `openai`   | Which LLM provider by default |
| `VECTORFORGE_LLM_DEFAULT_MODEL`    | `gpt-4o`   | Default model name             |
| `VECTORFORGE_LLM_TEMPERATURE`      | `0.7`      | Sampling temperature (0.0–2.0) |
| `VECTORFORGE_LLM_MAX_TOKENS`       | `2048`     | Max tokens in LLM response     |
| `VECTORFORGE_LLM_SYSTEM_PROMPT`    | *(empty)*  | Optional system prompt override |

**Example (OpenAI):**

```env
VECTORFORGE_OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
VECTORFORGE_LLM_DEFAULT_PROVIDER=openai
VECTORFORGE_LLM_DEFAULT_MODEL=gpt-4o
VECTORFORGE_LLM_TEMPERATURE=0.7
VECTORFORGE_LLM_MAX_TOKENS=2048
```

---

### Chunking

| Variable                                  | Default      | Description                           |
|-------------------------------------------|--------------|---------------------------------------|
| `VECTORFORGE_CHUNKING_STRATEGY`           | `recursive`  | Strategy: `recursive`, `token`, `markdown`, `html`, `xml` |
| `VECTORFORGE_CHUNKING_CHUNK_SIZE`         | `1000`       | Target chunk size in characters       |
| `VECTORFORGE_CHUNKING_CHUNK_OVERLAP`      | `200`        | Overlap between chunks in characters  |
| `VECTORFORGE_CHUNKING_BREAKPOINT_THRESHOLD` | `0.5`      | Semantic chunking breakpoint threshold |

> **Rule:** `chunk_overlap` must be strictly less than `chunk_size`. VectorForge validates this at startup.

---

### Storage

| Variable                               | Default          | Description                          |
|----------------------------------------|------------------|--------------------------------------|
| `VECTORFORGE_STORAGE_DEFAULT_BACKEND`  | `pg`             | Default backend: `pg` (PostgreSQL)   |
| `VECTORFORGE_STORAGE_THRESHOLD_MB`     | `10`             | Size threshold for S3 routing (MB)   |
| `VECTORFORGE_STORAGE_S3_BUCKET`        | *(empty)*        | S3 bucket name (enables S3 backend)  |
| `VECTORFORGE_STORAGE_S3_REGION`        | `eu-central-1`   | AWS region                           |
| `VECTORFORGE_STORAGE_S3_ACCESS_KEY`    | *(empty)*        | AWS access key ID                    |
| `VECTORFORGE_STORAGE_S3_SECRET_KEY`    | *(empty)*        | AWS secret access key                |
| `VECTORFORGE_STORAGE_S3_ENDPOINT_URL`  | *(empty)*        | Custom S3 endpoint (MinIO, etc.)     |

S3 is optional. When configured, documents larger than `threshold_mb` are automatically stored in S3 instead of PostgreSQL.

---

### Monitoring

| Variable                                             | Default | Description                          |
|------------------------------------------------------|---------|--------------------------------------|
| `VECTORFORGE_MONITORING_LOG_LEVEL`                   | `INFO`  | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `VECTORFORGE_MONITORING_LOG_FORMAT`                  | `json`  | `json` or `text`                     |
| `VECTORFORGE_MONITORING_LOG_FILE`                    | *(none)* | Optional log file path              |
| `VECTORFORGE_MONITORING_METRICS_ENABLED`             | `true`  | Enable metrics collection            |
| `VECTORFORGE_MONITORING_METRICS_FLUSH_INTERVAL_SECONDS` | `60` | Metrics flush interval               |
| `VECTORFORGE_MONITORING_HEALTH_CHECK_TIMEOUT_SECONDS`   | `5`  | Health check timeout                 |

---

## 5. Run Database Migrations

VectorForge uses Alembic for database schema migrations. After configuring your `.env`:

```bash
# Apply all migrations (creates tables: collections, documents, chunks, embeddings, query_logs)
alembic upgrade head
```

To verify the tables were created:

```bash
psql -U vectorforge -d vectorforge -c "\dt"
```

Expected output:

```
         List of relations
 Schema |    Name     | Type  |    Owner
--------+-------------+-------+-------------
 public | chunks      | table | vectorforge
 public | collections | table | vectorforge
 public | documents   | table | vectorforge
 public | embeddings  | table | vectorforge
 public | query_logs  | table | vectorforge
```

---

## 6. Validate the Setup

### Validate Configuration

```bash
vectorforge config validate
```

Expected output:

```
Configuration is valid.
  Database: localhost:5432
  Embedding: voyage
  LLM: openai
```

If there are errors, VectorForge will report them:

```
Configuration errors:
  [database → port] port must be between 1 and 65535, got 99999
```

### Show Current Configuration

```bash
vectorforge config show
```

This prints the full configuration as JSON with sensitive values (passwords, API keys) redacted.

### Verify the CLI

```bash
vectorforge version
```

Expected output: `VectorForge v0.1.0`

### Test Database Connectivity

```bash
# Quick test: list collections (should return empty)
vectorforge collections list
```

Expected output: `No collections found.`

---

## 7. Optional Extras

### S3 Storage (MinIO for Local Development)

If you want to test S3 routing locally:

```bash
docker run -d \
  --name minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

Then in your `.env`:

```env
VECTORFORGE_STORAGE_S3_BUCKET=vectorforge
VECTORFORGE_STORAGE_S3_REGION=us-east-1
VECTORFORGE_STORAGE_S3_ACCESS_KEY=minioadmin
VECTORFORGE_STORAGE_S3_SECRET_KEY=minioadmin
VECTORFORGE_STORAGE_S3_ENDPOINT_URL=http://localhost:9000
```

### Re-Ranking (Cross-Encoder)

Install the rerankers extra to enable cross-encoder based re-ranking:

```bash
pip install -e ".[rerankers]"
```

This installs `sentence-transformers`, which downloads models on first use. The default cross-encoder model is loaded lazily — no additional configuration is needed.

---

## Troubleshooting

### `psycopg2` / `asyncpg` connection errors

- Verify PostgreSQL is running: `pg_isready -h localhost -p 5432`
- Check credentials: `psql -U vectorforge -d vectorforge -c "SELECT 1"`
- Ensure `VECTORFORGE_DB_PASSWORD` matches the PostgreSQL user password

### `CREATE EXTENSION vector` fails

- pgvector is not installed. See [Step 1](#1-install-postgresql-and-pgvector).
- On managed databases (RDS, Cloud SQL), enable pgvector through the provider's console.

### `No embedding providers discovered` warning

- No API key environment variables are set. Set at least one of:
  - `VECTORFORGE_VOYAGE_API_KEY`
  - `VECTORFORGE_COHERE_API_KEY`
  - `VECTORFORGE_LITELLM_API_KEY`

### `No LLM providers discovered — generation disabled`

- This is a warning, not an error. VectorForge still works for ingestion and retrieval.
- To enable RAG query generation, set at least one LLM API key (`VECTORFORGE_OPENAI_API_KEY`, `VECTORFORGE_ANTHROPIC_API_KEY`, or `VECTORFORGE_LITELLM_API_KEY`).

### Embedding dimension mismatch

- Error: `expected X dimensions, got Y`
- The `VECTORFORGE_EMBEDDING_DIMENSIONS` value doesn't match your model's output.
- Common dimensions: OpenAI `text-embedding-3-small` = 1536, Voyage `voyage-3` = 1024, Cohere `embed-v4.0` = 1024.

### `alembic upgrade head` fails

- Ensure your `.env` file is in the project root (same directory as `alembic.ini`).
- Verify the database exists and the user has `CREATE TABLE` permissions.
- Check `alembic.ini` for the correct `script_location`.

### Import errors after installation

- Ensure the virtual environment is activated.
- Re-run `pip install -e ".[dev]"` to pick up any new dependencies.
- Check Python version: `python --version` (must be 3.11+).

# VectorForge — CLI Reference

> Complete reference for the `vectorforge` command-line interface.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Global Options](#global-options)
- [Commands](#commands)
  - [version](#version)
  - [config show](#config-show)
  - [config validate](#config-validate)
  - [collections list](#collections-list)
  - [collections get](#collections-get)
  - [collections create](#collections-create)
  - [collections delete](#collections-delete)
  - [query run](#query-run)
- [Environment Variables](#environment-variables)
- [Exit Codes](#exit-codes)
- [Examples](#examples)

---

## Overview

VectorForge ships a CLI built with [Typer](https://typer.tiangolo.com/). The entry point is registered as the `vectorforge` command when the package is installed.

```
vectorforge [OPTIONS] COMMAND [ARGS]...
```

The CLI is organised into **command groups**:

| Group          | Purpose                        |
|----------------|--------------------------------|
| `config`       | View and validate configuration |
| `collections`  | Manage document collections    |
| `query`        | Execute RAG queries            |
| *(root)*       | `version` command              |

All commands that access the database read configuration from environment variables (or a `.env` file). See the [Setup Guide](setup-guide.md) for configuration details.

---

## Installation

The CLI is available after installing VectorForge:

```bash
pip install -e .
```

Verify:

```bash
vectorforge --help
```

---

## Global Options

| Option            | Short | Type | Description                |
|-------------------|-------|------|----------------------------|
| `--verbose`       | `-v`  | Flag | Enable debug-level logging to stderr |
| `--help`          |       | Flag | Show help and exit         |

Debug logging prints timestamped log lines to stderr, useful for diagnosing connection or provider issues:

```bash
vectorforge --verbose collections list
```

---

## Commands

### `version`

Print the installed VectorForge version.

```bash
vectorforge version
```

**Output:**

```
VectorForge v0.1.0
```

**Options:** None.

---

### `config show`

Display the current configuration loaded from environment variables, with sensitive values redacted.

```bash
vectorforge config show
```

**Output:** JSON document with the full configuration tree. Sensitive fields (`password`, `api_key`, `secret`, `token`) are replaced with `***REDACTED***`.

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "vectorforge",
    "user": "vectorforge",
    "password": "***REDACTED***",
    "pool_size": 10,
    "max_overflow": 5,
    "echo_sql": false
  },
  "embedding": {
    "default_provider": "voyage",
    "default_model": "voyage-3",
    "dimensions": 1024,
    "batch_size": 100
  },
  "llm": {
    "default_provider": "openai",
    "default_model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2048,
    "system_prompt": ""
  },
  ...
}
```

**Options:** None.

---

### `config validate`

Validate the configuration and report any errors. This checks all environment variables against their expected types, ranges, and constraints.

```bash
vectorforge config validate
```

**Output (success):**

```
Configuration is valid.
  Database: localhost:5432
  Embedding: voyage
  LLM: openai
```

**Output (errors):**

```
Configuration errors:
  [database → port] port must be between 1 and 65535, got 99999
  [chunking] chunk_overlap (500) must be < chunk_size (200)
```

**Exit code:** `0` on success, `1` on validation failure.

**Options:** None.

---

### `collections list`

List all document collections.

```bash
vectorforge collections list [OPTIONS]
```

**Options:**

| Option     | Short | Type | Default | Description               |
|------------|-------|------|---------|---------------------------|
| `--limit`  | `-n`  | int  | `20`    | Maximum number of results |
| `--offset` |       | int  | `0`     | Pagination offset         |

**Output:**

```
  a1b2c3d4-...  research-papers  (ML research collection)
  e5f6g7h8-...  legal-docs       (Contract analysis)
```

If no collections exist:

```
No collections found.
```

**Examples:**

```bash
# List first 5 collections
vectorforge collections list --limit 5

# Paginate: skip the first 20, show next 20
vectorforge collections list --offset 20 --limit 20
```

---

### `collections get`

Get full details of a single collection by its UUID.

```bash
vectorforge collections get COLLECTION_ID
```

**Arguments:**

| Argument        | Type   | Required | Description      |
|-----------------|--------|----------|------------------|
| `COLLECTION_ID` | string | Yes      | Collection UUID  |

**Output:** JSON document with all collection fields:

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "research-papers",
  "description": "ML research collection",
  "embedding_config": null,
  "chunking_config": null,
  "created_at": "2026-03-14T10:30:00",
  "updated_at": null
}
```

**Error cases:**

| Condition         | Output                          | Exit Code |
|-------------------|---------------------------------|-----------|
| Invalid UUID      | `Invalid UUID: <value>`         | 1         |
| Not found         | `Collection <id> not found.`    | 1         |

**Examples:**

```bash
vectorforge collections get a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

### `collections create`

Create a new document collection.

```bash
vectorforge collections create NAME [OPTIONS]
```

**Arguments:**

| Argument | Type   | Required | Description      |
|----------|--------|----------|------------------|
| `NAME`   | string | Yes      | Collection name  |

**Options:**

| Option          | Short | Type   | Default | Description              |
|-----------------|-------|--------|---------|--------------------------|
| `--description` | `-d`  | string | `""`    | Collection description   |

**Output:**

```
Created collection: a1b2c3d4-e5f6-7890-abcd-ef1234567890 (research-papers)
```

**Examples:**

```bash
# Create with just a name
vectorforge collections create my-docs

# Create with a description
vectorforge collections create research-papers -d "ML research paper collection"
```

---

### `collections delete`

Delete a collection by its UUID. Prompts for confirmation unless `--force` is used.

```bash
vectorforge collections delete COLLECTION_ID [OPTIONS]
```

**Arguments:**

| Argument        | Type   | Required | Description      |
|-----------------|--------|----------|------------------|
| `COLLECTION_ID` | string | Yes      | Collection UUID  |

**Options:**

| Option    | Short | Type | Default | Description           |
|-----------|-------|------|---------|-----------------------|
| `--force` | `-f`  | Flag | `false` | Skip confirmation     |

**Confirmation prompt (without `--force`):**

```
Delete collection a1b2c3d4-...? [y/N]:
```

If declined:

```
Aborted.
```

If confirmed or `--force`:

```
Deleted collection: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Error cases:**

| Condition    | Output                    | Exit Code |
|--------------|---------------------------|-----------|
| Invalid UUID | `Invalid UUID: <value>`   | 1         |

**Examples:**

```bash
# Delete with confirmation prompt
vectorforge collections delete a1b2c3d4-e5f6-7890-abcd-ef1234567890

# Delete without confirmation
vectorforge collections delete a1b2c3d4-e5f6-7890-abcd-ef1234567890 --force
```

---

### `query run`

Execute a RAG query against a collection. Retrieves relevant chunks, assembles context, and generates an answer using the configured LLM.

```bash
vectorforge query run COLLECTION_ID QUESTION [OPTIONS]
```

**Arguments:**

| Argument        | Type   | Required | Description                    |
|-----------------|--------|----------|--------------------------------|
| `COLLECTION_ID` | string | Yes      | Collection UUID to query       |
| `QUESTION`      | string | Yes      | The question or query text     |

**Options:**

| Option        | Short | Type   | Default | Description                          |
|---------------|-------|--------|---------|--------------------------------------|
| `--top-k`     | `-k`  | int    | `10`    | Number of chunks to retrieve         |
| `--min-score`  |       | float  | `0.0`   | Minimum similarity score threshold   |
| `--llm`        |       | string | *(config default)* | Override LLM provider name  |
| `--model`      |       | string | *(config default)* | Override LLM model name     |
| `--sources`    | `-s`  | Flag   | `false` | Show source chunks in output         |

**Output:**

```
Answer:
Retrieval-Augmented Generation (RAG) combines document retrieval with
language model generation to produce grounded, factual answers...

Latency: retrieval=45ms, generation=1200ms, total=1245ms
```

With `--sources`:

```
Answer:
Retrieval-Augmented Generation (RAG) combines document retrieval with...

Latency: retrieval=45ms, generation=1200ms, total=1245ms

Sources:
  - research-papers/rag-survey.pdf (chunk 3)
  - research-papers/rag-survey.pdf (chunk 7)
  - research-papers/neural-retrieval.pdf (chunk 1)
```

**Error cases:**

| Condition             | Output                            | Exit Code |
|-----------------------|-----------------------------------|-----------|
| Invalid UUID          | `Invalid UUID: <value>`           | 1         |
| Collection not found  | `Collection <id> not found.`      | 1         |
| No LLM configured    | Error from LLM registry           | 1         |
| No embedding provider | Error from embedding registry     | 1         |

**Examples:**

```bash
# Basic query
vectorforge query run a1b2c3d4-... "What is retrieval-augmented generation?"

# Retrieve more chunks with higher quality threshold
vectorforge query run a1b2c3d4-... "Explain HNSW indexing" --top-k 20 --min-score 0.5

# Use a specific LLM provider and model
vectorforge query run a1b2c3d4-... "Summarize the key findings" --llm anthropic --model claude-sonnet-4-20250514

# Show source citations
vectorforge query run a1b2c3d4-... "What are the main results?" --sources
```

---

## Environment Variables

The CLI reads configuration from environment variables. A `.env` file in the project root is loaded automatically. For the complete list, see the [Setup Guide](setup-guide.md#4-configure-environment-variables).

**Key variables for CLI usage:**

| Variable                             | Purpose                                |
|--------------------------------------|----------------------------------------|
| `VECTORFORGE_DB_HOST`                | Database host                          |
| `VECTORFORGE_DB_PORT`                | Database port                          |
| `VECTORFORGE_DB_DATABASE`            | Database name                          |
| `VECTORFORGE_DB_USER`                | Database user                          |
| `VECTORFORGE_DB_PASSWORD`            | Database password                      |
| `VECTORFORGE_VOYAGE_API_KEY`         | Voyage embedding provider              |
| `VECTORFORGE_COHERE_API_KEY`         | Cohere embedding provider              |
| `VECTORFORGE_LITELLM_API_KEY`        | LiteLLM embedding/LLM provider         |
| `VECTORFORGE_OPENAI_API_KEY`         | OpenAI LLM provider                   |
| `VECTORFORGE_ANTHROPIC_API_KEY`      | Anthropic LLM provider                 |
| `VECTORFORGE_LLM_DEFAULT_PROVIDER`   | Default LLM for `query run`            |
| `VECTORFORGE_LLM_DEFAULT_MODEL`      | Default LLM model for `query run`      |

---

## Exit Codes

| Code | Meaning                              |
|------|--------------------------------------|
| `0`  | Success                              |
| `1`  | Error (invalid input, not found, config error) |

---

## Examples

### Full Workflow: Create, Query, and Delete

```bash
# 1. Validate your setup
vectorforge config validate

# 2. Create a collection
vectorforge collections create my-project -d "Project documentation"

# (Note the UUID from the output, e.g., a1b2c3d4-...)
# 3. Ingest documents into the collection (via Python SDK — see API docs)

# 4. Query the collection
vectorforge query run a1b2c3d4-... "How does authentication work?" --sources

# 5. List all collections
vectorforge collections list

# 6. View collection details
vectorforge collections get a1b2c3d4-...

# 7. Clean up
vectorforge collections delete a1b2c3d4-... --force
```

### Debugging with Verbose Mode

```bash
# See debug logs for connection issues
vectorforge --verbose config validate

# Debug a failing query
vectorforge --verbose query run a1b2c3d4-... "test query"
```

### Scripting with JSON Output

```bash
# Parse collection details with jq
vectorforge collections get a1b2c3d4-... | jq '.name'

# Get config as JSON
vectorforge config show | jq '.database.host'
```

# VectorForge — CLI Testing Guide

> How to verify the CLI is working correctly after setup.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Smoke Tests (No Database Required)](#1-smoke-tests-no-database-required)
- [2. Configuration Tests](#2-configuration-tests)
- [3. Database Connectivity Tests](#3-database-connectivity-tests)
- [4. Collections CRUD Tests](#4-collections-crud-tests)
- [5. Query Pipeline Test](#5-query-pipeline-test)
- [6. Running the Automated Test Suite](#6-running-the-automated-test-suite)
- [7. Understanding the Test Architecture](#7-understanding-the-test-architecture)
- [8. Writing New CLI Tests](#8-writing-new-cli-tests)
- [9. Troubleshooting Test Failures](#9-troubleshooting-test-failures)

---

## Prerequisites

Before testing, ensure you have completed the [Setup Guide](setup-guide.md):

- Python 3.11+ with virtual environment activated
- VectorForge installed with dev dependencies: `pip install -e ".[dev]"`
- PostgreSQL running with pgvector extension enabled
- `.env` file configured with at least database credentials and one embedding provider API key

---

## 1. Smoke Tests (No Database Required)

These tests verify the CLI is installed and responds correctly. They do **not** require a database connection.

### Test: CLI is installed

```bash
vectorforge --help
```

**Expected:** Help text showing available commands (`collections`, `config`, `query`, `version`).

### Test: Version output

```bash
vectorforge version
```

**Expected:** `VectorForge v0.1.0` (or current version).

### Test: Sub-command help

```bash
vectorforge collections --help
vectorforge config --help
vectorforge query --help
```

**Expected:** Each shows its available sub-commands and options.

### Test: Verbose flag

```bash
vectorforge --verbose version
```

**Expected:** Version output. Debug-level log lines may appear on stderr.

---

## 2. Configuration Tests

These tests verify environment variables are loaded and validated correctly.

### Test: Show configuration

```bash
vectorforge config show
```

**Expected:**
- JSON output with all configuration sections (`database`, `storage`, `embedding`, `chunking`, `llm`, `monitoring`).
- Sensitive values (`password`, `api_key`, `secret`, `token`) show `***REDACTED***`.

**Verify redaction works:**

```bash
vectorforge config show | python -c "
import json, sys
data = json.load(sys.stdin)
pwd = data.get('database', {}).get('password', '')
assert pwd == '***REDACTED***' or pwd == '', f'Password not redacted: {pwd}'
print('Redaction: OK')
"
```

### Test: Validate configuration (success)

```bash
vectorforge config validate
```

**Expected:**

```
Configuration is valid.
  Database: localhost:5432
  Embedding: voyage
  LLM: openai
```

Exit code: `0`.

### Test: Validate configuration (failure)

Temporarily set an invalid value:

```bash
# PowerShell:
$env:VECTORFORGE_DB_PORT = "99999"
vectorforge config validate
Remove-Item Env:VECTORFORGE_DB_PORT

# Bash:
VECTORFORGE_DB_PORT=99999 vectorforge config validate
```

**Expected:**

```
Configuration errors:
  [database → port] port must be between 1 and 65535, got 99999
```

Exit code: `1`.

---

## 3. Database Connectivity Tests

These tests verify the CLI can connect to PostgreSQL.

### Test: List collections (empty database)

```bash
vectorforge collections list
```

**Expected:** `No collections found.` (on a fresh database) or a list of existing collections. Exit code: `0`.

### Test: Connection failure

Temporarily point to a non-existent host:

```bash
# PowerShell:
$env:VECTORFORGE_DB_HOST = "nonexistent-host"
vectorforge collections list
Remove-Item Env:VECTORFORGE_DB_HOST

# Bash:
VECTORFORGE_DB_HOST=nonexistent-host vectorforge collections list
```

**Expected:** A connection error. Exit code: non-zero.

---

## 4. Collections CRUD Tests

Walk through the full Create → Read → List → Delete lifecycle.

### Step 1: Create a collection

```bash
vectorforge collections create test-collection -d "Testing the CLI"
```

**Expected:**

```
Created collection: <UUID> (test-collection)
```

Save the UUID for subsequent steps:

```bash
# PowerShell:
$COLL_ID = (vectorforge collections create test-cli-guide -d "Guide test" | Select-String -Pattern '[0-9a-f-]{36}' | ForEach-Object { $_.Matches[0].Value })
echo $COLL_ID

# Bash:
COLL_ID=$(vectorforge collections create test-cli-guide -d "Guide test" | grep -oP '[0-9a-f-]{36}')
echo $COLL_ID
```

### Step 2: Get collection details

```bash
vectorforge collections get $COLL_ID
```

**Expected:** JSON output containing:
- `"name": "test-cli-guide"`
- `"description": "Guide test"`
- A valid `id`, `created_at` field

### Step 3: List collections

```bash
vectorforge collections list
```

**Expected:** The newly created collection appears in the list.

### Step 4: Test pagination

```bash
vectorforge collections list --limit 1
vectorforge collections list --limit 1 --offset 1
```

**Expected:** Different collections returned (if more than one exists).

### Step 5: Delete with confirmation

```bash
vectorforge collections delete $COLL_ID
```

**Expected:** Confirmation prompt: `Delete collection <UUID>? [y/N]:`

- Type `N` → Output: `Aborted.`
- Type `y` → Output: `Deleted collection: <UUID>`

### Step 6: Delete with force

```bash
# Create another test collection
vectorforge collections create temp-delete-test
# Get the UUID from output, then:
vectorforge collections delete <UUID> --force
```

**Expected:** `Deleted collection: <UUID>` (no prompt).

### Step 7: Verify deletion

```bash
vectorforge collections get <deleted-UUID>
```

**Expected:** `Collection <UUID> not found.` Exit code: `1`.

### Error Cases to Verify

```bash
# Invalid UUID format
vectorforge collections get not-a-uuid
# Expected: "Invalid UUID: not-a-uuid", exit code 1

vectorforge collections delete not-a-uuid
# Expected: "Invalid UUID: not-a-uuid", exit code 1

# Non-existent UUID
vectorforge collections get 00000000-0000-0000-0000-000000000000
# Expected: "Collection 00000000-... not found.", exit code 1
```

---

## 5. Query Pipeline Test

This requires:
- A collection with ingested documents (chunks + embeddings already in the database)
- At least one embedding provider API key configured
- At least one LLM provider API key configured

### Test: Query with missing collection

```bash
vectorforge query run 00000000-0000-0000-0000-000000000000 "test question"
```

**Expected:** `Collection 00000000-0000-0000-0000-000000000000 not found.` Exit code: `1`.

### Test: Query with invalid UUID

```bash
vectorforge query run bad-uuid "test question"
```

**Expected:** `Invalid UUID: bad-uuid` Exit code: `1`.

### Test: Full RAG query (requires ingested data)

```bash
vectorforge query run <COLLECTION_UUID> "What is the main topic?" --top-k 5 --sources
```

**Expected:**

```
Answer:
<LLM-generated answer based on retrieved context>

Latency: retrieval=XXms, generation=XXXms, total=XXXms

Sources:
  - <document_source> (chunk N)
  - ...
```

### Test: Override LLM provider

```bash
vectorforge query run <COLLECTION_UUID> "Summarize" --llm anthropic --model claude-sonnet-4-20250514
```

**Expected:** Answer generated by the specified provider.

---

## 6. Running the Automated Test Suite

VectorForge includes a comprehensive automated test suite for the CLI. These tests use mocked dependencies and do **not** require a live database or API keys.

### Run all tests

```bash
python -m pytest tests/ -v
```

### Run only CLI tests

```bash
python -m pytest tests/unit/test_cli.py -v
```

**Expected output:**

```
tests/unit/test_cli.py::TestVersionCommand::test_version_output PASSED
tests/unit/test_cli.py::TestConfigCommands::test_show_config PASSED
tests/unit/test_cli.py::TestConfigCommands::test_validate_config_success PASSED
tests/unit/test_cli.py::TestCollectionsCommands::test_list_calls_run_async PASSED
tests/unit/test_cli.py::TestCollectionsCommands::test_get_with_valid_uuid PASSED
tests/unit/test_cli.py::TestCollectionsCommands::test_get_with_invalid_uuid PASSED
tests/unit/test_cli.py::TestCollectionsCommands::test_create PASSED
tests/unit/test_cli.py::TestCollectionsCommands::test_delete_invalid_uuid PASSED
tests/unit/test_cli.py::TestCollectionsCommands::test_delete_with_force PASSED
tests/unit/test_cli.py::TestQueryCommands::test_run_invalid_uuid PASSED
tests/unit/test_cli.py::TestQueryCommands::test_run_valid PASSED
tests/unit/test_cli.py::TestVerboseFlag::test_verbose_sets_debug PASSED
tests/unit/test_cli.py::TestRunAsync::test_run_async_simple_coro PASSED
```

### Run with coverage

```bash
python -m pytest tests/unit/test_cli.py --cov=vectorforge.cli --cov-report=term-missing
```

This shows which lines of CLI code are covered by tests and which are not.

### Run a specific test

```bash
# By test class
python -m pytest tests/unit/test_cli.py::TestCollectionsCommands -v

# By specific test method
python -m pytest tests/unit/test_cli.py::TestCollectionsCommands::test_get_with_invalid_uuid -v
```

---

## 7. Understanding the Test Architecture

The CLI tests in `tests/unit/test_cli.py` use Typer's `CliRunner` to invoke commands in-process without spawning a shell. Dependencies are mocked to isolate the CLI layer from the database and external APIs.

### Key patterns

**CliRunner invocation:**

```python
import typer.testing
from vectorforge.cli.main import app

runner = typer.testing.CliRunner()

def test_example():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "VectorForge v" in result.output
```

**Mocking async functions:**

The CLI wraps async operations with `run_async()`. Tests mock the async inner function and `run_async` to avoid database calls:

```python
from unittest.mock import MagicMock, patch

@patch("vectorforge.cli.collections._list_collections")
@patch("vectorforge.cli.collections.run_async")
def test_list(mock_run, mock_fn):
    mock_run.return_value = None
    result = runner.invoke(app, ["collections", "list"])
    assert result.exit_code == 0
    mock_run.assert_called_once()
```

**Mocking configuration:**

```python
@patch("vectorforge.config.settings.load_config")
def test_config(mock_load):
    mock_config = MagicMock()
    mock_config.model_dump.return_value = {"database": {"host": "localhost"}}
    mock_load.return_value = mock_config
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
```

### Test classes

| Class                      | Tests                                    |
|----------------------------|------------------------------------------|
| `TestVersionCommand`       | Version output                           |
| `TestConfigCommands`       | `config show`, `config validate`         |
| `TestCollectionsCommands`  | `list`, `get`, `create`, `delete`, UUID validation |
| `TestQueryCommands`        | `query run`, UUID validation             |
| `TestVerboseFlag`          | `--verbose` flag sets logging level      |
| `TestRunAsync`             | `run_async` helper executes coroutines   |

---

## 8. Writing New CLI Tests

When adding new CLI commands or modifying existing ones, follow these patterns:

### Template for a new command test

```python
class TestNewCommand:
    """Tests for the 'new-group sub-command' command."""

    @patch("vectorforge.cli.module._async_function")
    @patch("vectorforge.cli.module.run_async")
    def test_happy_path(self, mock_run: MagicMock, mock_fn: MagicMock) -> None:
        mock_run.return_value = None
        result = runner.invoke(app, ["group", "command", "arg1"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_invalid_input(self) -> None:
        result = runner.invoke(app, ["group", "command", "bad-input"])
        assert result.exit_code == 1
        assert "error message" in result.output
```

### Rules for CLI tests

1. **Always use `typer.testing.CliRunner`** — never invoke the CLI via subprocess.
2. **Mock `run_async` and the async inner function** — this prevents actual database access.
3. **Test exit codes** — `0` for success, `1` for errors.
4. **Test error messages** — verify the user sees helpful output.
5. **Test edge cases** — invalid UUIDs, missing arguments, empty results.
6. **Keep tests fast** — all CLI tests should run in under 1 second total.

### Running after changes

```bash
# Run CLI tests
python -m pytest tests/unit/test_cli.py -v

# Run linter
python -m ruff check vectorforge/cli/

# Run type checker
python -m mypy vectorforge/cli/

# Run full suite to check for regressions
python -m pytest tests/ -v
```

---

## 9. Troubleshooting Test Failures

### `ModuleNotFoundError: No module named 'vectorforge'`

The package isn't installed in the virtual environment:

```bash
pip install -e ".[dev]"
```

### `RuntimeWarning: coroutine was never awaited`

This is a known warning from `unittest.mock.AsyncMock` and does not indicate a test failure. It appears when mock objects create coroutines that aren't consumed. The tests still pass correctly.

### Tests pass but live CLI fails

The automated tests mock all external dependencies. If tests pass but the live CLI fails:

1. Check database connectivity: `vectorforge config validate`
2. Check API keys are set: `vectorforge config show` (look for non-empty provider sections)
3. Run with verbose mode: `vectorforge --verbose <command>`
4. Check PostgreSQL logs for connection or query errors

### `typer.Exit` raised in tests

`typer.Exit` is the normal mechanism for non-zero exit codes. The `CliRunner` captures these — check `result.exit_code` rather than catching exceptions.

### Coverage gaps

To identify untested CLI paths:

```bash
python -m pytest tests/unit/test_cli.py --cov=vectorforge.cli --cov-report=term-missing
```

Lines shown as "missing" in the report need additional tests. Focus on:
- Error branches (exception handlers, validation failures)
- Edge cases (empty results, boundary values)
- New commands added without corresponding tests

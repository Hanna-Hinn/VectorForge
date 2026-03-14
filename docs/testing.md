# VectorForge — Testing Guide

> Comprehensive guide covering the Python backend test suite, frontend unit tests with Jest, and end-to-end tests with Playwright.

---

## Table of Contents

- [Overview](#overview)
- [Backend Testing (pytest)](#backend-testing-pytest)
  - [Running Tests](#running-tests)
  - [Test Architecture](#test-architecture)
  - [Fixtures](#fixtures)
  - [Mocking Patterns](#mocking-patterns)
  - [Writing New Tests](#writing-new-tests)
- [Server Integration Tests](#server-integration-tests)
  - [Test Setup](#test-setup)
  - [Testing Endpoints](#testing-endpoints)
  - [Testing Auth](#testing-auth)
  - [Testing Middleware](#testing-middleware)
- [Frontend Unit Tests (Jest)](#frontend-unit-tests-jest)
  - [Running Jest](#running-jest)
  - [Test Structure](#test-structure)
  - [Testing Utilities](#testing-utilities)
  - [Testing Hooks](#testing-hooks)
  - [Testing Components](#testing-components)
  - [Testing API Client](#testing-api-client)
- [End-to-End Tests (Playwright)](#end-to-end-tests-playwright)
  - [Running Playwright](#running-playwright)
  - [Test Structure](#e2e-test-structure)
  - [Mock API Setup](#mock-api-setup)
  - [Writing Page Tests](#writing-page-tests)
- [Coverage](#coverage)
- [CI Integration](#ci-integration)

---

## Overview

VectorForge uses three testing layers:

| Layer | Tool | Location | Scope |
|-------|------|----------|-------|
| **Backend unit** | pytest | `tests/unit/` | Core library modules |
| **Backend integration** | pytest + TestClient | `tests/integration/` | API routes, middleware, auth |
| **Frontend unit** | Jest + React Testing Library | `frontend/src/__tests__/` | Components, hooks, utilities |
| **Frontend E2E** | Playwright | `frontend/e2e/` | Full page interactions |

---

## Backend Testing (pytest)

### Running Tests

```bash
# Activate virtual environment first
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux/macOS

# Run full suite
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage
python -m pytest --cov=vectorforge --cov=server --cov-report=term-missing

# Run specific test file
python -m pytest tests/unit/test_chunking.py

# Run specific test class or function
python -m pytest tests/unit/test_chunking.py::TestRecursiveChunker -v
python -m pytest tests/integration/test_api.py::TestCollectionRoutes::test_create_collection -v

# Run only server integration tests
python -m pytest tests/integration/test_api.py -v
```

### Test Architecture

```
tests/
├── conftest.py                    # Shared fixtures (config, domain DTOs, sessions)
├── fixtures/
│   ├── mock_embeddings.py         # Embedding test data
│   └── sample_documents/          # Sample .txt, .md, .html files
├── integration/
│   ├── test_api.py                # REST API route tests
│   └── test_rag_pipeline.py       # End-to-end pipeline tests
└── unit/
    ├── test_analytics.py
    ├── test_chunking.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_context.py
    ├── test_embedding.py
    ├── test_hybrid.py
    ├── test_ingestion.py
    ├── test_llm.py
    ├── test_loaders.py
    ├── test_models.py
    ├── test_monitoring.py
    ├── test_rag_pipeline.py
    ├── test_repositories.py
    ├── test_reranker.py
    ├── test_retriever.py
    └── test_storage.py
```

### Fixtures

Shared fixtures in `tests/conftest.py`:

| Fixture | Type | Description |
|---------|------|-------------|
| `mock_config` | `VectorForgeConfig` | Config with test values + env vars set via `monkeypatch` |
| `sample_collection` | `CreateCollectionDTO` | Collection creation DTO |
| `sample_document` | `CreateDocumentDTO` | Document creation DTO |
| `sample_chunks` | `list[CreateChunkDTO]` | Three test chunks |
| `sample_embedding_dto` | `CreateEmbeddingDTO` | 4-dimensional test embedding |
| `mock_async_session` | `AsyncMock` | Mocked SQLAlchemy `AsyncSession` |
| `_reset_metrics` | autouse | Resets the metrics collector before/after each test |

### Mocking Patterns

**Mock a repository in route tests:**

```python
from unittest.mock import AsyncMock, patch

def test_list_items(client: TestClient) -> None:
    with patch("server.routes.collections.CollectionRepository") as MockRepo:
        MockRepo.return_value.find_all = AsyncMock(return_value=[])
        resp = client.get("/api/collections")
    assert resp.status_code == 200
```

**Mock an async database session:**

```python
@pytest.fixture()
def mock_async_session() -> AsyncMock:
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=mock_result)
    return session
```

**Mock external services (embedding/LLM providers):**

```python
@pytest.fixture()
def mock_embedder() -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    embedder.dimensions = 4
    return embedder
```

### Writing New Tests

1. **Choose the right directory**: `tests/unit/` for isolated logic, `tests/integration/` for cross-module tests
2. **Follow the naming pattern**: `test_{module}.py` → `class Test{Feature}` → `def test_{scenario}`
3. **Use fixtures** from `conftest.py` — don't recreate domain objects in every test
4. **Mock at boundaries** — repositories, external APIs, database sessions
5. **Assert specific outcomes** — status codes, response shapes, side effects

---

## Server Integration Tests

### Test Setup

Server tests in `tests/integration/test_api.py` use FastAPI's `TestClient`:

```python
from fastapi.testclient import TestClient

@pytest.fixture()
def client() -> TestClient:
    app = _mock_app()          # Creates FastAPI with mocked lifespan state
    with TestClient(app) as c:
        yield c
```

The `_mock_app()` function:
- Creates a FastAPI app with mocked registries, DB engine, and config
- Wires all route routers under `/api`
- Adds middleware (error handler, request logging, CORS)
- Sets `auth_required=False` by default (separate auth test class)

### Testing Endpoints

Each endpoint group has its own test class:

| Class | Endpoint Group | Tests |
|-------|---------------|-------|
| `TestCollectionRoutes` | `/api/collections` | List, create, get, delete (success + error) |
| `TestDocumentRoutes` | `/api/documents` | List, get, delete (success + not found) |
| `TestStatusRoutes` | `/api/status` | Health check, provider listing |
| `TestAnalyticsRoutes` | `/api/analytics` | Summary, top queries, latency |
| `TestAuth` | Auth layer | Missing key, wrong key, correct key |
| `TestMiddleware` | Middleware | Request IDs, error mapping |

### Testing Auth

The auth tests use a separate `_make_auth_app()` that sets `auth_required=True`:

```python
def test_missing_api_key(self) -> None:
    with self._make_auth_app() as client:
        resp = client.get("/api/collections")
    assert resp.status_code == 401

def test_correct_api_key(self) -> None:
    with self._make_auth_app() as client:
        resp = client.get(
            "/api/collections",
            headers={"X-Api-Key": "test-secret"},
        )
    assert resp.status_code == 200
```

### Testing Middleware

```python
def test_request_id_header(self, client: TestClient) -> None:
    resp = client.get("/api/collections")
    assert "x-request-id" in resp.headers

def test_error_handler_catches_vectorforge_errors(self, client: TestClient) -> None:
    # Force a NotFoundError — middleware should convert to 404
    with patch(...) as MockRepo:
        MockRepo.return_value.find_by_id = AsyncMock(
            side_effect=NotFoundError("not found")
        )
        resp = client.get(f"/api/collections/{uuid.uuid4()}")
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"
```

---

## Frontend Unit Tests (Jest)

### Running Jest

```bash
cd frontend

# Run all unit tests
npm test

# Run in watch mode
npm run test:watch

# Run with coverage
npm run test:coverage

# Run specific file
npx jest src/__tests__/utils/format.test.ts
```

### Test Structure

```
frontend/src/__tests__/
├── utils/
│   └── format.test.ts           # formatDate, formatNumber, formatBytes, formatMs
├── api/
│   └── client.test.ts           # HTTP client (fetch mocking)
├── hooks/
│   ├── useCollections.test.tsx   # Collection hooks
│   └── useAnalytics.test.tsx    # Analytics hooks
└── components/
    ├── ui/
    │   ├── Button.test.tsx
    │   └── Badge.test.tsx
    └── features/
        └── CollectionCard.test.tsx
```

### Testing Utilities

Pure function tests require no React setup:

```typescript
import { formatBytes, formatMs } from "@/utils/format";

describe("formatBytes", () => {
  it("formats bytes", () => {
    expect(formatBytes(500)).toBe("500 B");
  });
  it("formats kilobytes", () => {
    expect(formatBytes(2048)).toBe("2.0 KB");
  });
  it("formats megabytes", () => {
    expect(formatBytes(1048576)).toBe("1.0 MB");
  });
});
```

### Testing Hooks

Use `@testing-library/react` with a `QueryClientProvider` wrapper:

```typescript
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

function createWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

it("fetches collections", async () => {
  vi.spyOn(global, "fetch").mockResolvedValueOnce(
    new Response(JSON.stringify({ collections: [] }), { status: 200 }),
  );

  const { result } = renderHook(() => useCollections(), {
    wrapper: createWrapper(),
  });

  await waitFor(() => expect(result.current.isSuccess).toBe(true));
  expect(result.current.data).toEqual([]);
});
```

### Testing Components

```typescript
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Button from "@/components/ui/Button";

it("renders with text", () => {
  render(<Button>Click me</Button>);
  expect(screen.getByRole("button", { name: "Click me" })).toBeInTheDocument();
});

it("shows spinner when loading", () => {
  render(<Button loading>Save</Button>);
  expect(screen.getByRole("button")).toBeDisabled();
});

it("calls onClick", async () => {
  const onClick = vi.fn();
  render(<Button onClick={onClick}>Go</Button>);
  await userEvent.click(screen.getByRole("button"));
  expect(onClick).toHaveBeenCalledTimes(1);
});
```

### Testing API Client

Mock `fetch` globally to test the API layer:

```typescript
import { get, post, setApiKey } from "@/api/client";

beforeEach(() => {
  vi.restoreAllMocks();
  setApiKey(null);
});

it("sends GET request with JSON content type", async () => {
  vi.spyOn(global, "fetch").mockResolvedValueOnce(
    new Response(JSON.stringify({ ok: true }), { status: 200 }),
  );
  const result = await get("/test");
  expect(result).toEqual({ ok: true });
});

it("includes API key header when set", async () => {
  setApiKey("my-key");
  vi.spyOn(global, "fetch").mockResolvedValueOnce(
    new Response(JSON.stringify({}), { status: 200 }),
  );
  await get("/test");
  expect(fetch).toHaveBeenCalledWith(
    expect.any(String),
    expect.objectContaining({
      headers: expect.objectContaining({ "X-Api-Key": "my-key" }),
    }),
  );
});
```

---

## End-to-End Tests (Playwright)

### Running Playwright

```bash
cd frontend

# Run all E2E tests
npx playwright test

# Run with browser visible
npx playwright test --headed

# Run specific test file
npx playwright test e2e/collections.spec.ts

# Open Playwright test report
npx playwright show-report
```

### E2E Test Structure

```
frontend/e2e/
├── collections.spec.ts    # Collections CRUD flows
├── documents.spec.ts      # Document ingestion flows
├── query.spec.ts          # RAG query flows
├── analytics.spec.ts      # Analytics dashboard
├── settings.spec.ts       # Settings/health page
└── fixtures/
    └── mock-api.ts        # Route handler mocks
```

### Mock API Setup

E2E tests mock the `/api` routes using Playwright's `page.route()`:

```typescript
import { test, expect } from "@playwright/test";

const MOCK_COLLECTIONS = [
  {
    id: "11111111-1111-1111-1111-111111111111",
    name: "test-docs",
    description: "Test collection",
    embedding_config: { default_provider: "voyage" },
    chunking_config: null,
    created_at: "2025-01-01T00:00:00Z",
    updated_at: null,
  },
];

test.beforeEach(async ({ page }) => {
  // Mock collections endpoint
  await page.route("**/api/collections", (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ collections: MOCK_COLLECTIONS }),
      });
    }
    return route.continue();
  });
});
```

### Writing Page Tests

```typescript
test("displays collections list", async ({ page }) => {
  await page.goto("/collections");
  await expect(page.getByText("test-docs")).toBeVisible();
});

test("creates a new collection", async ({ page }) => {
  await page.route("**/api/collections", async (route) => {
    if (route.request().method() === "POST") {
      const body = route.request().postDataJSON();
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          id: "22222222-2222-2222-2222-222222222222",
          name: body.name,
          description: body.description,
          created_at: new Date().toISOString(),
        }),
      });
    }
    return route.continue();
  });

  await page.goto("/collections");
  await page.getByRole("button", { name: /create/i }).click();
  await page.getByLabel(/name/i).fill("new-collection");
  await page.getByRole("button", { name: /create/i }).last().click();
  await expect(page.getByText("new-collection")).toBeVisible();
});
```

---

## Coverage

### Backend Coverage

```bash
python -m pytest --cov=vectorforge --cov=server --cov-report=term-missing --cov-report=html
# Open htmlcov/index.html in a browser
```

### Frontend Coverage

```bash
cd frontend
npm run test:coverage
# Coverage report is written to frontend/coverage/
```

---

## CI Integration

Recommended CI pipeline steps:

```yaml
# Backend
- python -m pytest --cov=vectorforge --cov=server --tb=short -q
- python -m ruff check .
- python -m mypy vectorforge/ server/

# Frontend unit tests
- cd frontend && npm ci && npm test

# Frontend E2E
- cd frontend && npx playwright install --with-deps
- cd frontend && npx playwright test
```

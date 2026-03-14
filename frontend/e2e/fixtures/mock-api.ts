/**
 * Shared mock API data and route handlers for Playwright E2E tests.
 */
import type { Page } from "@playwright/test";

export const COLLECTION_ID = "11111111-1111-1111-1111-111111111111";

export const MOCK_COLLECTIONS = [
  {
    id: COLLECTION_ID,
    name: "test-docs",
    description: "A test document collection",
    embedding_config: { default_provider: "voyage" },
    chunking_config: { strategy: "recursive", chunk_size: 1000 },
    created_at: "2025-01-15T10:00:00Z",
    updated_at: null,
  },
];

export const MOCK_DOCUMENTS = [
  {
    id: "22222222-2222-2222-2222-222222222222",
    collection_id: COLLECTION_ID,
    source_uri: "guide.md",
    content_type: "text/markdown",
    status: "indexed",
    content_size_bytes: 4096,
    metadata: { author: "Jane" },
    created_at: "2025-01-15T11:00:00Z",
    updated_at: null,
  },
  {
    id: "33333333-3333-3333-3333-333333333333",
    collection_id: COLLECTION_ID,
    source_uri: "readme.txt",
    content_type: "text/plain",
    status: "indexed",
    content_size_bytes: 1024,
    metadata: {},
    created_at: "2025-01-15T12:00:00Z",
    updated_at: null,
  },
];

export const MOCK_SUMMARY = {
  total_queries: 42,
  unique_queries: 15,
  latency: {
    avg_ms: 120.5,
    min_ms: 45.0,
    max_ms: 350.0,
    p50_ms: 100.0,
    p95_ms: 280.0,
    sample_count: 42,
  },
  top_queries: [
    { query_text: "What is vector search?", count: 8 },
    { query_text: "How does RAG work?", count: 5 },
    { query_text: "Explain embeddings", count: 3 },
  ],
  volume: [
    { date: "2025-01-10", count: 10 },
    { date: "2025-01-11", count: 15 },
    { date: "2025-01-12", count: 17 },
  ],
};

export const MOCK_HEALTH = {
  status: "healthy",
  components: [
    { name: "database", status: "healthy", latency_ms: 5.2, message: null },
    { name: "pgvector", status: "healthy", latency_ms: 3.1, message: null },
  ],
  checked_at: "2025-01-15T10:00:00Z",
};

export const MOCK_PROVIDERS = {
  embedding_providers: ["voyage", "openai"],
  llm_providers: ["openai", "anthropic"],
};

/**
 * Install all API mock routes on a Playwright page.
 */
export async function setupMockApi(page: Page): Promise<void> {
  // Collections
  await page.route("**/api/collections", async (route) => {
    const method = route.request().method();
    if (method === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ collections: MOCK_COLLECTIONS }),
      });
    }
    if (method === "POST") {
      const body = route.request().postDataJSON();
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          id: "44444444-4444-4444-4444-444444444444",
          name: body.name,
          description: body.description ?? null,
          embedding_config: null,
          chunking_config: null,
          created_at: new Date().toISOString(),
          updated_at: null,
        }),
      });
    }
    return route.continue();
  });

  // Single collection
  await page.route(`**/api/collections/${COLLECTION_ID}`, async (route) => {
    const method = route.request().method();
    if (method === "GET") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_COLLECTIONS[0]),
      });
    }
    if (method === "DELETE") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ message: "Collection deleted" }),
      });
    }
    return route.continue();
  });

  // Documents
  await page.route(
    `**/api/collections/${COLLECTION_ID}/documents**`,
    async (route) => {
      if (route.request().method() === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            documents: MOCK_DOCUMENTS,
            total: MOCK_DOCUMENTS.length,
          }),
        });
      }
      if (route.request().method() === "POST") {
        return route.fulfill({
          status: 201,
          contentType: "application/json",
          body: JSON.stringify(MOCK_DOCUMENTS[0]),
        });
      }
      return route.continue();
    },
  );

  // Analytics
  await page.route(
    `**/api/analytics/${COLLECTION_ID}/summary**`,
    async (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_SUMMARY),
      }),
  );

  await page.route(
    `**/api/analytics/${COLLECTION_ID}/top-queries**`,
    async (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ queries: MOCK_SUMMARY.top_queries }),
      }),
  );

  await page.route(
    `**/api/analytics/${COLLECTION_ID}/latency**`,
    async (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_SUMMARY.latency),
      }),
  );

  // Status
  await page.route("**/api/status", async (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_HEALTH),
    }),
  );

  await page.route("**/api/status/providers", async (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_PROVIDERS),
    }),
  );

  // Query stream
  await page.route("**/api/query/stream", async (route) => {
    const sseBody = [
      'data: {"type":"token","content":"Vector"}',
      "",
      'data: {"type":"token","content":" search"}',
      "",
      'data: {"type":"token","content":" is great."}',
      "",
      'data: {"type":"done","latency_ms":150.0}',
      "",
    ].join("\n");

    return route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sseBody,
    });
  });

  // Sync query
  await page.route("**/api/query", async (route) => {
    if (route.request().method() === "POST") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          answer: "Vector search is great.",
          sources: [
            {
              document_source: "guide.md",
              chunk_index: 0,
              score: 0.95,
              snippet: "Vector search enables semantic similarity...",
            },
          ],
          retrieval_latency_ms: 50.0,
          generation_latency_ms: 100.0,
          total_latency_ms: 150.0,
        }),
      });
    }
    return route.continue();
  });
}

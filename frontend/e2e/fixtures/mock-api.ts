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

// --- Evaluation ---

export const EVAL_RUN_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee";
const EVAL_REC_ID = "ffffffff-ffff-ffff-ffff-ffffffffffff";
const EVAL_QUERY_LOG_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

export const MOCK_EVALUATION_RUNS = [
  {
    run_id: EVAL_RUN_ID,
    status: "completed",
    sample_size: 50,
    started_at: "2025-01-15T09:00:00Z",
    completed_at: "2025-01-15T09:05:00Z",
    summary_scores: {
      retrieval_relevance: {
        avg: 0.78,
        min: 0.35,
        max: 0.98,
        p50: 0.8,
        below_threshold: 5,
        sample_count: 50,
      },
      chunk_coverage: {
        avg: 0.65,
        min: 0.2,
        max: 0.95,
        p50: 0.68,
        below_threshold: 12,
        sample_count: 50,
      },
      faithfulness: {
        avg: 0.82,
        min: 0.4,
        max: 1.0,
        p50: 0.85,
        below_threshold: 4,
        sample_count: 50,
      },
      answer_relevance: {
        avg: 0.74,
        min: 0.3,
        max: 0.97,
        p50: 0.76,
        below_threshold: 7,
        sample_count: 50,
      },
      hallucination: {
        avg: 0.88,
        min: 0.5,
        max: 1.0,
        p50: 0.92,
        below_threshold: 2,
        sample_count: 50,
      },
      embedding_drift: {
        avg: 0.91,
        min: 0.7,
        max: 1.0,
        p50: 0.93,
        below_threshold: 1,
        sample_count: 50,
      },
    },
    created_at: "2025-01-15T09:00:00Z",
  },
];

export const MOCK_EVALUATION_RESULTS = [
  {
    id: "aaaa1111-1111-1111-1111-111111111111",
    run_id: EVAL_RUN_ID,
    query_log_id: EVAL_QUERY_LOG_ID,
    evaluator_name: "retrieval_relevance",
    score: 0.35,
    details: { per_chunk_scores: [0.3, 0.4] },
    reasoning: "Chunks were only tangentially related to the query.",
  },
  {
    id: "aaaa2222-2222-2222-2222-222222222222",
    run_id: EVAL_RUN_ID,
    query_log_id: EVAL_QUERY_LOG_ID,
    evaluator_name: "faithfulness",
    score: 0.6,
    details: { supported: 3, unsupported: 2, total_claims: 5 },
    reasoning: "Two claims lacked supporting evidence in context.",
  },
  {
    id: "aaaa3333-3333-3333-3333-333333333333",
    run_id: EVAL_RUN_ID,
    query_log_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    evaluator_name: "retrieval_relevance",
    score: 0.9,
    details: { per_chunk_scores: [0.85, 0.95] },
    reasoning: "Highly relevant chunks retrieved.",
  },
];

export const MOCK_RECOMMENDATIONS = [
  {
    id: EVAL_REC_ID,
    run_id: EVAL_RUN_ID,
    category: "chunking",
    severity: "medium",
    title: "Incomplete Chunk Coverage",
    description:
      "Retrieved chunks don't cover all aspects of queries. Consider reducing chunk size or increasing top_k.",
    evidence: { avg_coverage: 0.65, threshold: 0.5 },
    status: "open",
  },
  {
    id: "ffffffff-ffff-ffff-ffff-fffffffffff2",
    run_id: EVAL_RUN_ID,
    category: "generation",
    severity: "low",
    title: "Low Answer Relevance",
    description:
      "Answers could be more focused on user queries. Consider improving system prompt.",
    evidence: { avg_relevance: 0.74, threshold: 0.6 },
    status: "open",
  },
];

export const MOCK_TRENDS = [
  {
    evaluator: "retrieval_relevance",
    scores: [0.7, 0.73, 0.78],
    direction: "improving" as const,
    change_pct: 11.43,
  },
  {
    evaluator: "chunk_coverage",
    scores: [0.68, 0.66, 0.65],
    direction: "degrading" as const,
    change_pct: -4.41,
  },
  {
    evaluator: "faithfulness",
    scores: [0.8, 0.81, 0.82],
    direction: "stable" as const,
    change_pct: 2.5,
  },
  {
    evaluator: "answer_relevance",
    scores: [0.72, 0.73, 0.74],
    direction: "stable" as const,
    change_pct: 2.78,
  },
  {
    evaluator: "hallucination",
    scores: [0.85, 0.87, 0.88],
    direction: "stable" as const,
    change_pct: 3.53,
  },
  {
    evaluator: "embedding_drift",
    scores: [0.9, 0.91, 0.91],
    direction: "stable" as const,
    change_pct: 1.11,
  },
];

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

  // --- Evaluations ---

  // POST /evaluations/run — trigger
  await page.route("**/api/evaluations/run", async (route) => {
    if (route.request().method() === "POST") {
      return route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify(MOCK_EVALUATION_RUNS[0]),
      });
    }
    return route.continue();
  });

  // GET /evaluations/runs/:id/results
  await page.route("**/api/evaluations/runs/*/results**", async (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ results: MOCK_EVALUATION_RESULTS }),
    }),
  );

  // GET /evaluations/runs/:id (single run)
  await page.route(
    `**/api/evaluations/runs/${EVAL_RUN_ID}`,
    async (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_EVALUATION_RUNS[0]),
      }),
  );

  // GET /evaluations/runs (list)
  await page.route("**/api/evaluations/runs**", async (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ runs: MOCK_EVALUATION_RUNS }),
    }),
  );

  // PATCH /evaluations/recommendations/:id
  await page.route(
    "**/api/evaluations/recommendations/*",
    async (route) => {
      if (route.request().method() === "PATCH") {
        const body = route.request().postDataJSON();
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            ...MOCK_RECOMMENDATIONS[0],
            status: body.status,
          }),
        });
      }
      return route.continue();
    },
  );

  // GET /evaluations/recommendations
  await page.route(
    "**/api/evaluations/recommendations**",
    async (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          recommendations: MOCK_RECOMMENDATIONS,
        }),
      }),
  );

  // GET /evaluations/trends
  await page.route("**/api/evaluations/trends**", async (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ trends: MOCK_TRENDS }),
    }),
  );
}

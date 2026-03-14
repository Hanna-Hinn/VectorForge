/**
 * Vite plugin that mocks all /api endpoints with realistic data.
 * Usage: import and add to vite plugins when running in mock mode.
 * Start with: VITE_MOCK=true npm run dev
 */
import type { Plugin } from "vite";

const COLLECTION_ID = "11111111-1111-1111-1111-111111111111";

const COLLECTIONS = [
  {
    id: COLLECTION_ID,
    name: "engineering-docs",
    description: "Internal engineering documentation and guides",
    embedding_config: { default_provider: "voyage", metric: "cosine" },
    chunking_config: { strategy: "recursive", chunk_size: 1000, chunk_overlap: 200 },
    created_at: "2025-01-15T10:00:00Z",
    updated_at: null,
  },
  {
    id: "22222222-2222-2222-2222-222222222222",
    name: "product-specs",
    description: "Product specifications and requirements",
    embedding_config: { default_provider: "openai", metric: "cosine" },
    chunking_config: { strategy: "markdown", chunk_size: 800 },
    created_at: "2025-01-10T08:30:00Z",
    updated_at: "2025-01-14T16:00:00Z",
  },
];

const DOCUMENTS = [
  {
    id: "aaaa-1111",
    collection_id: COLLECTION_ID,
    source_uri: "architecture-overview.md",
    content_type: "text/markdown",
    status: "indexed",
    content_size_bytes: 8192,
    metadata: { author: "Jane", team: "platform" },
    created_at: "2025-01-15T11:00:00Z",
    updated_at: null,
  },
  {
    id: "aaaa-2222",
    collection_id: COLLECTION_ID,
    source_uri: "deployment-guide.md",
    content_type: "text/markdown",
    status: "indexed",
    content_size_bytes: 4096,
    metadata: { author: "Bob" },
    created_at: "2025-01-15T12:00:00Z",
    updated_at: null,
  },
  {
    id: "aaaa-3333",
    collection_id: COLLECTION_ID,
    source_uri: "api-reference.html",
    content_type: "text/html",
    status: "indexed",
    content_size_bytes: 15360,
    metadata: { version: "2.0" },
    created_at: "2025-01-16T09:00:00Z",
    updated_at: null,
  },
  {
    id: "aaaa-4444",
    collection_id: COLLECTION_ID,
    source_uri: "troubleshooting.txt",
    content_type: "text/plain",
    status: "pending",
    content_size_bytes: 2048,
    metadata: {},
    created_at: "2025-01-16T10:00:00Z",
    updated_at: null,
  },
];

const SUMMARY = {
  total_queries: 156,
  unique_queries: 42,
  latency: {
    avg_ms: 185.3,
    min_ms: 45.0,
    max_ms: 890.0,
    p50_ms: 150.0,
    p95_ms: 520.0,
    sample_count: 156,
  },
  top_queries: [
    { query_text: "How to deploy to production?", count: 18 },
    { query_text: "What is the database schema?", count: 14 },
    { query_text: "Explain the RAG pipeline", count: 11 },
    { query_text: "Vector search performance tuning", count: 9 },
    { query_text: "API authentication setup", count: 7 },
  ],
  volume: [
    { date: "2025-01-10", count: 12 },
    { date: "2025-01-11", count: 25 },
    { date: "2025-01-12", count: 18 },
    { date: "2025-01-13", count: 30 },
    { date: "2025-01-14", count: 28 },
    { date: "2025-01-15", count: 35 },
    { date: "2025-01-16", count: 8 },
  ],
};

const HEALTH = {
  status: "healthy",
  components: [
    { name: "database", status: "healthy", latency_ms: 4.2, message: null },
    { name: "pgvector", status: "healthy", latency_ms: 3.5, message: null },
    { name: "embedding_providers", status: "healthy", latency_ms: 120.0, message: null },
    { name: "llm_providers", status: "healthy", latency_ms: 200.0, message: null },
  ],
  checked_at: new Date().toISOString(),
};

const PROVIDERS = {
  embedding_providers: ["voyage", "openai"],
  llm_providers: ["openai", "anthropic"],
};

// --- Evaluation ---

const EVAL_RUN_ID = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee";

const EVALUATION_RUNS = [
  {
    run_id: EVAL_RUN_ID,
    status: "completed",
    sample_size: 50,
    started_at: "2025-01-15T09:00:00Z",
    completed_at: "2025-01-15T09:05:00Z",
    summary_scores: {
      retrieval_relevance: {
        avg: 0.78, min: 0.35, max: 0.98,
        p50: 0.8, below_threshold: 5, sample_count: 50,
      },
      chunk_coverage: {
        avg: 0.65, min: 0.2, max: 0.95,
        p50: 0.68, below_threshold: 12, sample_count: 50,
      },
      faithfulness: {
        avg: 0.82, min: 0.4, max: 1.0,
        p50: 0.85, below_threshold: 4, sample_count: 50,
      },
      answer_relevance: {
        avg: 0.74, min: 0.3, max: 0.97,
        p50: 0.76, below_threshold: 7, sample_count: 50,
      },
      hallucination: {
        avg: 0.88, min: 0.5, max: 1.0,
        p50: 0.92, below_threshold: 2, sample_count: 50,
      },
      embedding_drift: {
        avg: 0.91, min: 0.7, max: 1.0,
        p50: 0.93, below_threshold: 1, sample_count: 50,
      },
    },
    created_at: "2025-01-15T09:00:00Z",
  },
];

const EVALUATION_RESULTS = [
  {
    id: "aaaa1111-1111-1111-1111-111111111111",
    run_id: EVAL_RUN_ID,
    query_log_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    evaluator_name: "retrieval_relevance",
    score: 0.35,
    details: { per_chunk_scores: [0.3, 0.4] },
    reasoning: "Chunks were only tangentially related.",
  },
  {
    id: "aaaa2222-2222-2222-2222-222222222222",
    run_id: EVAL_RUN_ID,
    query_log_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    evaluator_name: "faithfulness",
    score: 0.6,
    details: { supported: 3, unsupported: 2, total_claims: 5 },
    reasoning: "Two claims lacked supporting evidence.",
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

const RECOMMENDATIONS = [
  {
    id: "ffffffff-ffff-ffff-ffff-ffffffffffff",
    run_id: EVAL_RUN_ID,
    category: "chunking",
    severity: "medium",
    title: "Incomplete Chunk Coverage",
    description:
      "Retrieved chunks don't cover all aspects of queries. Consider reducing chunk size.",
    evidence: { avg_coverage: 0.65, threshold: 0.5 },
    status: "open",
  },
  {
    id: "ffffffff-ffff-ffff-ffff-fffffffffff2",
    run_id: EVAL_RUN_ID,
    category: "generation",
    severity: "low",
    title: "Low Answer Relevance",
    description: "Answers could be more focused. Consider improving system prompt.",
    evidence: { avg_relevance: 0.74, threshold: 0.6 },
    status: "open",
  },
];

const TRENDS = [
  { evaluator: "retrieval_relevance", scores: [0.7, 0.73, 0.78], direction: "improving", change_pct: 11.43 },
  { evaluator: "chunk_coverage", scores: [0.68, 0.66, 0.65], direction: "degrading", change_pct: -4.41 },
  { evaluator: "faithfulness", scores: [0.8, 0.81, 0.82], direction: "stable", change_pct: 2.5 },
  { evaluator: "answer_relevance", scores: [0.72, 0.73, 0.74], direction: "stable", change_pct: 2.78 },
  { evaluator: "hallucination", scores: [0.85, 0.87, 0.88], direction: "stable", change_pct: 3.53 },
  { evaluator: "embedding_drift", scores: [0.9, 0.91, 0.91], direction: "stable", change_pct: 1.11 },
];

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export default function mockApiPlugin(): Plugin {
  return {
    name: "vite-plugin-mock-api",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        if (!url.startsWith("/api/")) return next();

        res.setHeader("Content-Type", "application/json");

        // Collections
        if (url === "/api/collections" && req.method === "GET") {
          res.end(JSON.stringify({ collections: COLLECTIONS }));
          return;
        }
        if (url === "/api/collections" && req.method === "POST") {
          let body = "";
          req.on("data", (c: Buffer) => (body += c.toString()));
          req.on("end", () => {
            const parsed = JSON.parse(body);
            res.statusCode = 201;
            res.end(
              JSON.stringify({
                id: crypto.randomUUID(),
                name: parsed.name,
                description: parsed.description ?? null,
                embedding_config: null,
                chunking_config: null,
                created_at: new Date().toISOString(),
                updated_at: null,
              }),
            );
          });
          return;
        }

        // Single collection
        const colMatch = url.match(/^\/api\/collections\/([\w-]+)$/);
        if (colMatch) {
          if (req.method === "GET") {
            const col = COLLECTIONS.find((c) => c.id === colMatch[1]);
            res.end(JSON.stringify(col ?? COLLECTIONS[0]));
            return;
          }
          if (req.method === "DELETE") {
            res.end(JSON.stringify({ message: "Collection deleted" }));
            return;
          }
        }

        // Documents list
        const docsMatch = url.match(
          /^\/api\/collections\/([\w-]+)\/documents/,
        );
        if (docsMatch && req.method === "GET") {
          res.end(
            JSON.stringify({ documents: DOCUMENTS, total: DOCUMENTS.length }),
          );
          return;
        }
        if (docsMatch && req.method === "POST") {
          res.statusCode = 201;
          res.end(JSON.stringify(DOCUMENTS[0]));
          return;
        }

        // Single document
        if (url.match(/^\/api\/documents\//) && req.method === "DELETE") {
          res.end(JSON.stringify({ message: "Document deleted" }));
          return;
        }

        // Analytics
        if (url.includes("/analytics/") && url.includes("/summary")) {
          res.end(JSON.stringify(SUMMARY));
          return;
        }
        if (url.includes("/analytics/") && url.includes("/top-queries")) {
          res.end(JSON.stringify({ queries: SUMMARY.top_queries }));
          return;
        }
        if (url.includes("/analytics/") && url.includes("/latency")) {
          res.end(JSON.stringify(SUMMARY.latency));
          return;
        }

        // Status
        if (url === "/api/status" && req.method === "GET") {
          res.end(JSON.stringify(HEALTH));
          return;
        }
        if (url === "/api/status/providers") {
          res.end(JSON.stringify(PROVIDERS));
          return;
        }

        // Query stream
        if (url === "/api/query/stream" && req.method === "POST") {
          res.setHeader("Content-Type", "text/event-stream");
          res.setHeader("Cache-Control", "no-cache");
          res.setHeader("Connection", "keep-alive");

          const tokens = [
            "VectorForge",
            " is",
            " a",
            " high-performance",
            " RAG",
            " engine",
            " that",
            " uses",
            " pgvector",
            " for",
            " semantic",
            " search.",
            " It",
            " supports",
            " multiple",
            " embedding",
            " providers",
            " and",
            " LLM",
            " backends",
            " for",
            " flexible",
            " retrieval-augmented",
            " generation.",
          ];

          let i = 0;
          const interval = setInterval(() => {
            if (i < tokens.length) {
              res.write(
                `data: ${JSON.stringify({ type: "token", content: tokens[i] })}\n\n`,
              );
              i++;
            } else {
              res.write(
                `data: ${JSON.stringify({ type: "done", latency_ms: 250.0 })}\n\n`,
              );
              clearInterval(interval);
              res.end();
            }
          }, 50);
          return;
        }

        // Sync query
        if (url === "/api/query" && req.method === "POST") {
          res.end(
            JSON.stringify({
              answer:
                "VectorForge is a high-performance RAG engine that uses pgvector for semantic search.",
              sources: [
                {
                  document_source: "architecture-overview.md",
                  chunk_index: 0,
                  score: 0.95,
                  snippet:
                    "VectorForge is a standalone Retrieval-Augmented Generation engine...",
                },
                {
                  document_source: "deployment-guide.md",
                  chunk_index: 2,
                  score: 0.87,
                  snippet:
                    "The system uses pgvector as its primary vector store backend...",
                },
              ],
              retrieval_latency_ms: 65.0,
              generation_latency_ms: 185.0,
              total_latency_ms: 250.0,
            }),
          );
          return;
        }

        // --- Evaluations ---

        // POST /evaluations/run
        if (url === "/api/evaluations/run" && req.method === "POST") {
          res.statusCode = 202;
          res.end(JSON.stringify(EVALUATION_RUNS[0]));
          return;
        }

        // GET /evaluations/runs/:id/results
        if (
          url.match(/^\/api\/evaluations\/runs\/[\w-]+\/results/)
          && req.method === "GET"
        ) {
          res.end(JSON.stringify({ results: EVALUATION_RESULTS }));
          return;
        }

        // GET /evaluations/runs/:id
        const evalRunMatch = url.match(
          /^\/api\/evaluations\/runs\/([\w-]+)$/,
        );
        if (evalRunMatch && req.method === "GET") {
          res.end(JSON.stringify(EVALUATION_RUNS[0]));
          return;
        }

        // GET /evaluations/runs
        if (url.match(/^\/api\/evaluations\/runs(\?|$)/) && req.method === "GET") {
          res.end(JSON.stringify({ runs: EVALUATION_RUNS }));
          return;
        }

        // PATCH /evaluations/recommendations/:id
        const recPatchMatch = url.match(
          /^\/api\/evaluations\/recommendations\/([\w-]+)$/,
        );
        if (recPatchMatch && req.method === "PATCH") {
          let body = "";
          req.on("data", (c: Buffer) => (body += c.toString()));
          req.on("end", () => {
            const parsed = JSON.parse(body);
            res.end(
              JSON.stringify({ ...RECOMMENDATIONS[0], status: parsed.status }),
            );
          });
          return;
        }

        // GET /evaluations/recommendations
        if (
          url.match(/^\/api\/evaluations\/recommendations(\?|$)/)
          && req.method === "GET"
        ) {
          res.end(JSON.stringify({ recommendations: RECOMMENDATIONS }));
          return;
        }

        // GET /evaluations/trends
        if (url.match(/^\/api\/evaluations\/trends/) && req.method === "GET") {
          res.end(JSON.stringify({ trends: TRENDS }));
          return;
        }

        // Fallback
        res.statusCode = 404;
        res.end(JSON.stringify({ error: "not_found", detail: "Mock not found" }));
      });
    },
  };
}

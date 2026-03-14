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

        // Fallback
        res.statusCode = 404;
        res.end(JSON.stringify({ error: "not_found", detail: "Mock not found" }));
      });
    },
  };
}

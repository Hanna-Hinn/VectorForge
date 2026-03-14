/** API request and response types. */

import type {
  AnalyticsSummary,
  Collection,
  Document,
  LatencyStats,
  QueryFrequency,
  SourceCitation,
  SystemHealth,
  ProvidersInfo,
} from "./models";

// --- Collections ---

export interface CreateCollectionRequest {
  name: string;
  description?: string | null;
  embedding_provider?: string | null;
  embedding_model?: string | null;
  chunking_strategy?: string | null;
  chunk_size?: number | null;
  chunk_overlap?: number | null;
}

export interface CollectionListResponse {
  collections: Collection[];
}

// --- Documents ---

export interface IngestDocumentRequest {
  source: string;
  metadata?: Record<string, unknown>;
  chunking_strategy?: string;
  chunk_size?: number;
  chunk_overlap?: number;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface DocumentDetailResponse extends Document {
  raw_content: string;
  chunk_count: number;
}

export interface BatchIngestResponse {
  results: Array<{
    source: string;
    document_id: string | null;
    error: string | null;
  }>;
  succeeded: number;
  failed: number;
}

// --- Query ---

export interface QueryRequest {
  query: string;
  collection_id: string;
  top_k?: number;
  min_score?: number;
  filters?: Record<string, unknown>;
  llm_provider?: string;
  llm_model?: string;
  temperature?: number;
  max_tokens?: number;
  include_sources?: boolean;
  max_context_tokens?: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceCitation[];
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  total_latency_ms: number;
}

// --- Analytics ---

export type { AnalyticsSummary, LatencyStats, QueryFrequency };

export interface TopQueriesResponse {
  queries: QueryFrequency[];
}

// --- Status ---

export type { SystemHealth, ProvidersInfo };

// --- Common ---

export interface ErrorResponse {
  error: string;
  detail: string;
}

export interface MessageResponse {
  message: string;
}

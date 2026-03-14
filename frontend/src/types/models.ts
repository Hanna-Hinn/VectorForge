/** Domain model types mirroring the Python Pydantic models. */

export interface Collection {
  id: string;
  name: string;
  description: string | null;
  embedding_config: Record<string, unknown> | null;
  chunking_config: Record<string, unknown> | null;
  created_at: string;
  updated_at: string | null;
}

export interface Document {
  id: string;
  collection_id: string;
  source_uri: string;
  content_type: string;
  status: DocumentStatus;
  content_size_bytes: number;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string | null;
}

export type DocumentStatus = "pending" | "processing" | "indexed" | "failed";

export interface SourceCitation {
  document_source: string;
  chunk_index: number;
  score: number;
  snippet: string;
}

export interface LatencyStats {
  avg_ms: number;
  min_ms: number;
  max_ms: number;
  p50_ms: number;
  p95_ms: number;
  sample_count: number;
}

export interface QueryFrequency {
  query_text: string;
  count: number;
}

export interface VolumeDataPoint {
  date: string;
  count: number;
}

export interface AnalyticsSummary {
  total_queries: number;
  unique_queries: number;
  latency: LatencyStats | null;
  top_queries: QueryFrequency[];
  volume: VolumeDataPoint[];
}

export interface ComponentHealth {
  name: string;
  status: string;
  latency_ms: number | null;
  message: string | null;
}

export interface SystemHealth {
  status: string;
  components: ComponentHealth[];
  checked_at: string;
}

export interface ProvidersInfo {
  embedding_providers: string[];
  llm_providers: string[];
}

// --- Evaluation ---

export interface EvaluationRun {
  run_id: string;
  status: string;
  sample_size: number;
  started_at: string | null;
  completed_at: string | null;
  summary_scores: Record<string, EvaluatorScores>;
  created_at: string | null;
}

export interface EvaluatorScores {
  avg: number;
  min: number;
  max: number;
  p50: number;
  below_threshold: number;
  sample_count: number;
}

export interface EvaluationResult {
  id: string;
  run_id: string;
  query_log_id: string;
  evaluator_name: string;
  score: number | null;
  details: Record<string, unknown>;
  reasoning: string | null;
}

export interface EvaluationRecommendation {
  id: string;
  run_id: string;
  category: string;
  severity: string;
  title: string;
  description: string;
  evidence: Record<string, unknown>;
  status: string;
}

export interface TrendData {
  evaluator: string;
  scores: number[];
  direction: "improving" | "stable" | "degrading";
  change_pct: number;
}

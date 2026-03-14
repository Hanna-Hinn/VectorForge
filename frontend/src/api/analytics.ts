/** Analytics API calls. */

import { get } from "./client";
import type {
  AnalyticsSummary,
  LatencyStats,
  TopQueriesResponse,
} from "../types/api";

export function getSummary(
  collectionId: string,
  since?: string,
): Promise<AnalyticsSummary> {
  const params = since ? `?from=${encodeURIComponent(since)}` : "";
  return get<AnalyticsSummary>(`/analytics/${collectionId}/summary${params}`);
}

export function getTopQueries(
  collectionId: string,
  limit = 20,
  since?: string,
): Promise<TopQueriesResponse> {
  const searchParams = new URLSearchParams({ limit: String(limit) });
  if (since) searchParams.set("from", since);
  return get<TopQueriesResponse>(
    `/analytics/${collectionId}/top-queries?${searchParams}`,
  );
}

export function getLatency(
  collectionId: string,
  since?: string,
): Promise<LatencyStats | null> {
  const params = since ? `?from=${encodeURIComponent(since)}` : "";
  return get<LatencyStats | null>(
    `/analytics/${collectionId}/latency${params}`,
  );
}

/** React Query hooks for analytics data. */

import { useQuery } from "@tanstack/react-query";
import * as analyticsApi from "../api/analytics";

export function useAnalyticsSummary(collectionId: string, since?: string) {
  return useQuery({
    queryKey: ["analytics", "summary", collectionId, since],
    queryFn: () => analyticsApi.getSummary(collectionId, since),
    enabled: !!collectionId,
    refetchInterval: 30_000,
  });
}

export function useTopQueries(
  collectionId: string,
  limit = 20,
  since?: string,
) {
  return useQuery({
    queryKey: ["analytics", "top-queries", collectionId, limit, since],
    queryFn: () => analyticsApi.getTopQueries(collectionId, limit, since),
    enabled: !!collectionId,
    select: (data) => data.queries,
  });
}

export function useLatencyStats(collectionId: string, since?: string) {
  return useQuery({
    queryKey: ["analytics", "latency", collectionId, since],
    queryFn: () => analyticsApi.getLatency(collectionId, since),
    enabled: !!collectionId,
  });
}

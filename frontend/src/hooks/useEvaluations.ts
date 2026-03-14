/** React Query hooks for evaluation data. */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as evalApi from "../api/evaluations";

export function useEvaluationRuns(limit = 10) {
  return useQuery({
    queryKey: ["evaluations", "runs", limit],
    queryFn: () => evalApi.listRuns(limit),
    select: (data) => data.runs,
    refetchInterval: 30_000,
  });
}

export function useEvaluationRun(runId: string) {
  return useQuery({
    queryKey: ["evaluations", "run", runId],
    queryFn: () => evalApi.getRun(runId),
    enabled: !!runId,
  });
}

export function useEvaluationResults(runId: string, evaluator?: string) {
  return useQuery({
    queryKey: ["evaluations", "results", runId, evaluator],
    queryFn: () => evalApi.getRunResults(runId, evaluator),
    enabled: !!runId,
    select: (data) => data.results,
  });
}

export function useRecommendations(status?: string, category?: string) {
  return useQuery({
    queryKey: ["evaluations", "recommendations", status, category],
    queryFn: () => evalApi.listRecommendations(status, category),
    select: (data) => data.recommendations,
    refetchInterval: 30_000,
  });
}

export function useTrends(limit = 10) {
  return useQuery({
    queryKey: ["evaluations", "trends", limit],
    queryFn: () => evalApi.getTrends(limit),
    select: (data) => data.trends,
  });
}

export function useTriggerEvaluation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      sampleSize,
      strategy,
    }: {
      sampleSize?: number;
      strategy?: string;
    }) => evalApi.triggerRun(sampleSize, strategy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["evaluations"] });
    },
  });
}

export function useUpdateRecommendation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ recId, status }: { recId: string; status: string }) =>
      evalApi.updateRecommendation(recId, status),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["evaluations", "recommendations"],
      });
    },
  });
}

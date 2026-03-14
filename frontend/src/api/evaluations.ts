/** Evaluation API calls. */

import { get, post, patch } from "./client";
import type {
  EvaluationRunListResponse,
  EvaluationResultListResponse,
  RecommendationListResponse,
  TrendListResponse,
} from "../types/api";
import type {
  EvaluationRun,
  EvaluationRecommendation,
} from "../types/models";

export function triggerRun(
  sampleSize?: number,
  strategy?: string,
): Promise<EvaluationRun> {
  const body: Record<string, unknown> = {};
  if (sampleSize != null) body.sample_size = sampleSize;
  if (strategy != null) body.sample_strategy = strategy;
  return post<EvaluationRun>("/evaluations/run", body);
}

export function listRuns(limit = 10): Promise<EvaluationRunListResponse> {
  return get<EvaluationRunListResponse>(
    `/evaluations/runs?limit=${limit}`,
  );
}

export function getRun(runId: string): Promise<EvaluationRun> {
  return get<EvaluationRun>(`/evaluations/runs/${runId}`);
}

export function getRunResults(
  runId: string,
  evaluator?: string,
): Promise<EvaluationResultListResponse> {
  const params = evaluator ? `?evaluator=${encodeURIComponent(evaluator)}` : "";
  return get<EvaluationResultListResponse>(
    `/evaluations/runs/${runId}/results${params}`,
  );
}

export function listRecommendations(
  status?: string,
  category?: string,
): Promise<RecommendationListResponse> {
  const searchParams = new URLSearchParams();
  if (status) searchParams.set("status", status);
  if (category) searchParams.set("category", category);
  const qs = searchParams.toString();
  return get<RecommendationListResponse>(
    `/evaluations/recommendations${qs ? `?${qs}` : ""}`,
  );
}

export function updateRecommendation(
  recId: string,
  newStatus: string,
): Promise<EvaluationRecommendation> {
  return patch<EvaluationRecommendation>(
    `/evaluations/recommendations/${recId}`,
    { status: newStatus },
  );
}

export function getTrends(limit = 10): Promise<TrendListResponse> {
  return get<TrendListResponse>(`/evaluations/trends?limit=${limit}`);
}

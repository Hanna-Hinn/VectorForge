import {
  useEvaluationRuns,
  useEvaluationResults,
  useRecommendations,
  useTrends,
  useTriggerEvaluation,
} from "../hooks/useEvaluations";
import Header from "../components/layout/Header";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import ScoreRadar from "../components/features/ScoreRadar";
import TrendChart from "../components/features/TrendChart";
import RecommendationList from "../components/features/RecommendationList";
import WorstQueries from "../components/features/WorstQueries";
import { t } from "../i18n";

const DEFAULT_THRESHOLDS: Record<string, number> = {
  retrieval_relevance: 0.6,
  chunk_coverage: 0.5,
  faithfulness: 0.7,
  answer_relevance: 0.6,
  hallucination: 0.3,
  embedding_drift: 0.5,
};

export default function EvaluationPage() {
  const { data: runs, isLoading: runsLoading } = useEvaluationRuns(5);
  const latestRun = runs?.[0];
  const { data: results } = useEvaluationResults(latestRun?.run_id ?? "");
  const { data: recommendations } = useRecommendations("open");
  const { data: trends } = useTrends(10);
  const trigger = useTriggerEvaluation();

  const title = t("evaluation.title");

  if (runsLoading) {
    return (
      <>
        <Header title={title} />
        <div className="flex justify-center py-20">
          <Spinner size="lg" />
        </div>
      </>
    );
  }

  if (!latestRun) {
    return (
      <>
        <Header title={title} />
        <div className="flex flex-col items-center justify-center p-6 py-20 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-50 dark:bg-brand-900/20">
            <svg className="h-8 w-8 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            </svg>
          </div>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{t("evaluation.noRuns")}</p>
          <button
            onClick={() => trigger.mutate({})}
            disabled={trigger.isPending}
            className="mt-4 inline-flex items-center gap-1.5 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white transition-all duration-150 hover:bg-brand-700 active:scale-[0.97] disabled:opacity-50"
          >
            {trigger.isPending ? t("common.loading") : t("evaluation.runNow")}
          </button>
        </div>
      </>
    );
  }

  return (
    <>
      <Header title={title} />
      <div className="p-6 space-y-6">
        {/* Summary bar */}
        <div className="flex items-center justify-between">
          <div className="flex gap-4 text-sm text-gray-600 dark:text-gray-400">
            <span>
              {t("evaluation.status")}:{" "}
              <strong className="text-gray-800 dark:text-gray-200">
                {latestRun.status}
              </strong>
            </span>
            <span>
              {t("evaluation.samples")}:{" "}
              <strong>{latestRun.sample_size}</strong>
            </span>
            {latestRun.completed_at && (
              <span>
                {t("evaluation.completedAt")}:{" "}
                <strong>
                  {new Date(latestRun.completed_at).toLocaleString()}
                </strong>
              </span>
            )}
          </div>

          <button
            onClick={() => trigger.mutate({})}
            disabled={trigger.isPending}
            className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white transition-all duration-150 hover:bg-brand-700 active:scale-[0.97] disabled:opacity-50"
          >
            {trigger.isPending ? t("common.loading") : t("evaluation.runNow")}
          </button>
        </div>

        {/* Score overview */}
        {latestRun.summary_scores && (
          <ScoreRadar
            scores={latestRun.summary_scores}
            thresholds={DEFAULT_THRESHOLDS}
          />
        )}

        {/* Trends */}
        {trends && trends.length > 0 && <TrendChart trends={trends} />}

        {/* Recent runs */}
        {runs && runs.length > 1 && (
          <Card title={t("evaluation.recentRuns")}>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                    {t("evaluation.runId")}
                  </th>
                  <th className="pb-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">
                    {t("evaluation.status")}
                  </th>
                  <th className="pb-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                    {t("evaluation.samples")}
                  </th>
                  <th className="pb-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">
                    {t("evaluation.completedAt")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr
                    key={run.run_id}
                    className="border-b border-gray-100 dark:border-gray-800"
                  >
                    <td className="py-2 font-mono text-xs text-gray-600 dark:text-gray-400">
                      {run.run_id.slice(0, 8)}…
                    </td>
                    <td className="py-2">{run.status}</td>
                    <td className="py-2 text-right">{run.sample_size}</td>
                    <td className="py-2 text-right text-xs text-gray-500">
                      {run.completed_at
                        ? new Date(run.completed_at).toLocaleString()
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}

        {/* Recommendations */}
        {recommendations && recommendations.length > 0 && (
          <RecommendationList recommendations={recommendations} />
        )}

        {/* Worst queries */}
        {results && results.length > 0 && <WorstQueries results={results} />}
      </div>
    </>
  );
}

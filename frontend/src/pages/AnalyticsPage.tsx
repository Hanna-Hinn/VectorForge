import { useParams } from "react-router-dom";
import { useCollection } from "../hooks/useCollections";
import { useAnalyticsSummary, useLatencyStats } from "../hooks/useAnalytics";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import LatencyChart from "../components/features/LatencyChart";
import Header from "../components/layout/Header";
import { t } from "../i18n";
import { formatNumber, formatMs } from "../utils/format";

export default function AnalyticsPage() {
  const { collectionId = "" } = useParams();
  const { data: collection } = useCollection(collectionId);
  const { data: summary, isLoading } = useAnalyticsSummary(collectionId);
  const { data: latency } = useLatencyStats(collectionId);

  const title = collection
    ? `${t("analytics.title")} — ${collection.name}`
    : t("analytics.title");

  if (isLoading) {
    return (
      <>
        <Header title={title} />
        <div className="flex justify-center py-20"><Spinner size="lg" /></div>
      </>
    );
  }

  if (!summary) {
    return (
      <>
        <Header title={title} />
        <p className="flex flex-col items-center py-20 text-center text-gray-400">{t("analytics.noStats")}</p>
      </>
    );
  }

  return (
    <>
      <Header title={title} />
      <div className="p-6 space-y-6">
        {/* Summary cards */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Card>
            <p className="text-xs text-gray-500">{t("analytics.totalQueries")}</p>
            <p className="mt-1 text-2xl font-bold">{formatNumber(summary.total_queries)}</p>
          </Card>
          <Card>
            <p className="text-xs text-gray-500">{t("analytics.uniqueQueries")}</p>
            <p className="mt-1 text-2xl font-bold">{formatNumber(summary.unique_queries)}</p>
          </Card>
          {summary.latency && (
            <>
              <Card>
                <p className="text-xs text-gray-500">{t("analytics.avgLatency")}</p>
                <p className="mt-1 text-2xl font-bold">{formatMs(summary.latency.avg_ms)}</p>
              </Card>
              <Card>
                <p className="text-xs text-gray-500">{t("analytics.p95Latency")}</p>
                <p className="mt-1 text-2xl font-bold">{formatMs(summary.latency.p95_ms)}</p>
              </Card>
            </>
          )}
        </div>

        {/* Latency chart */}
        {latency && <LatencyChart stats={latency} />}

        {/* Top queries */}
        {summary.top_queries.length > 0 && (
          <Card title={t("analytics.topQueries")}>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">#</th>
                  <th className="pb-2 text-left text-xs font-semibold uppercase tracking-wider text-gray-500">Query</th>
                  <th className="pb-2 text-right text-xs font-semibold uppercase tracking-wider text-gray-500">Count</th>
                </tr>
              </thead>
              <tbody>
                {summary.top_queries.map((q, i) => (
                  <tr key={q.query_text} className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-2 text-gray-400">{i + 1}</td>
                    <td className="py-2">{q.query_text}</td>
                    <td className="py-2 text-right font-medium">{q.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>
    </>
  );
}

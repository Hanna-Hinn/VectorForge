import Card from "../ui/Card";
import type { EvaluationResult } from "../../types/models";

interface WorstQueriesProps {
  results: EvaluationResult[];
}

interface AggregatedQuery {
  queryLogId: string;
  scores: Record<string, number>;
  composite: number;
}

function aggregateByQuery(results: EvaluationResult[]): AggregatedQuery[] {
  const groups = new Map<string, Record<string, number>>();

  for (const r of results) {
    if (r.score === null) continue;
    const existing = groups.get(r.query_log_id) ?? {};
    existing[r.evaluator_name] = r.score;
    groups.set(r.query_log_id, existing);
  }

  const aggregated: AggregatedQuery[] = [];
  for (const [queryLogId, scores] of groups) {
    const vals = Object.values(scores);
    const composite = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    aggregated.push({ queryLogId, scores, composite });
  }

  return aggregated.sort((a, b) => a.composite - b.composite).slice(0, 10);
}

function formatScore(score: number | undefined): string {
  if (score === undefined) return "—";
  return (score * 100).toFixed(0) + "%";
}

function scoreClass(score: number | undefined): string {
  if (score === undefined) return "text-gray-400";
  if (score >= 0.7) return "text-green-600";
  if (score >= 0.5) return "text-yellow-600";
  return "text-red-600";
}

export default function WorstQueries({ results }: WorstQueriesProps) {
  const worst = aggregateByQuery(results);

  if (worst.length === 0) {
    return null;
  }

  const evaluators = [
    ...new Set(results.map((r) => r.evaluator_name)),
  ].sort();

  return (
    <Card title="Worst Performing Queries">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="pb-2 text-left font-medium text-gray-500">#</th>
              <th className="pb-2 text-left font-medium text-gray-500">Query ID</th>
              {evaluators.map((e) => (
                <th key={e} className="pb-2 text-right font-medium text-gray-500">
                  {e.replace(/_/g, " ")}
                </th>
              ))}
              <th className="pb-2 text-right font-medium text-gray-500">Avg</th>
            </tr>
          </thead>
          <tbody>
            {worst.map((q, i) => (
              <tr
                key={q.queryLogId}
                className="border-b border-gray-100 dark:border-gray-800"
              >
                <td className="py-2 text-gray-400">{i + 1}</td>
                <td className="py-2 font-mono text-xs text-gray-600 dark:text-gray-400">
                  {q.queryLogId.slice(0, 8)}…
                </td>
                {evaluators.map((e) => (
                  <td
                    key={e}
                    className={`py-2 text-right font-medium ${scoreClass(q.scores[e])}`}
                  >
                    {formatScore(q.scores[e])}
                  </td>
                ))}
                <td
                  className={`py-2 text-right font-bold ${scoreClass(q.composite)}`}
                >
                  {(q.composite * 100).toFixed(0)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

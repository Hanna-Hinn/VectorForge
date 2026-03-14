import Card from "../ui/Card";
import type { EvaluatorScores } from "../../types/models";

interface ScoreRadarProps {
  scores: Record<string, EvaluatorScores>;
  thresholds?: Record<string, number>;
}

const EVALUATOR_LABELS: Record<string, string> = {
  retrieval_relevance: "Retrieval Relevance",
  chunk_coverage: "Chunk Coverage",
  faithfulness: "Faithfulness",
  answer_relevance: "Answer Relevance",
  hallucination: "Hallucination Score",
  embedding_drift: "Embedding Stability",
};

function scoreColor(score: number, threshold: number): string {
  if (score >= threshold) return "bg-green-500";
  if (score >= threshold * 0.8) return "bg-yellow-500";
  return "bg-red-500";
}

export default function ScoreRadar({ scores, thresholds = {} }: ScoreRadarProps) {
  const entries = Object.entries(EVALUATOR_LABELS).filter(
    ([key]) => key in scores,
  );

  if (entries.length === 0) {
    return null;
  }

  const maxVal = 1.0;

  return (
    <Card title="Evaluator Scores">
      <div className="flex items-end gap-3" style={{ height: 160 }}>
        {entries.map(([key, label]) => {
          const avg = scores[key]?.avg ?? 0;
          const threshold = thresholds[key] ?? 0.5;
          return (
            <div
              key={key}
              className="flex flex-1 flex-col items-center gap-1"
            >
              <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                {(avg * 100).toFixed(0)}%
              </span>
              <div className="relative w-full" style={{ height: 120 }}>
                {/* Threshold line */}
                <div
                  className="absolute w-full border-t border-dashed border-gray-400"
                  style={{ bottom: `${(threshold / maxVal) * 100}%` }}
                />
                {/* Score bar */}
                <div
                  className={`absolute bottom-0 w-full rounded-t ${scoreColor(avg, threshold)}`}
                  style={{ height: `${(avg / maxVal) * 100}%`, minHeight: 4 }}
                />
              </div>
              <span className="text-[10px] text-center leading-tight text-gray-500">
                {label}
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

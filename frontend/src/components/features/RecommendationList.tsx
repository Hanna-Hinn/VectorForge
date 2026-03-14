import Card from "../ui/Card";
import { useUpdateRecommendation } from "../../hooks/useEvaluations";
import type { EvaluationRecommendation } from "../../types/models";

interface RecommendationListProps {
  recommendations: EvaluationRecommendation[];
}

const SEVERITY_ICONS: Record<string, string> = {
  critical: "\u{1F534}",
  high: "\u{1F7E0}",
  medium: "\u{1F7E1}",
  low: "\u{1F7E2}",
};

const STATUS_ACTIONS: Record<string, string[]> = {
  open: ["acknowledged", "resolved", "dismissed"],
  acknowledged: ["resolved", "dismissed"],
  resolved: [],
  dismissed: ["open"],
};

export default function RecommendationList({ recommendations }: RecommendationListProps) {
  const { mutate: updateRec, isPending } = useUpdateRecommendation();

  if (recommendations.length === 0) {
    return null;
  }

  return (
    <Card title="Recommendations">
      <div className="space-y-3">
        {recommendations.map((rec) => {
          const icon = SEVERITY_ICONS[rec.severity] ?? "\u26AA";
          const actions = STATUS_ACTIONS[rec.status] ?? [];

          return (
            <div
              key={rec.id}
              className="rounded-md border border-gray-200 p-3 dark:border-gray-700"
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 text-base" role="img" aria-label={rec.severity}>
                  {icon}
                </span>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                      {rec.title}
                    </h4>
                    <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] uppercase text-gray-500 dark:bg-gray-800">
                      {rec.category}
                    </span>
                    <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] uppercase text-gray-500 dark:bg-gray-800">
                      {rec.status}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-gray-600 dark:text-gray-400">
                    {rec.description}
                  </p>

                  {actions.length > 0 && (
                    <div className="mt-2 flex gap-2">
                      {actions.map((action) => (
                        <button
                          key={action}
                          disabled={isPending}
                          onClick={() => updateRec({ recId: rec.id, status: action })}
                          className="rounded bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700 transition-colors hover:bg-brand-100 disabled:opacity-50 dark:bg-brand-900/20 dark:text-brand-400"
                        >
                          {action.charAt(0).toUpperCase() + action.slice(1)}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

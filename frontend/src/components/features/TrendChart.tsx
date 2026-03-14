import Card from "../ui/Card";
import type { TrendData } from "../../types/models";

interface TrendChartProps {
  trends: TrendData[];
}

const DIRECTION_ICONS: Record<string, string> = {
  improving: "↑",
  stable: "→",
  degrading: "↓",
};

const DIRECTION_COLORS: Record<string, string> = {
  improving: "text-green-600",
  stable: "text-gray-500",
  degrading: "text-red-600",
};

export default function TrendChart({ trends }: TrendChartProps) {
  if (trends.length === 0) {
    return null;
  }

  return (
    <Card title="Score Trends">
      <div className="space-y-3">
        {trends.map((trend) => {
          const latest = trend.scores[trend.scores.length - 1] ?? 0;
          const icon = DIRECTION_ICONS[trend.direction] ?? "?";
          const colorClass = DIRECTION_COLORS[trend.direction] ?? "text-gray-500";

          return (
            <div key={trend.evaluator} className="flex items-center gap-3">
              <span className="w-40 truncate text-sm text-gray-700 dark:text-gray-300">
                {trend.evaluator.replace(/_/g, " ")}
              </span>

              {/* Mini sparkline */}
              <div className="flex flex-1 items-end gap-0.5" style={{ height: 24 }}>
                {trend.scores.map((score, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t bg-brand-400"
                    style={{
                      height: `${Math.max(score * 100, 4)}%`,
                      minHeight: 2,
                    }}
                  />
                ))}
              </div>

              <span className="w-12 text-right text-sm font-medium">
                {(latest * 100).toFixed(0)}%
              </span>
              <span className={`w-6 text-center text-lg font-bold ${colorClass}`}>
                {icon}
              </span>
              <span className="w-16 text-right text-xs text-gray-400">
                {trend.change_pct > 0 ? "+" : ""}
                {trend.change_pct.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

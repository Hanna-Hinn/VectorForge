import Card from "../ui/Card";
import type { LatencyStats } from "../../types/models";
import { formatMs } from "../../utils/format";

interface LatencyChartProps {
  stats: LatencyStats;
}

export default function LatencyChart({ stats }: LatencyChartProps) {
  const bars = [
    { label: "Min", value: stats.min_ms },
    { label: "Avg", value: stats.avg_ms },
    { label: "p50", value: stats.p50_ms },
    { label: "p95", value: stats.p95_ms },
    { label: "Max", value: stats.max_ms },
  ];

  const maxVal = Math.max(...bars.map((b) => b.value), 1);

  return (
    <Card title="Latency Distribution">
      <div className="flex items-end gap-3" style={{ height: 120 }}>
        {bars.map((bar) => (
          <div key={bar.label} className="flex flex-1 flex-col items-center gap-1">
            <span className="text-xs text-gray-500">{formatMs(bar.value)}</span>
            <div
              className="w-full rounded-t bg-brand-500"
              style={{ height: `${(bar.value / maxVal) * 100}%`, minHeight: 4 }}
            />
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
              {bar.label}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-gray-400 text-right">
        {stats.sample_count} samples
      </p>
    </Card>
  );
}

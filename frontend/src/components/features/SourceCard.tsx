import type { SourceCitation } from "../../types/models";
import Card from "../ui/Card";
import { t } from "../../i18n";

interface SourceCardProps {
  sources: SourceCitation[];
}

export default function SourceCard({ sources }: SourceCardProps) {
  if (sources.length === 0) return null;

  return (
    <Card title={t("query.sources")}>
      <ul className="space-y-2">
        {sources.map((src, i) => (
          <li
            key={`${src.chunk_index}-${i}`}
            className="rounded border border-gray-200 p-2 text-xs dark:border-gray-700"
          >
            <div className="flex items-center justify-between">
              <span className="font-medium text-gray-700 dark:text-gray-300">
                {src.document_source}
              </span>
              <span className="text-gray-400">{src.score.toFixed(3)}</span>
            </div>
            <p className="mt-1 line-clamp-2 text-gray-500">{src.snippet}</p>
          </li>
        ))}
      </ul>
    </Card>
  );
}

import { Link } from "react-router-dom";
import type { Collection } from "../../types/models";
import Badge from "../ui/Badge";
import Button from "../ui/Button";
import { t } from "../../i18n";
import { formatDate } from "../../utils/format";

interface CollectionCardProps {
  collection: Collection;
  onDelete: (id: string) => void;
}

export default function CollectionCard({
  collection,
  onDelete,
}: CollectionCardProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm transition-shadow hover:shadow-md dark:border-gray-700 dark:bg-gray-900">
      <div className="flex items-start justify-between">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-base font-semibold text-gray-900 dark:text-gray-100">
            {collection.name}
          </h3>
          {collection.description && (
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {collection.description}
            </p>
          )}
          <p className="mt-2 text-xs text-gray-400">
            {formatDate(collection.created_at)}
          </p>
        </div>
        <Badge variant="success">{t("status.healthy")}</Badge>
      </div>

      <div className="mt-4 flex gap-2">
        <Link
          to={`/collections/${collection.id}/documents`}
          className="text-sm text-brand-600 hover:underline"
        >
          {t("documents.title")}
        </Link>
        <Link
          to={`/collections/${collection.id}/query`}
          className="text-sm text-brand-600 hover:underline"
        >
          {t("query.title")}
        </Link>
        <Link
          to={`/collections/${collection.id}/analytics`}
          className="text-sm text-brand-600 hover:underline"
        >
          {t("analytics.title")}
        </Link>
        <div className="ml-auto">
          <Button
            variant="danger"
            size="sm"
            onClick={() => onDelete(collection.id)}
          >
            {t("common.delete")}
          </Button>
        </div>
      </div>
    </div>
  );
}

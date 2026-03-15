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
    <div className="group rounded-xl border border-gray-200 bg-white p-5 shadow-sm transition-all duration-200 hover:shadow-md hover:border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:hover:border-gray-600">
      <div className="flex items-start justify-between">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-base font-semibold text-gray-900 dark:text-gray-100">
            {collection.name}
          </h3>
          {collection.description && (
            <p className="mt-1 line-clamp-2 text-sm text-gray-500 dark:text-gray-400">
              {collection.description}
            </p>
          )}
          <p className="mt-2 text-xs text-gray-400">
            {formatDate(collection.created_at)}
          </p>
        </div>
        <Badge variant="success">{t("status.healthy")}</Badge>
      </div>

      <div className="mt-4 flex items-center gap-1 border-t border-gray-100 pt-3 dark:border-gray-800">
        <Link
          to={`/collections/${collection.id}/documents`}
          className="rounded-md px-2.5 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-brand-50 hover:text-brand-700 dark:text-gray-400 dark:hover:bg-brand-900/20 dark:hover:text-brand-400"
        >
          {t("documents.title")}
        </Link>
        <Link
          to={`/collections/${collection.id}/query`}
          className="rounded-md px-2.5 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-brand-50 hover:text-brand-700 dark:text-gray-400 dark:hover:bg-brand-900/20 dark:hover:text-brand-400"
        >
          {t("query.title")}
        </Link>
        <Link
          to={`/collections/${collection.id}/analytics`}
          className="rounded-md px-2.5 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-brand-50 hover:text-brand-700 dark:text-gray-400 dark:hover:bg-brand-900/20 dark:hover:text-brand-400"
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

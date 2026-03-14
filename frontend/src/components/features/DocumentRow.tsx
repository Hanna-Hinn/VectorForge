import type { Document } from "../../types/models";
import Badge from "../ui/Badge";
import Button from "../ui/Button";
import { t } from "../../i18n";
import { formatDate, formatBytes } from "../../utils/format";

const statusVariant: Record<string, "default" | "success" | "warning" | "error"> = {
  pending: "default",
  processing: "warning",
  indexed: "success",
  failed: "error",
};

interface DocumentRowProps {
  document: Document;
  onDelete: (id: string) => void;
}

export default function DocumentRow({ document, onDelete }: DocumentRowProps) {
  return (
    <tr className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
      <td className="px-4 py-3 text-sm">
        <span className="font-medium">{document.source_uri}</span>
      </td>
      <td className="px-4 py-3 text-sm text-gray-500">
        {document.content_type}
      </td>
      <td className="px-4 py-3 text-sm">
        <Badge variant={statusVariant[document.status] ?? "default"}>
          {document.status}
        </Badge>
      </td>
      <td className="px-4 py-3 text-sm text-gray-500">
        {formatBytes(document.content_size_bytes)}
      </td>
      <td className="px-4 py-3 text-sm text-gray-400">
        {formatDate(document.created_at)}
      </td>
      <td className="px-4 py-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onDelete(document.id)}
        >
          {t("common.delete")}
        </Button>
      </td>
    </tr>
  );
}

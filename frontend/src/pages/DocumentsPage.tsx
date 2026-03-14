import { useState } from "react";
import { useParams } from "react-router-dom";
import { useDocuments, useIngestDocument, useDeleteDocument } from "../hooks/useDocuments";
import { useCollection } from "../hooks/useCollections";
import DocumentRow from "../components/features/DocumentRow";
import UploadDialog from "../components/features/UploadDialog";
import Button from "../components/ui/Button";
import Spinner from "../components/ui/Spinner";
import Header from "../components/layout/Header";
import { t } from "../i18n";
import { showToast } from "../components/ui/Toast";

export default function DocumentsPage() {
  const { collectionId = "" } = useParams();
  const { data: collection } = useCollection(collectionId);
  const { data, isLoading } = useDocuments(collectionId);
  const ingestMutation = useIngestDocument(collectionId);
  const deleteMutation = useDeleteDocument(collectionId);
  const [uploadOpen, setUploadOpen] = useState(false);

  const handleIngest = (source: string) => {
    ingestMutation.mutate(
      { source },
      {
        onSuccess: () => {
          setUploadOpen(false);
          showToast("Document ingested", "success");
        },
        onError: (err) => showToast(err.message, "error"),
      },
    );
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id, {
      onSuccess: () => showToast("Document deleted", "success"),
      onError: (err) => showToast(err.message, "error"),
    });
  };

  const title = collection
    ? `${collection.name} — ${t("documents.title")}`
    : t("documents.title");

  return (
    <>
      <Header title={title} />
      <div className="p-6">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-gray-500">
            {data?.total ?? 0} document(s)
          </p>
          <Button onClick={() => setUploadOpen(true)}>
            {t("documents.ingest.title")}
          </Button>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-12">
            <Spinner size="lg" />
          </div>
        ) : data && data.documents.length > 0 ? (
          <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800">
                <tr>
                  <th className="px-4 py-3 font-medium text-gray-600 dark:text-gray-400">Source</th>
                  <th className="px-4 py-3 font-medium text-gray-600 dark:text-gray-400">Type</th>
                  <th className="px-4 py-3 font-medium text-gray-600 dark:text-gray-400">Status</th>
                  <th className="px-4 py-3 font-medium text-gray-600 dark:text-gray-400">Size</th>
                  <th className="px-4 py-3 font-medium text-gray-600 dark:text-gray-400">Created</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {data.documents.map((doc) => (
                  <DocumentRow
                    key={doc.id}
                    document={doc}
                    onDelete={handleDelete}
                  />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="py-12 text-center text-gray-400">
            {t("documents.empty")}
          </p>
        )}
      </div>

      <UploadDialog
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onSubmit={handleIngest}
        loading={ingestMutation.isPending}
      />
    </>
  );
}

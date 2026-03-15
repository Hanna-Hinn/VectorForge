import { useState } from "react";
import { useParams } from "react-router-dom";
import { useDocuments, useUploadDocument, useDeleteDocument } from "../hooks/useDocuments";
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
  const uploadMutation = useUploadDocument(collectionId);
  const deleteMutation = useDeleteDocument(collectionId);
  const [uploadOpen, setUploadOpen] = useState(false);

  const handleUpload = (file: File) => {
    uploadMutation.mutate(file, {
      onSuccess: () => {
        setUploadOpen(false);
        showToast("Document uploaded", "success");
      },
      onError: (err) => showToast(err.message, "error"),
    });
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
        <div className="mb-6 flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {data?.total ?? 0} document(s)
            </p>
          </div>
          <Button onClick={() => setUploadOpen(true)}>
            <svg className="-ml-0.5 mr-1.5 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
            {t("documents.upload.title")}
          </Button>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-16">
            <Spinner size="lg" />
          </div>
        ) : data && data.documents.length > 0 ? (
          <div className="overflow-hidden rounded-xl border border-gray-200 shadow-sm dark:border-gray-700">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-gray-200 bg-gray-50/80 dark:border-gray-700 dark:bg-gray-800/80">
                <tr>
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Source</th>
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Type</th>
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Status</th>
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Size</th>
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Created</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700/50">
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
          <div className="flex flex-col items-center justify-center py-20">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gray-100 dark:bg-gray-800">
              <svg className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
              {t("documents.empty")}
            </p>
            <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">
              {t("documents.empty.hint")}
            </p>
            <Button
              className="mt-4"
              size="sm"
              onClick={() => setUploadOpen(true)}
            >
              <svg className="-ml-0.5 mr-1.5 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
              {t("documents.upload.title")}
            </Button>
          </div>
        )}
      </div>

      <UploadDialog
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onSubmit={handleUpload}
        loading={uploadMutation.isPending}
      />
    </>
  );
}

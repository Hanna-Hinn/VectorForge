import { useState } from "react";
import {
  useCollections,
  useCreateCollection,
  useDeleteCollection,
} from "../hooks/useCollections";
import CollectionCard from "../components/features/CollectionCard";
import Button from "../components/ui/Button";
import Dialog from "../components/ui/Dialog";
import Input from "../components/ui/Input";
import Spinner from "../components/ui/Spinner";
import Header from "../components/layout/Header";
import { t } from "../i18n";
import { showToast } from "../components/ui/Toast";

export default function CollectionsPage() {
  const { data: collections, isLoading } = useCollections();
  const createMutation = useCreateCollection();
  const deleteMutation = useDeleteCollection();

  const [dialogOpen, setDialogOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const handleCreate = () => {
    createMutation.mutate(
      { name, description: description || null },
      {
        onSuccess: () => {
          setDialogOpen(false);
          setName("");
          setDescription("");
          showToast("Collection created", "success");
        },
        onError: (err) => showToast(err.message, "error"),
      },
    );
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id, {
      onSuccess: () => showToast("Collection deleted", "success"),
      onError: (err) => showToast(err.message, "error"),
    });
  };

  return (
    <>
      <Header title={t("collections.title")} />
      <div className="p-6">
        <div className="mb-6 flex items-center justify-between">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {collections?.length ?? 0} collection(s)
          </p>
          <Button onClick={() => setDialogOpen(true)}>
            <svg className="-ml-0.5 mr-1.5 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
            {t("common.create")}
          </Button>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-16">
            <Spinner size="lg" />
          </div>
        ) : collections && collections.length > 0 ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {collections.map((col) => (
              <CollectionCard
                key={col.id}
                collection={col}
                onDelete={handleDelete}
              />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gray-100 dark:bg-gray-800">
              <svg className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z" />
              </svg>
            </div>
            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
              {t("collections.empty")}
            </p>
            <Button
              className="mt-4"
              size="sm"
              onClick={() => setDialogOpen(true)}
            >
              <svg className="-ml-0.5 mr-1.5 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
              {t("common.create")}
            </Button>
          </div>
        )}
      </div>

      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        title={t("collections.create.title")}
        actions={
          <>
            <Button variant="secondary" onClick={() => setDialogOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              onClick={handleCreate}
              loading={createMutation.isPending}
              disabled={!name.trim()}
            >
              {t("common.create")}
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <Input
            label={t("collections.create.name")}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-collection"
            autoFocus
          />
          <Input
            label={t("collections.create.description")}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Optional description"
          />
        </div>
      </Dialog>
    </>
  );
}

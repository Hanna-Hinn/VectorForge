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
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-gray-500">
            {collections?.length ?? 0} collection(s)
          </p>
          <Button onClick={() => setDialogOpen(true)}>
            {t("common.create")}
          </Button>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-12">
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
          <p className="py-12 text-center text-gray-400">
            {t("collections.empty")}
          </p>
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

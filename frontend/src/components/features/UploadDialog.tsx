import { useState } from "react";
import Dialog from "../ui/Dialog";
import Input from "../ui/Input";
import Button from "../ui/Button";
import { t } from "../../i18n";

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (source: string) => void;
  loading: boolean;
}

export default function UploadDialog({
  open,
  onClose,
  onSubmit,
  loading,
}: UploadDialogProps) {
  const [source, setSource] = useState("");

  const handleSubmit = () => {
    if (!source.trim()) return;
    onSubmit(source);
    setSource("");
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={t("documents.ingest.title")}
      actions={
        <>
          <Button variant="secondary" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button onClick={handleSubmit} loading={loading} disabled={!source.trim()}>
            {t("common.create")}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <Input
          label={t("documents.ingest.sourceUri")}
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="e.g. /path/to/document.md"
        />
      </div>
    </Dialog>
  );
}

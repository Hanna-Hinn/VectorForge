import { useCallback, useRef, useState, type DragEvent } from "react";
import Dialog from "../ui/Dialog";
import Button from "../ui/Button";
import { t } from "../../i18n";

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (file: File) => void;
  loading: boolean;
}

const ACCEPTED_EXTENSIONS = [".txt", ".md", ".html", ".htm", ".pdf", ".xml"];

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadDialog({
  open,
  onClose,
  onSubmit,
  loading,
}: UploadDialogProps) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File | null) => {
    setFile(f);
    setDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const dropped = e.dataTransfer.files[0];
      if (dropped) handleFile(dropped);
    },
    [handleFile],
  );

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
  }, []);

  const handleSubmit = () => {
    if (!file) return;
    onSubmit(file);
    setFile(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleClear = () => {
    setFile(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={t("documents.upload.title")}
      actions={
        <>
          <Button variant="secondary" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button onClick={handleSubmit} loading={loading} disabled={!file}>
            {t("documents.upload.submit")}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        {/* Drop zone */}
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
          className={`group relative cursor-pointer rounded-xl border-2 border-dashed
            p-8 text-center transition-all duration-200
            ${
              dragging
                ? "border-brand-500 bg-brand-50 dark:bg-brand-900/20"
                : file
                  ? "border-green-400 bg-green-50 dark:border-green-600 dark:bg-green-900/10"
                  : "border-gray-300 bg-gray-50/50 hover:border-brand-400 hover:bg-brand-50/50 dark:border-gray-600 dark:bg-gray-800/30 dark:hover:border-brand-500 dark:hover:bg-brand-900/10"
            }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS.join(",")}
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            className="hidden"
            aria-label={t("documents.upload.chooseFile")}
          />

          {file ? (
            <div className="space-y-2">
              {/* File icon */}
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-green-100 dark:bg-green-900/30">
                <svg className="h-6 w-6 text-green-600 dark:text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {file.name}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {formatFileSize(file.size)}
              </p>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleClear();
                }}
                className="mt-1 text-xs font-medium text-red-600 hover:text-red-700 dark:text-red-400"
              >
                {t("documents.upload.remove")}
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Upload icon */}
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-brand-100 transition-colors group-hover:bg-brand-200 dark:bg-brand-900/30 dark:group-hover:bg-brand-900/50">
                <svg className="h-6 w-6 text-brand-600 dark:text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {t("documents.upload.dragDrop")}
                </p>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  {t("documents.upload.or")}{" "}
                  <span className="font-medium text-brand-600 dark:text-brand-400">
                    {t("documents.upload.browse")}
                  </span>
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Accepted formats */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-gray-400">{t("documents.upload.accepted")}:</span>
          {ACCEPTED_EXTENSIONS.map((ext) => (
            <span
              key={ext}
              className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-medium uppercase
                text-gray-500 dark:bg-gray-800 dark:text-gray-400"
            >
              {ext}
            </span>
          ))}
        </div>
      </div>
    </Dialog>
  );
}

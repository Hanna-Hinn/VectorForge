import { useEffect, useRef, type ReactNode } from "react";
import Button from "./Button";
import { t } from "../../i18n";

interface DialogProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  actions?: ReactNode;
}

export default function Dialog({
  open,
  onClose,
  title,
  children,
  actions,
}: DialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;

    if (open) {
      el.showModal();
    } else {
      el.close();
    }
  }, [open]);

  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;

    const handleClose = () => onClose();
    el.addEventListener("close", handleClose);
    return () => el.removeEventListener("close", handleClose);
  }, [onClose]);

  if (!open) return null;

  return (
    <dialog
      ref={dialogRef}
      className="w-full max-w-lg rounded-lg border border-gray-200 bg-white
        p-0 shadow-xl backdrop:bg-black/50 dark:border-gray-700 dark:bg-gray-900"
      aria-labelledby="dialog-title"
    >
      <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4 dark:border-gray-700">
        <h2 id="dialog-title" className="text-lg font-semibold">
          {title}
        </h2>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          aria-label={t("common.close")}
        >
          ✕
        </Button>
      </div>
      <div className="px-6 py-4">{children}</div>
      {actions && (
        <div className="flex justify-end gap-2 border-t border-gray-200 px-6 py-4 dark:border-gray-700">
          {actions}
        </div>
      )}
    </dialog>
  );
}

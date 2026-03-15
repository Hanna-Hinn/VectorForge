import { useCallback, useEffect, useState } from "react";

type ToastVariant = "success" | "error" | "info";

interface ToastMessage {
  id: number;
  variant: ToastVariant;
  text: string;
}

let nextId = 0;
const listeners = new Set<(msg: ToastMessage) => void>();

/** Show a toast from anywhere. */
export function showToast(text: string, variant: ToastVariant = "info") {
  const msg: ToastMessage = { id: nextId++, variant, text };
  listeners.forEach((fn) => fn(msg));
}

const variantStyles: Record<ToastVariant, string> = {
  success: "bg-green-600 text-white",
  error: "bg-red-600 text-white",
  info: "bg-gray-800 text-white",
};

export default function Toast() {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  useEffect(() => {
    const add = (msg: ToastMessage) => {
      setToasts((prev) => [...prev, msg]);
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== msg.id));
      }, 4000);
    };
    listeners.add(add);
    return () => {
      listeners.delete(add);
    };
  }, []);

  const remove = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div
      className="fixed bottom-4 right-4 z-50 flex flex-col gap-2"
      aria-live="polite"
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`flex items-center gap-3 rounded-xl px-4 py-3 shadow-lg
            animate-slide-in ${variantStyles[toast.variant]}`}
          role="alert"
        >
          <span className="text-sm">{toast.text}</span>
          <button
            onClick={() => remove(toast.id)}
            className="ml-auto opacity-70 hover:opacity-100"
            aria-label="Dismiss"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}

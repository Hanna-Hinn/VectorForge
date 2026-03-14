import type { ReactNode } from "react";

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export default function Card({ title, children, className = "" }: CardProps) {
  return (
    <div
      className={`rounded-lg border border-gray-200 bg-white p-4 shadow-sm
        dark:border-gray-700 dark:bg-gray-900 ${className}`}
    >
      {title && (
        <h3 className="mb-3 text-sm font-semibold text-gray-600 dark:text-gray-400">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

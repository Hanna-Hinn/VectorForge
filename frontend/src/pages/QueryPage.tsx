import { useRef, useState, type FormEvent } from "react";
import { useParams } from "react-router-dom";
import { useCollection } from "../hooks/useCollections";
import { useStreamQuery } from "../hooks/useQuery";
import ChatMessageBubble from "../components/features/ChatMessage";
import Spinner from "../components/ui/Spinner";
import Header from "../components/layout/Header";
import { t } from "../i18n";

export default function QueryPage() {
  const { collectionId = "" } = useParams();
  const { data: collection } = useCollection(collectionId);
  const { messages, isStreaming, sendQuery, clearMessages } = useStreamQuery();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    void sendQuery(text, collectionId);
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  };

  const title = collection
    ? `${t("query.title")} — ${collection.name}`
    : t("query.title");

  return (
    <div className="flex h-full flex-col">
      <Header title={title} />

      {/* Message area */}
      <div className="flex-1 overflow-y-auto p-6">
        {messages.length === 0 ? (
          <p className="text-center text-gray-400 py-20">{t("query.empty")}</p>
        ) : (
          <div className="mx-auto max-w-2xl space-y-4">
            {messages.map((msg, i) => (
              <ChatMessageBubble key={i} message={msg} />
            ))}
            {isStreaming && (
              <div className="flex justify-start">
                <Spinner size="sm" />
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="border-t border-gray-200 p-4 dark:border-gray-700">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-2xl gap-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t("query.placeholder")}
            disabled={isStreaming}
            className="flex-1 rounded-md border border-gray-300 px-4 py-2 text-sm
              focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500
              dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
            aria-label={t("query.placeholder")}
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white
              hover:bg-brand-700 disabled:opacity-50"
          >
            {t("query.send")}
          </button>
          {messages.length > 0 && !isStreaming && (
            <button
              type="button"
              onClick={clearMessages}
              className="rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-600
                hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300"
            >
              Clear
            </button>
          )}
        </form>
      </div>
    </div>
  );
}

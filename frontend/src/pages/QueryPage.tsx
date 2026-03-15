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
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-50 dark:bg-brand-900/20">
              <svg className="h-8 w-8 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
              </svg>
            </div>
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{t("query.empty")}</p>
            <p className="mt-1 text-sm text-gray-500">{t("query.placeholder")}</p>
          </div>
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
            className="flex-1 rounded-xl border border-gray-300 px-4 py-2.5 text-sm
              focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500
              dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
            aria-label={t("query.placeholder")}
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="inline-flex items-center gap-1.5 rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-medium text-white
              transition-all duration-150 hover:bg-brand-700 active:scale-[0.97] disabled:opacity-50"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
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

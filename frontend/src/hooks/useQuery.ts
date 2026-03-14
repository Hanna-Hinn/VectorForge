/** Hook for RAG query with SSE streaming support. */

import { useCallback, useRef, useState } from "react";
import { streamQuery } from "../api/query";
import type { QueryRequest } from "../types/api";
import type { SourceCitation } from "../types/models";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: SourceCitation[];
  latencyMs?: number;
}

export function useStreamQuery() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendQuery = useCallback(
    async (queryText: string, collectionId: string, config?: Partial<QueryRequest>) => {
      setMessages((prev) => [
        ...prev,
        { role: "user", content: queryText },
        { role: "assistant", content: "" },
      ]);
      setIsStreaming(true);

      abortRef.current = new AbortController();

      await streamQuery(
        {
          query: queryText,
          collection_id: collectionId,
          ...config,
        },
        {
          onToken: (token) => {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { ...last, content: last.content + token };
              }
              return next;
            });
          },
          onDone: (latencyMs) => {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { ...last, latencyMs };
              }
              return next;
            });
            setIsStreaming(false);
          },
          onError: (error) => {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  content: `Error: ${error.message}`,
                };
              }
              return next;
            });
            setIsStreaming(false);
          },
        },
      );
    },
    [],
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, sendQuery, clearMessages };
}

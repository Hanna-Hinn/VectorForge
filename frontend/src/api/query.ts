/** Query API calls including SSE streaming. */

import { post, getApiKey } from "./client";
import type { QueryRequest, QueryResponse } from "../types/api";
import type { SourceCitation } from "../types/models";

export function query(data: QueryRequest): Promise<QueryResponse> {
  return post<QueryResponse>("/query", data);
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: (latencyMs: number) => void;
  onError: (error: Error) => void;
  onSources?: (sources: SourceCitation[]) => void;
}

export async function streamQuery(
  data: QueryRequest,
  callbacks: StreamCallbacks,
): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const key = getApiKey();
  if (key) {
    headers["X-Api-Key"] = key;
  }

  const response = await fetch("/api/query/stream", {
    method: "POST",
    headers,
    body: JSON.stringify(data),
  });

  if (!response.ok || !response.body) {
    callbacks.onError(new Error(`Stream failed: ${response.statusText}`));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    while (buffer.includes("\n\n")) {
      const boundary = buffer.indexOf("\n\n");
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      for (const line of chunk.split("\n")) {
        if (!line.startsWith("data: ")) continue;

        try {
          const event = JSON.parse(line.slice(6)) as Record<string, unknown>;
          switch (event.type) {
            case "token":
              callbacks.onToken(event.content as string);
              break;
            case "done":
              callbacks.onDone(event.latency_ms as number);
              break;
            case "error":
              callbacks.onError(new Error(event.message as string));
              break;
          }
        } catch {
          // Skip malformed events
        }
      }
    }
  }
}

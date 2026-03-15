import type { ChatMessage } from "../../hooks/useQuery";
import { formatMs } from "../../utils/format";

interface ChatMessageProps {
  message: ChatMessage;
}

export default function ChatMessageBubble({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm shadow-sm ${
          isUser
            ? "bg-brand-600 text-white"
            : "bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100"
        }`}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>
        {message.latencyMs != null && (
          <p className="mt-1 text-xs opacity-60">
            {formatMs(message.latencyMs)}
          </p>
        )}
      </div>
    </div>
  );
}

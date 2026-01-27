import { cn } from "@/lib/utils";
import { User, Bot } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatMessageProps {
  content: string;
  role: "user" | "assistant";
  isTyping?: boolean;
}

export function ChatMessage({ content, role, isTyping }: ChatMessageProps) {
  const isUser = role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 animate-fade-in",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <div
        className={cn(
          "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
          isUser ? "bg-primary" : "bg-accent"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-primary-foreground" />
        ) : (
          <Bot className="w-4 h-4 text-accent-foreground" />
        )}
      </div>

      <div
        className={cn(
          "max-w-[75%] rounded-2xl px-4 py-3 shadow-message",
          isUser
            ? "bg-chat-user-bg text-chat-user-fg rounded-br-md"
            : "bg-chat-assistant-bg text-chat-assistant-fg rounded-bl-md border border-border"
        )}
      >
        {isTyping ? (
          <div className="flex gap-1.5 py-1">
            <span className="w-2 h-2 bg-current rounded-full animate-typing" style={{ animationDelay: "0ms" }} />
            <span className="w-2 h-2 bg-current rounded-full animate-typing" style={{ animationDelay: "200ms" }} />
            <span className="w-2 h-2 bg-current rounded-full animate-typing" style={{ animationDelay: "400ms" }} />
          </div>
        ) : (
          <div className={cn(
            "text-sm leading-relaxed prose prose-sm dark:prose-invert max-w-none",
            "prose-headings:mt-3 prose-headings:mb-2 prose-headings:font-semibold prose-h3:text-base",
            "prose-p:my-2 prose-ul:my-2 prose-li:my-1",
            "prose-strong:font-semibold",
            isUser 
              ? "prose-headings:text-chat-user-fg prose-p:text-chat-user-fg prose-ul:text-chat-user-fg prose-li:text-chat-user-fg prose-strong:text-chat-user-fg prose-code:text-chat-user-fg"
              : "prose-headings:text-chat-assistant-fg prose-p:text-chat-assistant-fg prose-ul:text-chat-assistant-fg prose-li:text-chat-assistant-fg prose-strong:text-chat-assistant-fg"
          )}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

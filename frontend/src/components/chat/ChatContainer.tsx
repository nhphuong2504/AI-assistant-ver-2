import { useState, useRef, useEffect } from "react";
import { MessageSquare } from "lucide-react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { askQuestion } from "@/lib/api";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
}

export function ChatContainer() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content: "Hello! I'm your business assistant. How can I help you today?",
      role: "assistant",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: "user",
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const answer = await askQuestion(content);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: answer,
        role: "assistant",
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "Sorry, I encountered an error. Please try again.",
        role: "assistant",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background rounded-2xl shadow-chat overflow-hidden border border-border">
      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-4 bg-chat-header-bg">
        <div className="w-10 h-10 rounded-full bg-accent flex items-center justify-center">
          <MessageSquare className="w-5 h-5 text-accent-foreground" />
        </div>
        <div>
          <h1 className="font-semibold text-primary-foreground">Business Assistant</h1>
          <p className="text-xs text-primary-foreground/70">Always here to help</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {messages.map((message) => (
          <ChatMessage key={message.id} content={message.content} role={message.role} />
        ))}
        {isLoading && (
          <ChatMessage content="" role="assistant" isTyping={true} />
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}

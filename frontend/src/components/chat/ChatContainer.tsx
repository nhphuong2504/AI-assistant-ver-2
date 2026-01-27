import { useState, useRef, useEffect } from "react";
import { MessageSquare, HelpCircle } from "lucide-react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { askQuestion } from "@/lib/api";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
}

export function ChatContainer() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content: "Hi, I analyze your customer data to support growth and retention decisions. I can answer questions using SQL queries, predictive models (CLV, churn), and segmentation analysis.",
      role: "assistant",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [useMemory, setUseMemory] = useState(true);
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
      const answer = await askQuestion(content, useMemory);
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
      <div className="flex items-center justify-between px-5 py-4 bg-chat-header-bg">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-accent flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-accent-foreground" />
          </div>
          <div>
            <h1 className="font-semibold text-primary-foreground">AI analyst for growth & retention</h1>
            <p className="text-xs text-primary-foreground/70">Analysis based on historical data up to December 9th, 2011.</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="memory-toggle" className="text-xs text-primary-foreground/70 cursor-pointer">
            Use memory
          </Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <HelpCircle className="w-3.5 h-3.5 text-primary-foreground/50 hover:text-primary-foreground/80 cursor-help" aria-label="What does memory do?" />
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-[240px]">
              When on, the assistant remembers earlier messages in this conversation and can refer to them (e.g. &quot;as I showed above&quot;). Turn off for a clean slate or to keep each question independent.
            </TooltipContent>
          </Tooltip>
          <Switch
            id="memory-toggle"
            checked={useMemory}
            onCheckedChange={setUseMemory}
            disabled={isLoading}
          />
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
        {/* Example Prompts - Show only when there's only the welcome message */}
        {messages.length === 1 && !isLoading && (
          <div className="space-y-2 mt-4">
            <p className="text-xs text-muted-foreground mb-3">Try asking:</p>
            <div className="grid grid-cols-1 gap-2">
              <Button
                variant="outline"
                className="justify-start text-left h-auto py-3 px-4 text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                onClick={() => handleSend("What were the top 10 products by revenue on November 2011?")}
                disabled={isLoading}
              >
                What were the top 10 products by revenue on November 2011?
              </Button>
              <Button
                variant="outline"
                className="justify-start text-left h-auto py-3 px-4 text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                onClick={() => handleSend("Which customers are currently at the highest risk of churning?")}
                disabled={isLoading}
              >
                Which customers are currently at the highest risk of churning?
              </Button>
              <Button
                variant="outline"
                className="justify-start text-left h-auto py-3 px-4 text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                onClick={() => handleSend("Show me the top 10 customers by predicted CLV.")}
                disabled={isLoading}
              >
                Show me the top 10 customers by predicted CLV.
              </Button>
              <Button
                variant="outline"
                className="justify-start text-left h-auto py-3 px-4 text-sm font-normal hover:bg-accent hover:text-accent-foreground"
                onClick={() => handleSend("Segment our user base and suggest retention actions for each group.")}
                disabled={isLoading}
              >
                Segment our user base and suggest retention actions for each group.
              </Button>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={isLoading} />
      {/* Footer */}
      <div className="px-5 py-3 bg-muted/50 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">
          Built by Phuong H. Nguyen · Data Science & Machine Learning
        </p>
      </div>
    </div>
  );
}

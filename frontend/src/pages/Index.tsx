import { ChatContainer } from "@/components/chat/ChatContainer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4 md:p-8">
      <div className="w-full max-w-2xl h-[600px] md:h-[700px]">
        <ChatContainer />
      </div>
    </div>
  );
};

export default Index;

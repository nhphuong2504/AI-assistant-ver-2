import { ChatContainer } from "@/components/chat/ChatContainer";


const Index = () => {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-4xl h-[85vh] min-h-[500px] shadow-xl rounded-xl overflow-hidden">
        <ChatContainer />
      </div>
    </div>
  );
};

export default Index;

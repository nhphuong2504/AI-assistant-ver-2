const API_BASE_URL = "http://127.0.0.1:8000";

export async function askQuestion(
  question: string,
  useMemory: boolean = true,
  threadId: string = "default"
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/ask-langchain`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      use_memory: useMemory,
      thread_id: threadId,
    }),
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`);
  }

  const data = await response.json();
  return data.answer;
}

export async function clearMemory(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/ask-langchain/clear-memory`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`);
  }
}


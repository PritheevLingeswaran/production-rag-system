import { api } from "./client";
import type { Message, QueryResponse, Session, SessionDetail } from "../types";

export const chatApi = {
  query: (question: string, session_id?: string) =>
    api.post<QueryResponse>("/chat/query", {
      question,
      session_id,
      retrieval_mode: "hybrid_rrf",
      top_k: 5,
    }),

  getSessions: () =>
    api.get<{ sessions: Session[] }>("/chat/sessions").then((r) => r.sessions),

  getSession: (id: string) =>
    api.get<SessionDetail>(`/chat/sessions/${id}`).then((session) => ({
      ...session,
      messages: session.messages.map(normalizeMessage),
    })),

  deleteSession: (id: string) =>
    api.delete<void>(`/chat/sessions/${id}`),
};

function normalizeMessage(message: Message): Message {
  return {
    ...message,
    role: message.role === "assistant" ? "assistant" : "user",
    citations: message.citations ?? [],
    sources: message.sources ?? [],
    metadata: message.metadata ?? {},
  };
}

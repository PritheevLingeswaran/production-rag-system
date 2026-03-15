import { API_BASE_URL } from "@/lib/config";
import type {
  ChatQueryResponse,
  ChatSessionDetail,
  ChatSession,
  DashboardPayload,
  DocumentDetail,
  DocumentItem,
  SettingsPayload,
  SummaryPayload
} from "@/types/api";

function authHeaders() {
  if (typeof window === "undefined") {
    return {};
  }

  const raw = window.localStorage.getItem("rag-smart-qa-auth-user");
  if (!raw) {
    return {};
  }

  try {
    const parsed = JSON.parse(raw) as { userId?: string };
    return parsed.userId ? { "x-user-id": parsed.userId } : {};
  } catch {
    return {};
  }
}

function mergeHeaders(init?: RequestInit) {
  const headers = new Headers(init?.headers);
  Object.entries(authHeaders()).forEach(([key, value]) => {
    if (value) {
      headers.set(key, value);
    }
  });
  return headers;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: mergeHeaders(init),
    cache: "no-store"
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  dashboard: () => request<DashboardPayload>("/api/v1/dashboard"),
  documents: () => request<{ documents: DocumentItem[] }>("/api/v1/documents"),
  document: (id: string) => request<DocumentDetail>(`/api/v1/documents/${id}`),
  documentSummary: (id: string) => request<SummaryPayload>(`/api/v1/documents/${id}/summary`),
  deleteDocument: (id: string) =>
    request<DocumentItem>(`/api/v1/documents/${id}`, { method: "DELETE" }),
  reindexDocument: (id: string) =>
    request<{ document: DocumentDetail }>(`/api/v1/documents/${id}/reindex`, { method: "POST" }),
  chatSessions: () => request<{ sessions: ChatSession[] }>("/api/v1/chat/sessions"),
  chatSession: (id: string) => request<ChatSessionDetail>(`/api/v1/chat/sessions/${id}`),
  deleteChatSession: (id: string) =>
    request<{ deleted: boolean }>(`/api/v1/chat/sessions/${id}`, { method: "DELETE" }),
  chatQuery: (body: {
    question: string;
    session_id?: string;
    retrieval_mode: string;
    top_k: number;
  }) =>
    request<ChatQueryResponse>("/api/v1/chat/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }),
  settings: () => request<SettingsPayload>("/api/v1/settings")
};

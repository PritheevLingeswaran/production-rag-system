import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { ToastProvider } from "@/components/toast-provider";
import { ChatPanel } from "@/features/chat/chat-panel";
import { api } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  api: {
    chatSession: vi.fn(),
    chatQuery: vi.fn(),
    chatSessions: vi.fn()
  }
}));

function renderChatPanel() {
  const client = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      },
      mutations: {
        retry: false
      }
    }
  });

  render(
    <QueryClientProvider client={client}>
      <ToastProvider>
        <ChatPanel />
      </ToastProvider>
    </QueryClientProvider>
  );
}

test("submits a chat request and exposes citation details", async () => {
  vi.mocked(api.chatSession).mockResolvedValue({
    id: "session-1",
    owner_id: "local-user",
    title: "Monitoring",
    created_at: "2026-03-13T00:00:00Z",
    updated_at: "2026-03-13T00:00:00Z",
    messages: []
  });
  vi.mocked(api.chatQuery).mockResolvedValue({
    session_id: "session-1",
    answer: "The monitoring stack exposes request metrics. [chunk-1]",
    confidence: 0.92,
    refusal: { is_refusal: false, reason: "" },
    citations: [
      {
        id: "citation-1",
        document_id: "doc-1",
        chunk_id: "chunk-1",
        source: "operations.md",
        page: 1,
        excerpt: "Prometheus metrics are exported at /metrics.",
        score: 0.94,
        created_at: "2026-03-13T00:00:00Z"
      }
    ],
    sources: [
      {
        chunk_id: "chunk-1",
        source: "operations.md",
        page: 1,
        score: 0.94,
        text: "Prometheus metrics are exported at /metrics."
      }
    ],
    timing: {
      total_latency_ms: 18.1,
      retrieval_latency_ms: 5.2,
      generation_latency_ms: 12.9
    }
  });

  renderChatPanel();

  await userEvent.type(
    screen.getByPlaceholderText(/Ask a question about the uploaded documents/i),
    "How is monitoring implemented?"
  );
  await userEvent.click(screen.getByRole("button", { name: /send/i }));

  await waitFor(() => {
    expect(api.chatQuery).toHaveBeenCalledWith({
      question: "How is monitoring implemented?",
      session_id: undefined,
      retrieval_mode: "hybrid_rrf",
      top_k: 8
    });
  });

  expect(await screen.findByText(/The monitoring stack exposes request metrics/i)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /chunk-1/i }));
  expect(await screen.findByText(/Prometheus metrics are exported at \/metrics/i)).toBeInTheDocument();
});

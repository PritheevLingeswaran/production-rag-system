import { useEffect, useRef, useState, useCallback } from "react";
import { chatApi } from "../../api/chat";
import type { Message } from "../../types";
import Sources from "../Sources/Sources";
import Composer from "../Composer/Composer";
import "./ChatPanel.css";

interface Props {
  sessionId?: string;
  onSessionCreated: (id: string) => void;
}

type ChatStatus = "idle" | "searching" | "generating" | "error";

export default function ChatPanel({ sessionId, onSessionCreated }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const activeSession = useRef<string | undefined>(sessionId);

  useEffect(() => {
    activeSession.current = sessionId;
    setError(null);

    if (!sessionId) {
      setMessages([]);
      setStatus("idle");
      return;
    }

    let cancelled = false;
    chatApi.getSession(sessionId).then((detail) => {
      if (!cancelled) setMessages(detail.messages);
    }).catch(() => {
      if (!cancelled) setError("Couldn't load this conversation.");
    });

    return () => { cancelled = true; };
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  const send = useCallback(async (question: string) => {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      session_id: activeSession.current ?? "pending-session",
      role: "user",
      content: question,
      created_at: new Date().toISOString(),
    };

    setMessages((m) => [...m, userMsg]);
    setStatus("searching");
    setError(null);

    try {
      setStatus("generating");
      const res = await chatApi.query(question, activeSession.current);

      if (!activeSession.current) {
        activeSession.current = res.session_id;
        onSessionCreated(res.session_id);
      }

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        session_id: res.session_id,
        role: "assistant",
        content: res.answer,
        confidence: res.confidence,
        refusal: res.refusal.is_refusal,
        latency_ms: res.timing.total_latency_ms,
        created_at: new Date().toISOString(),
        metadata: {
          retrieval_latency_ms: res.timing.retrieval_latency_ms,
          generation_latency_ms: res.timing.generation_latency_ms,
          llm_tokens_in: res.timing.llm_tokens_in,
          llm_tokens_out: res.timing.llm_tokens_out,
          llm_cost_usd: res.timing.llm_cost_usd,
        },
        citations: res.citations,
        sources: res.sources,
      };

      setMessages((m) => [...m, assistantMsg]);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      const text = err instanceof Error ? err.message : "Something went wrong.";
      setError(text);
    }
  }, [onSessionCreated]);

  const isEmpty = messages.length === 0 && status === "idle";

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {isEmpty && <EmptyState />}

        {messages.map((msg) => (
          <MessageRow key={msg.id} message={msg} />
        ))}

        {(status === "searching" || status === "generating") && (
          <StatusRow status={status} />
        )}

        {error && status === "error" && (
          <div className="chat-error">
            <AlertIcon />
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <Composer onSend={send} disabled={status !== "idle" && status !== "error"} />
    </div>
  );
}

function MessageRow({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`message-row ${isUser ? "message-row--user" : "message-row--assistant"}`}>
      {!isUser && (
        <div className="message-avatar">
          <AvatarIcon />
        </div>
      )}
      <div className="message-body">
        <div className="message-content">
          <MessageText content={message.content} />
        </div>
        {!isUser && (
          <MessageMeta message={message} />
        )}
        {message.sources && message.sources.length > 0 && (
          <Sources sources={message.sources} citations={message.citations ?? []} />
        )}
      </div>
      {isUser && <div className="message-spacer" />}
    </div>
  );
}

function MessageText({ content }: { content: string }) {
  // Render newlines as line breaks; keep it readable without a markdown parser
  return (
    <>
      {content.split("\n").map((line, i) => (
        <span key={i}>
          {line}
          {i < content.split("\n").length - 1 && <br />}
        </span>
      ))}
    </>
  );
}

function MessageMeta({ message }: { message: Message }) {
  const latency = typeof message.latency_ms === "number"
    ? `${Math.round(message.latency_ms)} ms`
    : null;
  const confidence = typeof message.confidence === "number"
    ? `${Math.round(message.confidence * 100)}% confidence`
    : null;
  const refusal = message.refusal ? "Refusal guardrail triggered" : null;
  const parts = [confidence, latency, refusal].filter(Boolean);

  if (parts.length === 0) return null;

  return <div className="message-meta">{parts.join(" · ")}</div>;
}

function StatusRow({ status }: { status: "searching" | "generating" }) {
  return (
    <div className="message-row message-row--assistant">
      <div className="message-avatar">
        <AvatarIcon />
      </div>
      <div className="status-indicator">
        <span className="status-dot" />
        <span className="status-dot" />
        <span className="status-dot" />
        <span className="status-label">
          {status === "searching" ? "Searching documents…" : "Generating answer…"}
        </span>
      </div>
    </div>
  );
}

function EmptyState() {
  const prompts = [
    "What are the key findings in the uploaded reports?",
    "Summarize the main argument of the latest document.",
    "Compare the conclusions across all documents.",
    "What does the data say about Q3 performance?",
  ];

  return (
    <div className="chat-empty">
      <div className="chat-empty-icon">
        <LogoMark />
      </div>
      <h2 className="chat-empty-title">Ask anything across your documents</h2>
      <p className="chat-empty-sub">
        Every answer is grounded in your uploaded content, with citations you can verify.
      </p>
      <div className="chat-empty-prompts">
        {prompts.map((p) => (
          <div key={p} className="chat-empty-prompt">
            {p}
          </div>
        ))}
      </div>
    </div>
  );
}

function AvatarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 20 20" fill="none">
      <rect x="2" y="2" width="7" height="7" rx="2" fill="var(--accent)" />
      <rect x="11" y="2" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.5" />
      <rect x="2" y="11" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.5" />
      <rect x="11" y="11" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.25" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M7 4.5v3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="7" cy="9.5" r="0.6" fill="currentColor" />
    </svg>
  );
}

function LogoMark() {
  return (
    <svg width="36" height="36" viewBox="0 0 20 20" fill="none">
      <rect x="2" y="2" width="7" height="7" rx="2" fill="var(--accent)" />
      <rect x="11" y="2" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.5" />
      <rect x="2" y="11" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.5" />
      <rect x="11" y="11" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.25" />
    </svg>
  );
}

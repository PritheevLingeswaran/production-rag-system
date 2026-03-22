import { useEffect, useState } from "react";
import { chatApi } from "../../api/chat";
import type { Session, View } from "../../types";
import "./Sidebar.css";

interface Props {
  refreshKey: number;
  activeView: View;
  activeSessionId?: string;
  onViewChange: (v: View) => void;
  onSessionSelect: (id: string) => void;
  onNewChat: () => void;
  mobileOpen: boolean;
  onMobileClose: () => void;
}

export default function Sidebar({
  refreshKey,
  activeView,
  activeSessionId,
  onViewChange,
  onSessionSelect,
  onNewChat,
  mobileOpen,
  onMobileClose,
}: Props) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const data = await chatApi.getSessions();
      setSessions(data);
    } catch {
      // non-fatal
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSessions();
  }, [activeSessionId, refreshKey]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await chatApi.deleteSession(id);
      setSessions((s) => s.filter((x) => x.id !== id));
    } catch {
      // ignore
    }
  };

  return (
    <>
      {mobileOpen && (
        <div className="sidebar-backdrop" onClick={onMobileClose} />
      )}
      <aside className={`sidebar ${mobileOpen ? "sidebar--open" : ""}`}>
        <div className="sidebar-header">
          <span className="sidebar-logo">
            <LogoMark />
            SmartQA
          </span>
          <button
            className="new-chat-btn"
            onClick={onNewChat}
            title="New conversation"
          >
            <PencilIcon />
          </button>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${activeView === "chat" && !activeSessionId ? "nav-item--active" : ""}`}
            onClick={() => { onNewChat(); }}
          >
            <ChatIcon />
            New chat
          </button>
          <button
            className={`nav-item ${activeView === "documents" ? "nav-item--active" : ""}`}
            onClick={() => onViewChange("documents")}
          >
            <DocIcon />
            Documents
          </button>
          <button
            className={`nav-item ${activeView === "settings" ? "nav-item--active" : ""}`}
            onClick={() => onViewChange("settings")}
          >
            <GearIcon />
            Settings
          </button>
        </nav>

        <div className="sidebar-divider" />

        <div className="sessions-section">
          <span className="sessions-label">Recent conversations</span>
          {loading ? (
            <div className="sessions-loading">
              {[1, 2, 3].map((i) => (
                <div key={i} className="session-skeleton" />
              ))}
            </div>
          ) : sessions.length === 0 ? (
            <p className="sessions-empty">No conversations yet.</p>
          ) : (
            <ul className="sessions-list">
              {sessions.map((s) => (
                <li key={s.id}>
                  <button
                      className={`session-item ${activeSessionId === s.id ? "session-item--active" : ""}`}
                      onClick={() => onSessionSelect(s.id)}
                    >
                    <span className="session-title">{s.title || "Untitled"}</span>
                    <span className="session-meta">
                      {formatDate(s.updated_at)}
                    </span>
                    <button
                      className="session-delete"
                      onClick={(e) => handleDelete(e, s.id)}
                      title="Delete"
                    >
                      <TrashIcon />
                    </button>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="sidebar-auth">
          <button
            className={`sidebar-auth-link ${activeView === "login" ? "sidebar-auth-link--active" : ""}`}
            onClick={() => onViewChange("login")}
          >
            <LoginIcon />
            Log in
          </button>
          <button
            className={`sidebar-auth-link ${activeView === "signup" ? "sidebar-auth-link--active" : ""}`}
            onClick={() => onViewChange("signup")}
          >
            <UserPlusIcon />
            Sign up
          </button>
        </div>
      </aside>
    </>
  );
}

function formatDate(iso: string) {
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 86_400_000) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (diff < 7 * 86_400_000) return d.toLocaleDateString([], { weekday: "short" });
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function LogoMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <rect x="2" y="2" width="7" height="7" rx="2" fill="var(--accent)" />
      <rect x="11" y="2" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.5" />
      <rect x="2" y="11" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.5" />
      <rect x="11" y="11" width="7" height="7" rx="2" fill="var(--accent)" opacity="0.25" />
    </svg>
  );
}

function PencilIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <path d="M11.5 2.5L12.5 3.5L4.5 11.5H3.5V10.5L11.5 2.5Z" stroke="currentColor" strokeWidth="1.25" strokeLinejoin="round" />
      <path d="M2 13H13" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <path d="M2 3.5A1.5 1.5 0 013.5 2h8A1.5 1.5 0 0113 3.5v6A1.5 1.5 0 0111.5 11H9l-1.5 2-1.5-2H3.5A1.5 1.5 0 012 9.5v-6z" stroke="currentColor" strokeWidth="1.25" />
    </svg>
  );
}

function DocIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <path d="M3 2h6l3 3v8a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.25" />
      <path d="M9 2v3h3" stroke="currentColor" strokeWidth="1.25" strokeLinejoin="round" />
      <path d="M5 7h5M5 9.5h3" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <circle cx="7.5" cy="7.5" r="2" stroke="currentColor" strokeWidth="1.25" />
      <path d="M7.5 1v1.5M7.5 12.5V14M1 7.5h1.5M12.5 7.5H14M2.9 2.9l1.06 1.06M11.04 11.04l1.06 1.06M2.9 12.1l1.06-1.06M11.04 3.96l1.06-1.06" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M2 3.5h9M5 3.5V2.5h3v1M5.5 6v3.5M7.5 6v3.5M3 3.5l.5 7.5h6L10 3.5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function LoginIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <path d="M9.5 2.5h2A1.5 1.5 0 0113 4v7a1.5 1.5 0 01-1.5 1.5h-2" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
      <path d="M7 10.5L10 7.5L7 4.5" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M10 7.5H2.5" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}

function UserPlusIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <circle cx="5.5" cy="5" r="2.25" stroke="currentColor" strokeWidth="1.25" />
      <path d="M1.75 12c.55-1.92 2.03-2.88 3.75-2.88S8.7 10.08 9.25 12" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
      <path d="M11.25 4.25v3.5M9.5 6h3.5" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}

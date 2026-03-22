import { useState } from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import ChatPanel from "./components/ChatPanel/ChatPanel";
import DocumentsView from "./components/DocumentsView/DocumentsView";
import SettingsView from "./components/SettingsView/SettingsView";
import type { View } from "./types";
import "./App.css";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessionsVersion, setSessionsVersion] = useState(0);

  return (
    <div className="app">
      <button
        className="mobile-menu-btn"
        onClick={() => setSidebarOpen((o) => !o)}
        aria-label="Toggle menu"
      >
        <MenuIcon />
      </button>

      <Sidebar
        refreshKey={sessionsVersion}
        activeView={view}
        activeSessionId={sessionId}
        onViewChange={(v) => {
          setView(v);
          setSidebarOpen(false);
        }}
        onSessionSelect={(id) => {
          setSessionId(id);
          setView("chat");
          setSidebarOpen(false);
        }}
        onNewChat={() => {
          setSessionId(undefined);
          setView("chat");
          setSidebarOpen(false);
        }}
        mobileOpen={sidebarOpen}
        onMobileClose={() => setSidebarOpen(false)}
      />

      <main className="main-content">
        {view === "chat" && (
          <ChatPanel
            sessionId={sessionId}
            onSessionCreated={(id) => {
              setSessionId(id);
              setSessionsVersion((current) => current + 1);
            }}
          />
        )}
        {view === "documents" && <DocumentsView />}
        {view === "settings" && <SettingsView />}
      </main>
    </div>
  );
}

function MenuIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <rect x="2" y="4" width="14" height="1.5" rx="0.75" fill="currentColor" />
      <rect x="2" y="8.25" width="14" height="1.5" rx="0.75" fill="currentColor" />
      <rect x="2" y="12.5" width="14" height="1.5" rx="0.75" fill="currentColor" />
    </svg>
  );
}

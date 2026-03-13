"use client";

import { useQuery } from "@tanstack/react-query";
import {
  BookOpen,
  Bot,
  Command,
  Database,
  MessageSquareText,
  Settings2,
  Sparkles,
  Zap
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { CommandPalette } from "@/components/command-palette";
import { StatusBadge } from "@/components/status-badge";
import { TopBar } from "@/components/top-bar";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const nav = [
  { href: "/", label: "Dashboard", icon: Sparkles },
  { href: "/chat", label: "Chat", icon: MessageSquareText },
  { href: "/knowledge-base", label: "Knowledge Base", icon: Database },
  { href: "/summaries", label: "Summaries", icon: BookOpen },
  { href: "/settings", label: "Settings", icon: Settings2 }
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const sessionsQuery = useQuery({
    queryKey: ["chat-sessions"],
    queryFn: () => api.chatSessions()
  });
  const documentsQuery = useQuery({
    queryKey: ["documents"],
    queryFn: api.documents
  });

  const heading = useMemo(() => {
    const current = nav.find((item) => item.href === pathname);
    return current?.label ?? "Workspace";
  }, [pathname]);

  const subtitle = useMemo(() => {
    if (pathname === "/chat") {
      return "Grounded answers with citations, retrieval controls, and source inspection.";
    }
    if (pathname === "/knowledge-base") {
      return "Ingest, index, and curate the corpus that powers your assistant.";
    }
    if (pathname === "/summaries") {
      return "Review cached document briefings, insights, and answerability context.";
    }
    if (pathname === "/settings") {
      return "Runtime defaults, model configuration, and workspace feature flags.";
    }
    return "A premium AI workspace for document retrieval, summaries, and cited chat.";
  }, [pathname]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((current) => !current);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <div className="app-bg-grid min-h-screen">
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        sessions={sessionsQuery.data?.sessions ?? []}
        documents={documentsQuery.data?.documents ?? []}
      />
      <div className="mx-auto flex min-h-screen max-w-[1680px] gap-5 p-4 md:p-6">
        <div className="panel fixed inset-x-4 top-4 z-20 flex items-center justify-between px-4 py-3 lg:hidden">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-500 text-slate-950">
              <Bot className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-200/80">
                rag-smart-qa
              </p>
              <p className="text-sm font-medium text-white">AI workspace</p>
            </div>
          </div>
          <div className="flex gap-2 overflow-x-auto">
            {nav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded-full px-3 py-2 text-xs font-medium",
                  pathname === item.href
                    ? "bg-white text-slate-950"
                    : "bg-white/5 text-slate-300"
                )}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
        <aside className="panel panel-glow hidden w-[320px] shrink-0 flex-col justify-between overflow-hidden p-6 lg:flex">
          <div className="space-y-7">
            <div className="premium-ring rounded-[28px] border border-white/8 bg-gradient-to-br from-cyan-400/20 via-white/6 to-blue-500/20 p-5">
              <div className="flex items-center gap-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-3xl bg-gradient-to-br from-cyan-300 to-blue-500 text-slate-950 shadow-[0_20px_40px_rgba(56,189,248,0.25)]">
                  <Bot className="h-6 w-6" />
                </div>
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-cyan-100/75">
                    rag-smart-qa
                  </p>
                  <h1 className="mt-1 text-2xl font-semibold tracking-tight text-white">
                    Retrieval workspace
                  </h1>
                </div>
              </div>
              <p className="mt-4 text-sm leading-7 text-slate-300">
                Premium document-grounded chat, knowledge base controls, and explainable AI answers.
              </p>
              <div className="mt-4 flex items-center gap-2">
                <StatusBadge label="api connected" tone="ready" />
                <StatusBadge
                  label={`${documentsQuery.data?.documents.length ?? 0} docs`}
                  tone="queued"
                />
              </div>
            </div>
            <nav className="space-y-2">
              {nav.map((item) => {
                const Icon = item.icon;
                const active = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "group flex items-center gap-3 rounded-[22px] px-4 py-3.5 text-sm transition",
                      active
                        ? "bg-white text-slate-950 shadow-[0_18px_40px_rgba(255,255,255,0.14)]"
                        : "text-slate-400 hover:bg-white/6 hover:text-white"
                    )}
                  >
                    <div
                      className={cn(
                        "flex h-10 w-10 items-center justify-center rounded-2xl border transition",
                        active
                          ? "border-slate-200 bg-slate-950 text-white"
                          : "border-white/10 bg-white/5 text-slate-300 group-hover:border-cyan-400/25 group-hover:text-cyan-100"
                      )}
                    >
                      <Icon className="h-4 w-4" />
                    </div>
                    <div>
                      <span className="block font-medium">{item.label}</span>
                      <span
                        className={cn(
                          "mt-0.5 block text-xs",
                          active ? "text-slate-500" : "text-slate-500 group-hover:text-slate-300"
                        )}
                      >
                        {item.href === "/chat"
                          ? "AI conversations"
                          : item.href === "/knowledge-base"
                            ? "Document operations"
                            : item.href === "/summaries"
                              ? "Briefing center"
                              : item.href === "/settings"
                                ? "Workspace config"
                                : "Mission control"}
                      </span>
                    </div>
                  </Link>
                );
              })}
            </nav>
            <div className="panel-muted p-4">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Recent Chats
                </p>
                <button
                  onClick={() => setPaletteOpen(true)}
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] uppercase tracking-[0.16em] text-slate-400 transition hover:border-cyan-400/30 hover:text-cyan-100"
                >
                  <Command className="h-3 w-3" />
                  search
                </button>
              </div>
              <div className="mt-4 space-y-2">
                {sessionsQuery.data?.sessions.slice(0, 5).map((session) => (
                  <Link
                    key={session.id}
                    href={`/chat?session=${session.id}`}
                    className="block rounded-[20px] border border-transparent bg-white/3 px-4 py-3 text-sm text-slate-300 transition hover:border-cyan-400/20 hover:bg-white/8 hover:text-white"
                  >
                    <p className="font-medium">{session.title}</p>
                    <p className="mt-1 text-xs text-slate-500">
                      Updated {new Date(session.updated_at).toLocaleDateString()}
                    </p>
                  </Link>
                ))}
                {!sessionsQuery.data?.sessions.length ? (
                  <div className="rounded-[22px] border border-dashed border-white/10 bg-white/3 px-4 py-6 text-center">
                    <p className="text-sm text-slate-300">No recent chats yet.</p>
                    <p className="mt-1 text-xs text-slate-500">
                      Start a conversation and it will appear here.
                    </p>
                  </div>
                ) : null}
              </div>
            </div>
            <div className="panel-muted p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                Workspace Pulse
              </p>
              <div className="mt-4 space-y-3 text-sm">
                <div className="flex items-center justify-between text-slate-300">
                  <span>Knowledge base</span>
                  <StatusBadge
                    label={documentsQuery.data?.documents.length ? "active" : "empty"}
                    tone={documentsQuery.data?.documents.length ? "ready" : "default"}
                  />
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span>Recent sessions</span>
                  <span className="text-slate-500">{sessionsQuery.data?.sessions.length ?? 0}</span>
                </div>
                <div className="flex items-center justify-between text-slate-300">
                  <span>Premium mode</span>
                  <span className="inline-flex items-center gap-2 text-cyan-100">
                    <Zap className="h-3.5 w-3.5" />
                    enabled
                  </span>
                </div>
              </div>
            </div>
          </div>
          <div className="premium-ring rounded-[28px] border border-white/8 bg-gradient-to-br from-white/8 via-white/4 to-cyan-400/10 px-5 py-5">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-100/70">
              Product note
            </p>
            <p className="mt-3 text-sm leading-7 text-slate-300">
              Typed APIs, persisted chats, indexed citations, and cached summaries are all live in
              this workspace.
            </p>
          </div>
        </aside>
        <main className="flex-1 pt-20 lg:pt-0">
          <TopBar title={heading} subtitle={subtitle} onOpenPalette={() => setPaletteOpen(true)} />
          {children}
        </main>
      </div>
    </div>
  );
}

"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Bot, Database, FilePlus2, Layers3, MessageSquareText, ShieldCheck } from "lucide-react";
import Link from "next/link";

import { EmptyState } from "@/components/empty-state";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/page-header";
import { StatCard } from "@/components/stat-card";
import { StatusBadge } from "@/components/status-badge";
import { UploadDropzone } from "@/features/upload/upload-dropzone";

export default function DashboardPage() {
  const dashboardQuery = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard });
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const stats = dashboardQuery.data?.stats;
  const totalDocs = stats?.total_documents ?? 0;
  const totalChats = stats?.total_sessions ?? 0;
  const totalChunks = stats?.total_chunks ?? 0;
  const readyDocs = stats?.indexing_status.ready ?? 0;
  const summaryEnabled = settingsQuery.data?.summaries_enabled ?? false;
  const retrievalMode = settingsQuery.data?.default_retrieval_mode ?? "hybrid_rrf";

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Workspace Command"
        title="Your premium AI knowledge workspace"
        description="Upload source material, monitor indexing health, and move straight into grounded chat with a UI designed to feel investor-demo ready from the first screenshot."
        kicker={
          <div className="flex flex-wrap gap-2">
            <StatusBadge label={`${readyDocs}/${totalDocs} indexed`} tone={readyDocs ? "ready" : "queued"} />
            <StatusBadge label={retrievalMode} tone="queued" />
            <StatusBadge label={summaryEnabled ? "summaries on" : "summaries off"} tone={summaryEnabled ? "ready" : "disabled"} />
          </div>
        }
        actions={
          <div className="flex flex-wrap gap-3">
            <Link
              href="/knowledge-base"
              className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
            >
              Upload documents
            </Link>
            <Link
              href="/chat"
              className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition hover:border-cyan-400/30 hover:bg-white/10"
            >
              Open chat
            </Link>
          </div>
        }
      />

      <div className="grid gap-4 xl:grid-cols-12">
        <StatCard
          label="Documents"
          value={totalDocs}
          subtext="Knowledge sources ready for retrieval and citation grounding."
          icon={<Database className="h-5 w-5" />}
          trend={totalDocs ? "Corpus is active" : "Ready for first upload"}
        />
        <StatCard
          label="Chunks"
          value={totalChunks}
          subtext="Chunked passages powering hybrid search, reranking, and source attribution."
          icon={<Layers3 className="h-5 w-5" />}
          trend={totalChunks ? "Retrieval context loaded" : "No chunks indexed yet"}
        />
        <StatCard
          label="Chats"
          value={totalChats}
          subtext="Persisted sessions with citation-aware answers and workspace memory."
          icon={<MessageSquareText className="h-5 w-5" />}
          trend={totalChats ? "Conversation history available" : "No active sessions"}
        />
        <div className="panel panel-hover premium-ring xl:col-span-3 p-5">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                Index Health
              </p>
              <p className="mt-4 text-4xl font-semibold tracking-tight text-white">
                {totalDocs ? `${Math.round((readyDocs / totalDocs) * 100)}%` : "0%"}
              </p>
            </div>
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
              <ShieldCheck className="h-5 w-5" />
            </div>
          </div>
          <p className="mt-4 text-sm leading-6 text-slate-400">
            Tracks how much of the workspace is ingestion-complete and ready for trustworthy answers.
          </p>
          <div className="mt-4 h-2 rounded-full bg-white/6">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500"
              style={{ width: `${totalDocs ? Math.max(8, (readyDocs / totalDocs) * 100) : 8}%` }}
            />
          </div>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="xl:col-span-8 space-y-5">
          <div className="panel panel-glow p-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">
                  Mission control
                </p>
                <h3 className="mt-2 text-2xl font-semibold tracking-tight text-white">
                  System status and recent activity
                </h3>
                <p className="mt-2 max-w-2xl text-sm leading-7 text-slate-400">
                  Keep an eye on ingestion readiness, retrieval defaults, and the latest documents entering the knowledge base.
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <StatusBadge label={summaryEnabled ? "summary cache enabled" : "summary cache off"} tone={summaryEnabled ? "ready" : "disabled"} />
                <StatusBadge label={`retrieval: ${retrievalMode}`} tone="queued" />
              </div>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <div className="rounded-[26px] border border-white/10 bg-white/5 p-5">
                <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Ready documents
                </p>
                <p className="mt-4 text-3xl font-semibold text-white">{readyDocs}</p>
                <p className="mt-2 text-sm text-slate-400">Documents with completed indexing state.</p>
              </div>
              <div className="rounded-[26px] border border-white/10 bg-white/5 p-5">
                <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Summary cache
                </p>
                <p className="mt-4 text-3xl font-semibold text-white">
                  {summaryEnabled ? "Active" : "Off"}
                </p>
                <p className="mt-2 text-sm text-slate-400">Generated briefings ready to load instantly.</p>
              </div>
              <div className="rounded-[26px] border border-white/10 bg-white/5 p-5">
                <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Retrieval default
                </p>
                <p className="mt-4 text-3xl font-semibold text-white">{retrievalMode}</p>
                <p className="mt-2 text-sm text-slate-400">Current default mode exposed by the backend.</p>
              </div>
            </div>
          </div>

          <div className="panel p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">
                  Activity stream
                </p>
                <h3 className="mt-2 text-2xl font-semibold tracking-tight text-white">
                  Recent document activity
                </h3>
              </div>
              <Link
                href="/knowledge-base"
                className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-medium text-slate-200 transition hover:border-cyan-400/30 hover:text-white"
              >
                Open knowledge base
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>

            <div className="mt-5 space-y-3">
              {dashboardQuery.data?.recent_documents.length ? (
                dashboardQuery.data?.recent_documents.map((document) => (
                  <div
                    key={document.id}
                    className="panel-muted flex flex-wrap items-center justify-between gap-4 p-4"
                  >
                    <div>
                      <p className="text-base font-semibold text-white">{document.filename}</p>
                      <p className="mt-1 text-sm text-slate-400">
                        {document.chunks_created} chunks • {document.pages} pages • {document.file_type.toUpperCase()}
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      <StatusBadge label={document.indexing_status} tone={document.indexing_status} />
                      <span className="text-sm text-slate-500">
                        {new Date(document.upload_time).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <EmptyState
                  icon={<FilePlus2 className="h-7 w-7" />}
                  title="Start by ingesting your first source"
                  description="The dashboard stays polished even at zero documents. Upload PDFs, markdown notes, HTML, or plain text and this activity stream will immediately become your workspace timeline."
                  action={
                    <Link
                      href="/knowledge-base"
                      className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
                    >
                      Go to knowledge base
                    </Link>
                  }
                  secondary="Upload documents to activate metrics, summaries, and grounded chat"
                />
              )}
            </div>
          </div>
        </div>

        <div className="xl:col-span-4 space-y-5">
          <UploadDropzone />

          <div className="panel p-6">
            <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">
              Quick actions
            </p>
            <div className="mt-4 space-y-3">
              {[
                {
                  href: "/chat",
                  label: "Launch AI chat",
                  description: "Ask follow-up questions with source citations.",
                  icon: Bot
                },
                {
                  href: "/summaries",
                  label: "Review summaries",
                  description: "Browse briefings and key insights before chatting.",
                  icon: Layers3
                },
                {
                  href: "/settings",
                  label: "Inspect settings",
                  description: "Review models, feature flags, and environment details.",
                  icon: ShieldCheck
                }
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="panel-muted flex items-center gap-4 p-4 transition hover:bg-white/8"
                  >
                    <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
                      <Icon className="h-4 w-4" />
                    </div>
                    <div>
                      <p className="font-semibold text-white">{item.label}</p>
                      <p className="mt-1 text-sm text-slate-400">{item.description}</p>
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>

          <div className="panel p-6">
            <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">
              Recent chats
            </p>
            <div className="mt-4 space-y-3">
              {dashboardQuery.data?.recent_sessions.length ? (
                dashboardQuery.data.recent_sessions.map((session) => (
                  <Link
                    key={session.id}
                    href={`/chat?session=${session.id}`}
                    className="panel-muted block p-4 transition hover:bg-white/8"
                  >
                    <p className="font-semibold text-white">{session.title}</p>
                    <p className="mt-1 text-sm text-slate-400">
                      Last updated {new Date(session.updated_at).toLocaleString()}
                    </p>
                  </Link>
                ))
              ) : (
                <div className="rounded-[24px] border border-dashed border-white/10 bg-white/3 p-5">
                  <p className="text-sm font-semibold text-white">No conversations yet</p>
                  <p className="mt-2 text-sm leading-6 text-slate-400">
                    Your chat timeline will appear here once you begin asking grounded questions.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="panel panel-glow flex flex-wrap items-center justify-between gap-4 px-6 py-5">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">
            Health strip
          </p>
          <p className="mt-2 text-sm text-slate-300">
            {totalDocs
              ? "Your workspace is populated and ready for grounded retrieval."
              : "The system is live and waiting for your first source upload."}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <StatusBadge label="api online" tone="ready" />
          <StatusBadge label={`docs ${totalDocs}`} tone={totalDocs ? "ready" : "default"} />
          <StatusBadge label={`chunks ${totalChunks}`} tone={totalChunks ? "queued" : "default"} />
        </div>
      </div>
    </div>
  );
}

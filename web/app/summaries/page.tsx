"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, BookOpenText, Sparkles } from "lucide-react";

import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { api } from "@/lib/api";

export default function SummariesPage() {
  const documentsQuery = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const summaryDocs = (documentsQuery.data?.documents ?? []).filter(
    (document) => document.summary_status !== "disabled"
  );

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Summaries"
        title="Cached document briefings with product-grade polish"
        description="Skim structured summaries, key insights, and document status before diving into long-form sources or citation-backed chat."
        kicker={<StatusBadge label={`${summaryDocs.length} briefing${summaryDocs.length === 1 ? "" : "s"}`} tone="ready" />}
        actions={
          <Link
            href="/knowledge-base"
            className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
          >
            Upload more sources
          </Link>
        }
      />
      {summaryDocs.length ? (
        <div className="grid gap-4 xl:grid-cols-2">
          {summaryDocs.map((document) => (
            <Link
              key={document.id}
              href={`/documents/${document.id}`}
              className="panel panel-hover premium-ring overflow-hidden p-6"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-4">
                  <div className="flex h-14 w-14 items-center justify-center rounded-3xl border border-white/10 bg-white/6 text-cyan-200">
                    <BookOpenText className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                      {document.file_type.toUpperCase()}
                    </p>
                    <h3 className="mt-1 text-xl font-semibold tracking-tight text-white">
                      {document.filename}
                    </h3>
                  </div>
                </div>
                <ArrowUpRight className="h-4 w-4 text-slate-500" />
              </div>
              <div className="mt-5 flex flex-wrap gap-2">
                <StatusBadge label={`summary ${document.summary_status}`} tone={document.summary_status} />
                <StatusBadge label={`index ${document.indexing_status}`} tone={document.indexing_status} subtle />
              </div>
              <div className="mt-5 rounded-[24px] border border-white/10 bg-black/20 p-4">
                <p className="text-sm leading-7 text-slate-300">
                  Cached briefing ready for fast skimming, insight review, and detail navigation.
                </p>
              </div>
              <div className="mt-5 flex items-center justify-between text-sm text-slate-400">
                <span>{document.chunks_created} chunks</span>
                <span>{new Date(document.upload_time).toLocaleDateString()}</span>
              </div>
            </Link>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<Sparkles className="h-7 w-7" />}
          title="No summaries yet, but the experience is ready"
          description="Once documents are ingested, this page becomes a polished briefing hub with insight cards, metadata signals, and quick navigation back to the original sources."
          action={
            <Link
              href="/knowledge-base"
              className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
            >
              Add documents
            </Link>
          }
        />
      )}
    </div>
  );
}

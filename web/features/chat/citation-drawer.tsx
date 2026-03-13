"use client";

import { FileStack, Quote, Sparkles } from "lucide-react";
import Link from "next/link";

import type { Citation } from "@/types/api";
import { StatusBadge } from "@/components/status-badge";

export function CitationDrawer({ citation }: { citation?: Citation | null }) {
  return (
    <aside className="panel panel-glow min-h-[320px] p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200/70">
        Source Focus
      </p>
      {citation ? (
        <div className="mt-4 space-y-4">
          <div className="rounded-[26px] border border-white/10 bg-white/5 p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
                  <FileStack className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-sm font-medium text-white">{citation.source}</p>
                  <p className="text-sm text-slate-400">
                    Page {citation.page} • Chunk {citation.chunk_id}
                  </p>
                </div>
              </div>
              <StatusBadge label={`${Math.round(citation.score * 100)}%`} tone="ready" subtle />
            </div>
          </div>
          <div className="rounded-[26px] border border-white/10 bg-black/20 p-5">
            <div className="flex items-center gap-2 text-cyan-200">
              <Quote className="h-4 w-4" />
              <span className="text-xs font-semibold uppercase tracking-[0.18em]">Excerpt</span>
            </div>
            <p className="mt-4 text-sm leading-7 text-slate-200">
              {citation.excerpt}
            </p>
          </div>
          {citation.document_id ? (
            <Link
              href={`/documents/${citation.document_id}`}
              className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
            >
              Open document detail
              <Sparkles className="h-4 w-4" />
            </Link>
          ) : null}
        </div>
      ) : (
        <div className="mt-5 rounded-[28px] border border-dashed border-white/10 bg-white/3 p-6">
          <div className="flex h-14 w-14 items-center justify-center rounded-3xl border border-white/10 bg-white/6 text-cyan-200">
            <Sparkles className="h-5 w-5" />
          </div>
          <p className="mt-5 text-lg font-semibold text-white">Inspect cited evidence here</p>
          <p className="mt-2 text-sm leading-7 text-slate-400">
            Click any citation from an assistant response to inspect the exact excerpt, source file, and page context in this drawer.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <StatusBadge label="exact chunk id" tone="queued" subtle />
            <StatusBadge label="page reference" tone="queued" subtle />
            <StatusBadge label="source jump" tone="queued" subtle />
          </div>
        </div>
      )}
    </aside>
  );
}

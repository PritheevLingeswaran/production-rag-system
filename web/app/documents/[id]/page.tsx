"use client";

import { useQuery } from "@tanstack/react-query";
import { FileStack, Layers3, ScrollText } from "lucide-react";
import { useParams } from "next/navigation";

import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { api } from "@/lib/api";
import { SummaryCard } from "@/features/summaries/summary-card";

export default function DocumentDetailPage() {
  const params = useParams<{ id: string }>();
  const documentQuery = useQuery({
    queryKey: ["document", params.id],
    queryFn: () => api.document(params.id)
  });

  const document = documentQuery.data;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Document"
        title={document?.filename ?? "Document detail"}
        description="Preview pages, inspect chunk boundaries, and review the cached AI summary in a more polished source intelligence workspace."
        kicker={
          document ? (
            <div className="flex flex-wrap gap-2">
              <StatusBadge label={document.indexing_status} tone={document.indexing_status} />
              <StatusBadge label={`summary ${document.summary_status}`} tone={document.summary_status} />
            </div>
          ) : undefined
        }
      />
      <SummaryCard summary={document?.summary ?? null} />
      <div className="grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
        <div className="panel p-5">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
              <ScrollText className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Source preview</h3>
              <p className="text-sm text-slate-400">Readable excerpts from the original file.</p>
            </div>
          </div>
          <div className="mt-4 space-y-4">
            {document?.preview.length ? (
              document?.preview.map((page) => (
                <div key={page.page} className="panel-muted p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-200/70">
                    Page {page.page}
                  </p>
                  <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-300">
                    {page.text}
                  </p>
                </div>
              ))
            ) : (
              <EmptyState
                icon={<FileStack className="h-7 w-7" />}
                title="No preview extracted"
                description="This document does not currently expose a preview payload. Once available, the page content will render here."
              />
            )}
          </div>
        </div>
        <div className="panel p-5">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
              <Layers3 className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Indexed chunks</h3>
              <p className="text-sm text-slate-400">The units used for retrieval and citation linking.</p>
            </div>
          </div>
          <div className="mt-4 space-y-3">
            {document?.chunks.length ? (
              document?.chunks.map((chunk) => (
                <div key={chunk.chunk_id} className="panel-muted p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-200/70">
                    {chunk.chunk_id} • page {chunk.page}
                  </p>
                  <p className="mt-2 text-sm leading-7 text-slate-300">{chunk.text}</p>
                </div>
              ))
            ) : (
              <EmptyState
                icon={<Layers3 className="h-7 w-7" />}
                title="No chunks available"
                description="Chunk metadata will appear here after indexing completes for this document."
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

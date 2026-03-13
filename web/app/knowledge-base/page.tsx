"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { ArrowRight, LibraryBig, Search } from "lucide-react";

import { PageHeader } from "@/components/page-header";
import { useToast } from "@/components/toast-provider";
import { DocumentsTable } from "@/features/documents/documents-table";
import { UploadDropzone } from "@/features/upload/upload-dropzone";
import { api } from "@/lib/api";

export default function KnowledgeBasePage() {
  const queryClient = useQueryClient();
  const { pushToast } = useToast();
  const documentsQuery = useQuery({ queryKey: ["documents"], queryFn: api.documents });

  const deleteMutation = useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: () => {
      pushToast({ tone: "success", title: "Document removed", description: "Corpus rebuilt in the background." });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    }
  });
  const reindexMutation = useMutation({
    mutationFn: api.reindexDocument,
    onSuccess: () => {
      pushToast({ tone: "info", title: "Reindex triggered", description: "The document is being reprocessed now." });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    }
  });

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Knowledge Base"
        title="Curate the corpus behind every answer"
        description="Blend upload workflows, search, metadata, and background indexing into a single polished knowledge management experience."
        kicker={
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-slate-300">
              {documentsQuery.data?.documents.length ?? 0} documents
            </span>
          </div>
        }
        actions={
          <div className="flex flex-wrap gap-3">
            <Link
              href="/chat"
              className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
            >
              Ask the corpus
            </Link>
            <Link
              href="/summaries"
              className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition hover:border-cyan-400/30 hover:bg-white/10"
            >
              View summaries
            </Link>
          </div>
        }
      />
      <div className="grid gap-5 xl:grid-cols-[0.88fr_1.12fr]">
        <UploadDropzone onUploaded={() => void queryClient.invalidateQueries({ queryKey: ["documents"] })} />
        <DocumentsTable
          documents={documentsQuery.data?.documents ?? []}
          onDelete={(id) => deleteMutation.mutate(id)}
          onReindex={(id) => reindexMutation.mutate(id)}
        />
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        {[
          {
            icon: LibraryBig,
            title: "Status-driven metadata",
            body: "Every document card highlights indexing state, chunk count, pages, and summary readiness."
          },
          {
            icon: Search,
            title: "Search-first browse experience",
            body: "Filter and locate files quickly before opening detail pages or triggering reindex operations."
          },
          {
            icon: ArrowRight,
            title: "Fast path to chat",
            body: "Move from upload to citation-backed Q&A without leaving the workspace design system."
          }
        ].map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.title} className="panel-muted p-5">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
                <Icon className="h-4 w-4" />
              </div>
              <p className="mt-4 text-lg font-semibold text-white">{item.title}</p>
              <p className="mt-2 text-sm leading-7 text-slate-400">{item.body}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

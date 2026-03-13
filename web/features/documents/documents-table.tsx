"use client";

import { Search } from "lucide-react";
import { useMemo, useState } from "react";

import { DocumentCard } from "@/components/document-card";
import { EmptyState } from "@/components/empty-state";
import type { DocumentItem } from "@/types/api";

export function DocumentsTable({
  documents,
  onDelete,
  onReindex
}: {
  documents: DocumentItem[];
  onDelete?: (id: string) => void;
  onReindex?: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  const filteredDocuments = useMemo(() => {
    return documents.filter((document) => {
      const matchesQuery = query
        ? `${document.filename} ${document.file_type} ${document.indexing_status}`
            .toLowerCase()
            .includes(query.toLowerCase())
        : true;
      const matchesStatus = statusFilter === "all" ? true : document.indexing_status === statusFilter;
      return matchesQuery && matchesStatus;
    });
  }, [documents, query, statusFilter]);

  if (!documents.length) {
    return (
      <EmptyState
        icon={<Search className="h-7 w-7" />}
        title="Your knowledge base is ready for its first source"
        description="Upload documents to unlock chunking, summaries, retrieval, and fully cited answers. This space will evolve into your searchable corpus view."
        secondary="Empty states stay visually rich so the app still feels complete before first use"
      />
    );
  }

  return (
    <div className="panel panel-glow p-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">
            Corpus explorer
          </p>
          <h3 className="mt-2 text-2xl font-semibold tracking-tight text-white">
            Searchable knowledge base
          </h3>
          <p className="mt-2 text-sm leading-7 text-slate-400">
            Review documents as polished cards with metadata, indexing state, and quick actions.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
            <Search className="h-4 w-4 text-slate-400" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search documents"
              className="bg-transparent text-sm text-white outline-none placeholder:text-slate-500"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value)}
            className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none"
          >
            <option value="all">All statuses</option>
            <option value="ready">Ready</option>
            <option value="processing">Processing</option>
            <option value="queued">Queued</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      <div className="mt-6 grid gap-4 xl:grid-cols-2">
        {filteredDocuments.map((document) => (
          <DocumentCard
            key={document.id}
            document={document}
            onDelete={onDelete}
            onReindex={onReindex}
          />
        ))}
      </div>

      {!filteredDocuments.length ? (
        <div className="mt-6 rounded-[28px] border border-dashed border-white/10 bg-white/4 px-6 py-10 text-center">
          <p className="text-lg font-semibold text-white">No documents match the current filter</p>
          <p className="mt-2 text-sm text-slate-400">
            Try a different search term or switch back to all statuses.
          </p>
        </div>
      ) : null}
    </div>
  );
}

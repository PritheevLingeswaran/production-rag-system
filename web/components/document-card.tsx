import { CalendarDays, FileText, Layers3 } from "lucide-react";
import Link from "next/link";

import { StatusBadge } from "@/components/status-badge";
import type { DocumentItem } from "@/types/api";

export function DocumentCard({
  document,
  onDelete,
  onReindex
}: {
  document: DocumentItem;
  onDelete?: (id: string) => void;
  onReindex?: (id: string) => void;
}) {
  return (
    <div className="panel panel-hover p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-3xl border border-white/10 bg-white/6 text-cyan-200">
            <FileText className="h-5 w-5" />
          </div>
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
              {document.file_type}
            </p>
            <Link
              href={`/documents/${document.id}`}
              className="mt-1 block text-lg font-semibold tracking-tight text-white transition hover:text-cyan-200"
            >
              {document.filename}
            </Link>
          </div>
        </div>
        <StatusBadge label={document.indexing_status} tone={document.indexing_status} />
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
          <p className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-slate-500">
            <Layers3 className="h-3.5 w-3.5" />
            Chunks
          </p>
          <p className="mt-2 text-xl font-semibold text-white">{document.chunks_created}</p>
        </div>
        <div className="rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
          <p className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-slate-500">
            <CalendarDays className="h-3.5 w-3.5" />
            Uploaded
          </p>
          <p className="mt-2 text-sm font-medium text-slate-200">
            {new Date(document.upload_time).toLocaleDateString()}
          </p>
        </div>
      </div>

      <div className="mt-5 flex items-center justify-between text-sm text-slate-400">
        <span>{document.pages} pages indexed</span>
        <span>Summary {document.summary_status}</span>
      </div>

      <div className="mt-5 flex gap-2">
        <button
          onClick={() => onReindex?.(document.id)}
          className="rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:border-cyan-400/30 hover:bg-white/10 hover:text-white"
        >
          Reindex
        </button>
        <button
          onClick={() => onDelete?.(document.id)}
          className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-2.5 text-sm font-medium text-rose-100 transition hover:bg-rose-500/20"
        >
          Remove
        </button>
      </div>
    </div>
  );
}

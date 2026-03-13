import { BrainCircuit, ListChecks, Tag } from "lucide-react";

import { EmptyState } from "@/components/empty-state";
import { StatusBadge } from "@/components/status-badge";
import type { SummaryPayload } from "@/types/api";

export function SummaryCard({ summary }: { summary?: SummaryPayload | null }) {
  if (!summary) {
    return (
      <EmptyState
        icon={<BrainCircuit className="h-7 w-7" />}
        title="Summary cache not available yet"
        description="Once this document is summarized, key insights, important points, and extractive metadata will appear here in a polished briefing layout."
      />
    );
  }

  return (
    <div className="panel panel-glow space-y-5 p-6">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge label={summary.status} tone={summary.status} />
          {summary.method ? <StatusBadge label={summary.method} tone="queued" subtle /> : null}
        </div>
        <p className="mt-4 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200/70">Summary</p>
        <h3 className="mt-2 text-3xl font-semibold tracking-tight text-white">
          {summary.title ?? "Document summary"}
        </h3>
        <p className="mt-4 text-sm leading-8 text-slate-300">{summary.summary}</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="panel-muted p-4">
          <div className="flex items-center gap-2 text-cyan-200">
            <BrainCircuit className="h-4 w-4" />
            <p className="text-sm font-medium text-white">Key insights</p>
          </div>
          <ul className="mt-3 space-y-2 text-sm text-slate-300">
            {summary.key_insights.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="panel-muted p-4">
          <div className="flex items-center gap-2 text-cyan-200">
            <ListChecks className="h-4 w-4" />
            <p className="text-sm font-medium text-white">Important points</p>
          </div>
          <ul className="mt-3 space-y-2 text-sm text-slate-300">
            {summary.important_points.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
      {!!summary.keywords.length && (
        <div className="panel-muted p-4">
          <div className="flex items-center gap-2 text-cyan-200">
            <Tag className="h-4 w-4" />
            <p className="text-sm font-medium text-white">Keywords</p>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {summary.keywords.map((item) => (
              <span
                key={item}
                className="rounded-full border border-white/10 bg-white/6 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-slate-200"
              >
                {item}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

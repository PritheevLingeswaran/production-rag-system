"use client";

import { cn } from "@/lib/utils";

const tones: Record<string, string> = {
  ready: "bg-emerald-500/15 text-emerald-200 ring-1 ring-inset ring-emerald-500/30",
  processing: "bg-amber-400/15 text-amber-100 ring-1 ring-inset ring-amber-400/30",
  queued: "bg-sky-400/15 text-sky-100 ring-1 ring-inset ring-sky-400/30",
  failed: "bg-rose-500/15 text-rose-100 ring-1 ring-inset ring-rose-500/30",
  success: "bg-emerald-500/15 text-emerald-200 ring-1 ring-inset ring-emerald-500/30",
  error: "bg-rose-500/15 text-rose-100 ring-1 ring-inset ring-rose-500/30",
  disabled: "bg-white/10 text-slate-300 ring-1 ring-inset ring-white/10",
  default: "bg-white/10 text-slate-200 ring-1 ring-inset ring-white/10"
};

export function StatusBadge({
  label,
  tone,
  subtle = false
}: {
  label: string;
  tone?: string;
  subtle?: boolean;
}) {
  const key = tone?.toLowerCase() ?? "default";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]",
        tones[key] ?? tones.default,
        subtle && "bg-black/5 text-slate-600 ring-slate-200 dark:bg-white/10 dark:text-slate-200"
      )}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label}
    </span>
  );
}

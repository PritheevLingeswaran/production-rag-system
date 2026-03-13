"use client";

import { Command, MoonStar, Search, SunMedium } from "lucide-react";

import { useTheme } from "@/components/theme-provider";

export function TopBar({
  title,
  subtitle,
  onOpenPalette
}: {
  title: string;
  subtitle: string;
  onOpenPalette: () => void;
}) {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="sticky top-0 z-20 mb-6 flex items-center justify-between rounded-[30px] border border-white/10 bg-slate-950/65 px-5 py-4 backdrop-blur-xl">
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-cyan-200/70">
          rag-smart-qa workspace
        </p>
        <h1 className="mt-2 text-xl font-semibold tracking-tight text-white">{title}</h1>
        <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenPalette}
          className="hidden items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300 transition hover:border-cyan-400/30 hover:bg-white/10 md:flex"
        >
          <Search className="h-4 w-4" />
          Search workspace
          <span className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-[11px] uppercase tracking-[0.18em] text-slate-400">
            <Command className="h-3 w-3" />K
          </span>
        </button>
        <button
          onClick={toggleTheme}
          className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-slate-300 transition hover:border-cyan-400/30 hover:bg-white/10"
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <SunMedium className="h-4 w-4" /> : <MoonStar className="h-4 w-4" />}
        </button>
      </div>
    </div>
  );
}

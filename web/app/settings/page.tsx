"use client";

import { useQuery } from "@tanstack/react-query";
import { Binary, Bot, LockKeyhole, ServerCog, Sparkles } from "lucide-react";

import { EmptyState } from "@/components/empty-state";
import { LoadingSkeleton } from "@/components/loading-skeleton";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { api } from "@/lib/api";

export default function SettingsPage() {
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const settings = settingsQuery.data;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Settings"
        title="Runtime configuration with product-level clarity"
        description="Model defaults, retrieval strategy, auth posture, and environment details are grouped into clean operational panels instead of a plain settings dump."
        kicker={settings ? <StatusBadge label={settings.environment} tone="ready" /> : undefined}
      />
      {settingsQuery.isLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="panel p-5">
              <LoadingSkeleton className="h-4 w-28" />
              <LoadingSkeleton className="mt-4 h-10 w-48" />
              <LoadingSkeleton className="mt-4 h-4 w-full" />
            </div>
          ))}
        </div>
      ) : settings ? (
        <div className="grid gap-5 xl:grid-cols-2">
          {[
            {
              title: "Model stack",
              icon: Bot,
              items: [
                ["Generation model", settings.default_generation_model],
                ["Embedding model", settings.default_embedding_model]
              ]
            },
            {
              title: "Retrieval setup",
              icon: Sparkles,
              items: [
                ["Default retrieval", settings.default_retrieval_mode],
                ["Vector store", settings.vector_store_provider]
              ]
            },
            {
              title: "Security and auth",
              icon: LockKeyhole,
              items: [["Auth", settings.auth_enabled ? settings.auth_provider : "disabled"]]
            },
            {
              title: "Environment",
              icon: ServerCog,
              items: [
                ["Environment", settings.environment],
                ["Summaries", settings.summaries_enabled ? "enabled" : "disabled"]
              ]
            }
          ].map((group) => {
            const Icon = group.icon;
            return (
              <div key={group.title} className="panel panel-glow p-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-200/70">
                      Configuration group
                    </p>
                    <h3 className="mt-1 text-2xl font-semibold tracking-tight text-white">{group.title}</h3>
                  </div>
                </div>
                <div className="mt-5 space-y-3">
                  {group.items.map(([label, value]) => (
                    <div key={label} className="panel-muted flex items-center justify-between gap-4 p-4">
                      <p className="text-sm text-slate-400">{label}</p>
                      <p className="text-sm font-semibold text-white">{value}</p>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
          <div className="panel panel-glow xl:col-span-2 p-6">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
                <Binary className="h-4 w-4" />
              </div>
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-200/70">
                  Operator note
                </p>
                <h3 className="mt-1 text-2xl font-semibold tracking-tight text-white">
                  Read-only runtime view
                </h3>
              </div>
            </div>
            <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-400">
              These panels surface backend settings in a polished way for demos and handoffs. If you later expose editable controls, this layout already separates model, retrieval, auth, and environment concerns cleanly.
            </p>
          </div>
        </div>
      ) : (
        <EmptyState
          icon={<ServerCog className="h-7 w-7" />}
          title="Settings are not available right now"
          description="The app could not load runtime configuration from the backend. Once restored, this page will resume showing grouped environment and model details."
        />
      )}
    </div>
  );
}

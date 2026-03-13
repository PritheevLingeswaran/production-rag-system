"use client";

import { useRef, useState } from "react";
import { ArrowUpRight, FileStack, UploadCloud } from "lucide-react";
import Link from "next/link";

import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/components/toast-provider";
import { API_BASE_URL } from "@/lib/config";

type UploadState = {
  name: string;
  progress: number;
  status: "uploading" | "success" | "error";
  detail?: string;
};

export function UploadDropzone({ onUploaded }: { onUploaded?: () => void }) {
  const [items, setItems] = useState<UploadState[]>([]);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { pushToast } = useToast();

  async function handleFiles(fileList: FileList | null) {
    if (!fileList?.length) return;
    for (const file of Array.from(fileList)) {
      await uploadFile(file);
    }
    onUploaded?.();
  }

  function uploadFile(file: File) {
    return new Promise<void>((resolve) => {
      setItems((current) => [...current, { name: file.name, progress: 0, status: "uploading" }]);
      const formData = new FormData();
      formData.append("files", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE_URL}/api/upload`);
      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.round((event.loaded / event.total) * 100);
        setItems((current) =>
          current.map((item) => (item.name === file.name ? { ...item, progress: percent } : item))
        );
      };
      xhr.onload = () => {
        const success = xhr.status >= 200 && xhr.status < 300;
        pushToast({
          tone: success ? "success" : "error",
          title: success ? `${file.name} uploaded` : `${file.name} failed`,
          description: success ? "Ingestion and indexing started in the background." : xhr.responseText
        });
        setItems((current) =>
          current.map((item) =>
            item.name === file.name
              ? {
                  ...item,
                  progress: 100,
                  status: success ? "success" : "error",
                  detail: success ? "Indexed in background" : xhr.responseText
                }
              : item
          )
        );
        resolve();
      };
      xhr.onerror = () => {
        pushToast({
          tone: "error",
          title: `${file.name} failed`,
          description: "Network error"
        });
        setItems((current) =>
          current.map((item) =>
            item.name === file.name ? { ...item, status: "error", detail: "Network error" } : item
          )
        );
        resolve();
      };
      xhr.send(formData);
    });
  }

  return (
    <div className="space-y-4">
      <div className="premium-ring panel panel-glow overflow-hidden p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200/70">
              Ingestion Studio
            </p>
            <h3 className="mt-2 text-2xl font-semibold tracking-tight text-white">
              Add documents and start indexing instantly
            </h3>
            <p className="mt-3 max-w-xl text-sm leading-7 text-slate-400">
              Bring PDFs, markdown, HTML, and text files into the workspace. We will store them, index them, and prepare summary cache automatically.
            </p>
          </div>
          <StatusBadge label="multi-file enabled" tone="ready" />
        </div>

        <label className="mt-6 flex cursor-pointer flex-col items-center justify-center gap-4 rounded-[28px] border border-dashed border-cyan-400/25 bg-gradient-to-br from-cyan-500/10 via-white/4 to-blue-500/10 px-8 py-10 text-center transition hover:border-cyan-300/45 hover:bg-white/8">
          <span className="flex h-16 w-16 items-center justify-center rounded-[24px] border border-white/10 bg-white/8 text-cyan-200">
            <UploadCloud className="h-7 w-7" />
          </span>
          <div>
            <p className="text-xl font-semibold tracking-tight text-white">Drop files to ingest and index</p>
            <p className="mt-2 text-sm leading-7 text-slate-400">
              PDF, TXT, Markdown, and HTML up to 25MB. Uploads trigger ingestion and summary caching.
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-2">
            {["PDF", "TXT", "MD", "HTML"].map((item) => (
              <span
                key={item}
                className="rounded-full border border-white/10 bg-white/6 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300"
              >
                {item}
              </span>
            ))}
          </div>
          <button
            type="button"
            onClick={(event) => {
              event.preventDefault();
              inputRef.current?.click();
            }}
            className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-100"
          >
            Choose files
          </button>
        </label>
        <input
          ref={inputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(event) => void handleFiles(event.target.files)}
        />
        <div className="mt-5 grid gap-3 md:grid-cols-2">
          <Link
            href="/knowledge-base"
            className="panel-muted flex items-center gap-3 p-4 transition hover:bg-white/8"
          >
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
              <FileStack className="h-4 w-4" />
            </div>
            <div>
              <p className="font-semibold text-white">Manage knowledge base</p>
              <p className="text-sm text-slate-400">Review docs, status, and chunk counts.</p>
            </div>
            <ArrowUpRight className="ml-auto h-4 w-4 text-slate-500" />
          </Link>
          <div className="panel-muted p-4">
            <p className="text-sm font-semibold text-white">Background-safe ingestion</p>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Files upload immediately while indexing and summary generation continue without blocking the UI.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        {items.map((item) => (
          <div key={item.name} className="panel-muted p-4">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-white">{item.name}</span>
              <StatusBadge label={item.status} tone={item.status} />
            </div>
            <div className="mt-3 h-2 rounded-full bg-white/8">
              <div
                className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all"
                style={{ width: `${item.progress}%` }}
              />
            </div>
            {item.detail ? <p className="mt-2 text-xs text-slate-400">{item.detail}</p> : null}
          </div>
        ))}
      </div>
    </div>
  );
}

"use client";

import { CheckCircle2, Info, X, XCircle } from "lucide-react";
import { createContext, useContext, useMemo, useState } from "react";

import { cn } from "@/lib/utils";

type ToastTone = "success" | "error" | "info";

type Toast = {
  id: number;
  title: string;
  description?: string;
  tone: ToastTone;
};

const ToastContext = createContext<{
  pushToast: (toast: Omit<Toast, "id">) => void;
}>({
  pushToast: () => undefined
});

const toneStyles: Record<ToastTone, string> = {
  success: "border-emerald-400/25 bg-emerald-500/10 text-emerald-100",
  error: "border-rose-400/25 bg-rose-500/10 text-rose-100",
  info: "border-cyan-400/25 bg-cyan-500/10 text-cyan-100"
};

const toneIcons = {
  success: CheckCircle2,
  error: XCircle,
  info: Info
};

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const value = useMemo(
    () => ({
      pushToast: (toast: Omit<Toast, "id">) => {
        const id = Date.now() + Math.random();
        setToasts((current) => [...current, { ...toast, id }]);
        window.setTimeout(() => {
          setToasts((current) => current.filter((item) => item.id !== id));
        }, 3600);
      }
    }),
    []
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-5 right-5 z-50 flex w-[360px] max-w-[calc(100vw-2rem)] flex-col gap-3">
        {toasts.map((toast) => {
          const Icon = toneIcons[toast.tone];
          return (
            <div
              key={toast.id}
              className={cn(
                "pointer-events-auto rounded-3xl border px-4 py-4 shadow-2xl backdrop-blur-xl transition",
                toneStyles[toast.tone]
              )}
            >
              <div className="flex items-start gap-3">
                <Icon className="mt-0.5 h-4 w-4 shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold">{toast.title}</p>
                  {toast.description ? (
                    <p className="mt-1 text-sm text-current/75">{toast.description}</p>
                  ) : null}
                </div>
                <button
                  onClick={() =>
                    setToasts((current) => current.filter((item) => item.id !== toast.id))
                  }
                  className="rounded-full p-1 text-current/70 transition hover:bg-white/10 hover:text-current"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}

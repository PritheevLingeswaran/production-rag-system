import { ArrowRight } from "lucide-react";

export function EmptyState({
  icon,
  title,
  description,
  action,
  secondary
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  secondary?: string;
}) {
  return (
    <div className="panel panel-glow flex min-h-[280px] flex-col items-center justify-center px-8 py-12 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-3xl border border-white/10 bg-white/8 text-cyan-200 shadow-[0_20px_50px_rgba(34,211,238,0.12)]">
        {icon}
      </div>
      <h3 className="mt-6 text-2xl font-semibold tracking-tight text-white">{title}</h3>
      <p className="mt-3 max-w-xl text-sm leading-7 text-slate-300">{description}</p>
      {action ? <div className="mt-6">{action}</div> : null}
      {secondary ? (
        <p className="mt-4 inline-flex items-center gap-2 text-xs uppercase tracking-[0.22em] text-slate-500">
          {secondary}
          <ArrowRight className="h-3.5 w-3.5" />
        </p>
      ) : null}
    </div>
  );
}

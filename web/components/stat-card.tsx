export function StatCard({
  label,
  value,
  subtext,
  icon,
  trend
}: {
  label: string;
  value: string | number;
  subtext: string;
  icon?: React.ReactNode;
  trend?: string;
}) {
  return (
    <div className="panel panel-hover premium-ring overflow-hidden p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
            {label}
          </p>
          <p className="mt-4 text-4xl font-semibold tracking-tight text-white">{value}</p>
        </div>
        {icon ? (
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/6 text-cyan-200">
            {icon}
          </div>
        ) : null}
      </div>
      <p className="mt-4 text-sm leading-6 text-slate-400">{subtext}</p>
      {trend ? (
        <p className="mt-3 text-xs uppercase tracking-[0.22em] text-cyan-200/80">{trend}</p>
      ) : null}
    </div>
  );
}

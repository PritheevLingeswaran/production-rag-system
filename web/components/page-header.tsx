export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  kicker
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: React.ReactNode;
  kicker?: React.ReactNode;
}) {
  return (
    <div className="premium-ring panel panel-glow mb-6 overflow-hidden p-6 md:p-8">
      <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
        <div className="max-w-3xl">
          <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-cyan-200/80">
            {eyebrow}
          </p>
          <h2 className="mt-3 text-4xl font-semibold tracking-[-0.04em] text-white md:text-5xl">
            {title}
          </h2>
          <p className="mt-3 text-base leading-8 text-slate-300">{description}</p>
        </div>
        <div className="flex flex-col items-start gap-3 md:items-end">
          {kicker ? <div>{kicker}</div> : null}
          {actions}
        </div>
      </div>
    </div>
  );
}

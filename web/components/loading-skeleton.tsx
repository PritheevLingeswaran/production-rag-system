export function LoadingSkeleton({
  className = ""
}: {
  className?: string;
}) {
  return <div className={`animate-pulse rounded-2xl bg-white/8 ${className}`} />;
}

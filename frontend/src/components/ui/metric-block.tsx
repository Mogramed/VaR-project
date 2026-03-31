import { cn } from "@/lib/utils";

export function MetricBlock({
  label,
  value,
  hint,
  tone = "neutral",
  className,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "neutral" | "accent" | "success" | "warning" | "danger";
  className?: string;
}) {
  const valueColor =
    tone === "success"
      ? "text-[var(--color-green)]"
      : tone === "warning"
        ? "text-[var(--color-amber)]"
        : tone === "danger"
          ? "text-[var(--color-red)]"
          : tone === "accent"
            ? "text-[var(--color-accent)]"
            : "text-[var(--color-text)]";

  return (
    <div
      className={cn(
        "surface rounded-[var(--radius-lg)] px-3.5 py-3",
        className,
      )}
    >
      <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
        {label}
      </div>
      <div className={cn("mono mt-1 text-xl font-semibold tracking-tight", valueColor)}>
        {value}
      </div>
      {hint ? (
        <div className="mt-0.5 truncate text-[11px] text-[var(--color-text-muted)]">
          {hint}
        </div>
      ) : null}
    </div>
  );
}

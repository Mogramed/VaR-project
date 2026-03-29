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
  const accent =
    tone === "success"
      ? "var(--color-green)"
      : tone === "warning"
        ? "var(--color-amber)"
        : tone === "danger"
          ? "var(--color-red)"
          : tone === "accent"
            ? "var(--color-accent)"
            : "rgba(243,239,231,0.92)";
  const glow =
    tone === "success"
      ? "rgba(95,212,166,0.18)"
      : tone === "warning"
        ? "rgba(242,180,93,0.18)"
        : tone === "danger"
          ? "rgba(242,117,117,0.18)"
          : tone === "accent"
            ? "rgba(216,155,73,0.18)"
            : "rgba(255,255,255,0.06)";

  return (
    <div
      className={cn(
        "surface relative overflow-hidden rounded-[1.5rem] p-5 transition duration-300 motion-safe:hover:-translate-y-[2px] motion-safe:hover:border-[var(--color-border-strong)] motion-safe:hover:shadow-[0_26px_60px_rgba(0,0,0,0.34)]",
        className,
      )}
    >
      <div
        className="absolute inset-x-0 top-0 h-px"
        style={{ background: `linear-gradient(90deg, transparent, ${glow}, transparent)` }}
      />
      <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
        {label}
      </div>
      <div
        className="mt-4 text-[2.35rem] font-semibold tracking-[-0.06em]"
        style={{ color: accent }}
      >
        {value}
      </div>
      {hint ? (
        <div className="mt-2 text-sm text-[var(--color-text-soft)]">{hint}</div>
      ) : null}
    </div>
  );
}

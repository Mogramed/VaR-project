"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/**
 * Detects when `value` changes and briefly flashes the direction
 * (green flash = value went up, red flash = value went down).
 */
function useValueFlash(value: string) {
  const prevRef = useRef(value);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (prevRef.current === value) return;
    const prevNum = parseFloat(prevRef.current.replace(/[^0-9.\-]/g, ""));
    const nextNum = parseFloat(value.replace(/[^0-9.\-]/g, ""));
    prevRef.current = value;
    if (Number.isFinite(prevNum) && Number.isFinite(nextNum) && prevNum !== nextNum) {
      const direction = nextNum > prevNum ? "up" as const : "down" as const;
      // Schedule flash on via microtask to avoid synchronous setState in effect
      const showId = setTimeout(() => setFlash(direction), 0);
      const hideId = setTimeout(() => setFlash(null), 600);
      return () => {
        clearTimeout(showId);
        clearTimeout(hideId);
      };
    }
  }, [value]);

  return flash;
}

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
  const flash = useValueFlash(value);

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

  const flashBorder =
    flash === "up"
      ? "border-[var(--color-green)]/30"
      : flash === "down"
        ? "border-[var(--color-red)]/30"
        : "";

  return (
    <div
      className={cn(
        "surface rounded-[var(--radius-lg)] px-3.5 py-3 transition-[border-color] duration-300",
        flashBorder,
        className,
      )}
    >
      <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
        {label}
      </div>
      <div className={cn(
        "mono mt-1 text-xl font-semibold tabular-nums tracking-tight transition-colors duration-300",
        valueColor,
      )}>
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

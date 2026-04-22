"use client";

import { useMemo } from "react";

import { useDashboardPrefs } from "@/lib/dashboard-preferences-context";
import { MODEL_LABELS } from "@/lib/dashboard-preferences";
import { cn } from "@/lib/utils";

export function DashboardActiveFilters({
  showSymbol = true,
  showHorizon = true,
  showModel = true,
  className,
}: {
  showSymbol?: boolean;
  showHorizon?: boolean;
  showModel?: boolean;
  className?: string;
}) {
  const { prefs, hasSymbolFilter } = useDashboardPrefs();
  const chips = useMemo(() => {
    const next: Array<{ key: string; label: string }> = [];
    if (showSymbol && hasSymbolFilter) {
      next.push({ key: "symbol", label: `Symbol ${prefs.symbolFilter}` });
    }
    if (showHorizon) {
      next.push({ key: "horizon", label: `Horizon ${prefs.horizon}` });
    }
    if (showModel) {
      next.push({ key: "model", label: `Model ${MODEL_LABELS[prefs.model]}` });
    }
    return next;
  }, [hasSymbolFilter, prefs.horizon, prefs.model, prefs.symbolFilter, showHorizon, showModel, showSymbol]);

  if (chips.length === 0) {
    return null;
  }

  return (
    <div className={cn("flex flex-wrap gap-1.5", className)}>
      {chips.map((chip) => (
        <span
          key={chip.key}
          className="rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-0.5 text-[10px] text-[var(--color-text-muted)]"
        >
          {chip.label}
        </span>
      ))}
    </div>
  );
}

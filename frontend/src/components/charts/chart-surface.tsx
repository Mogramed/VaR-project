"use client";

import type { ReactNode } from "react";
import ReactECharts from "echarts-for-react";
import type { ChartMode } from "@/lib/chart-options";
import { cn } from "@/lib/utils";

function resolveHeight(mode: ChartMode, dataCount: number) {
  if (mode === "trace") {
    return dataCount > 120 ? 440 : 380;
  }
  if (mode === "dense") {
    return 360;
  }
  if (mode === "comparison") {
    return dataCount <= 4 ? 280 : 340;
  }
  if (mode === "sparse") {
    return 260;
  }
  return 320;
}

export function ChartSurface({
  option,
  height,
  mode = "standard",
  dataCount = 0,
  eyebrow,
  title,
  description,
  meta,
  toolbar,
  footer,
  insight,
  insightLayout = "auto",
  emptyState,
  surface = "panel",
  className,
}: {
  option: Record<string, unknown>;
  height?: number;
  mode?: ChartMode;
  dataCount?: number;
  eyebrow?: string;
  title?: string;
  description?: string;
  meta?: string;
  toolbar?: ReactNode;
  footer?: ReactNode;
  insight?: ReactNode;
  insightLayout?: "auto" | "side" | "stack";
  emptyState?: ReactNode;
  surface?: "panel" | "bare";
  className?: string;
}) {
  const resolvedHeight = height ?? resolveHeight(mode, dataCount);
  const sparseMode = mode === "sparse" || (mode === "comparison" && dataCount <= 4);
  const hasData = dataCount > 0;
  const insightGridClass =
    insightLayout === "stack"
      ? "grid-cols-1"
      : insightLayout === "side"
        ? "xl:grid-cols-[minmax(220px,360px)_minmax(0,1fr)]"
        : "min-[1400px]:grid-cols-[minmax(220px,360px)_minmax(0,1fr)]";

  return (
    <section
      data-chart-surface
      className={cn(
        "overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)]",
        className,
      )}
    >
      {/* Header */}
      {(eyebrow || title || toolbar || meta) ? (
        <header className="flex items-center justify-between gap-3 border-b border-[var(--color-border)] px-3.5 py-2.5">
          <div className="flex items-center gap-3 min-w-0">
            {title ? (
              <h3 className="truncate text-[13px] font-semibold text-[var(--color-text)]">
                {title}
              </h3>
            ) : null}
            {eyebrow ? (
              <span className="hidden shrink-0 text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] sm:inline">
                {eyebrow}
              </span>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {toolbar}
            {meta ? (
              <span className="mono text-[10px] text-[var(--color-text-muted)]">
                {meta}
              </span>
            ) : null}
          </div>
        </header>
      ) : null}

      {/* Chart body */}
      {!hasData && emptyState ? (
        <div className="px-3.5 py-4">{emptyState}</div>
      ) : sparseMode && insight ? (
        <div className={cn("grid gap-4 p-3.5", insightGridClass)}>
          <div className="mx-auto w-full max-w-[380px]">
            <ReactECharts
              option={option}
              style={{ height: `${resolvedHeight}px`, width: "100%" }}
              opts={{ renderer: "svg" }}
              notMerge
              lazyUpdate
            />
          </div>
          <div className="min-w-0">{insight}</div>
        </div>
      ) : (
        <div className="px-1 py-1">
          <ReactECharts
            option={option}
            style={{ height: `${resolvedHeight}px`, width: "100%" }}
            opts={{ renderer: "svg" }}
            notMerge
            lazyUpdate
          />
        </div>
      )}

      {/* Footer */}
      {footer ? (
        <div className="border-t border-[var(--color-border)] px-3.5 py-2.5">{footer}</div>
      ) : null}
    </section>
  );
}

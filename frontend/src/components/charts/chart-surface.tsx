"use client";

import type { ReactNode } from "react";
import ReactECharts from "echarts-for-react";
import type { ChartMode } from "@/lib/chart-options";
import { cn } from "@/lib/utils";

function resolveHeight(mode: ChartMode, dataCount: number) {
  if (mode === "trace") {
    return dataCount > 120 ? 460 : 400;
  }
  if (mode === "dense") {
    return 380;
  }
  if (mode === "comparison") {
    return dataCount <= 4 ? 300 : 360;
  }
  if (mode === "sparse") {
    return 280;
  }
  return 340;
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
  showDescription = false,
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
  showDescription?: boolean;
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
        "group/chart relative overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)]",
        className,
      )}
    >
      {/* Subtle top-edge glow */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[rgba(240,185,11,0.15)] to-transparent" />

      {/* Header */}
      {(eyebrow || title || toolbar || meta) ? (
        <header className="flex items-center justify-between gap-3 border-b border-[var(--color-border)] px-4 py-2.5">
          <div className="flex items-center gap-2.5 min-w-0">
            {eyebrow ? (
              <span className="shrink-0 rounded-[3px] bg-[rgba(240,185,11,0.08)] px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-[0.08em] text-[#f0b90b]">
                {eyebrow}
              </span>
            ) : null}
            {title ? (
              <h3 className="truncate text-[13px] font-semibold tracking-[-0.01em] text-[var(--color-text)]">
                {title}
              </h3>
            ) : null}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {toolbar}
            {meta ? (
              <span className="font-mono text-[10px] text-[var(--color-text-muted)]">
                {meta}
              </span>
            ) : null}
          </div>
        </header>
      ) : null}

      {/* Description */}
      {showDescription && description ? (
        <div className="border-b border-[var(--color-border)] px-4 pb-2.5 pt-1.5 text-[11px] leading-relaxed text-[var(--color-text-muted)]">
          {description}
        </div>
      ) : null}

      {/* Chart body */}
      {!hasData && emptyState ? (
        <div className="px-4 py-6">{emptyState}</div>
      ) : sparseMode && insight ? (
        <div className={cn("grid gap-4 p-4", insightGridClass)}>
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
        <div className="relative px-1 py-1">
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
        <div className="border-t border-[var(--color-border)] px-4 py-2.5">{footer}</div>
      ) : null}
    </section>
  );
}

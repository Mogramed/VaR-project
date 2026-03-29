"use client";

import type { ReactNode } from "react";
import ReactECharts from "echarts-for-react";
import type { ChartMode } from "@/lib/chart-options";
import { cn } from "@/lib/utils";

function resolveHeight(mode: ChartMode, dataCount: number) {
  if (mode === "trace") {
    return dataCount > 120 ? 500 : 460;
  }
  if (mode === "dense") {
    return 420;
  }
  if (mode === "comparison") {
    return dataCount <= 4 ? 320 : 380;
  }
  if (mode === "sparse") {
    return 300;
  }
  return 360;
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
  const wrapperClassName =
    surface === "panel"
      ? "surface overflow-hidden rounded-[1.6rem] border border-white/8"
      : "overflow-hidden rounded-[1.6rem] border border-white/8 bg-black/18";
  const resolvedHeight = height ?? resolveHeight(mode, dataCount);
  const sparseMode = mode === "sparse" || (mode === "comparison" && dataCount <= 4);
  const hasData = dataCount > 0;
  const insightGridClass =
    insightLayout === "stack"
      ? "grid-cols-1"
      : insightLayout === "side"
        ? "xl:grid-cols-[minmax(260px,430px)_minmax(0,1fr)]"
        : "min-[1500px]:grid-cols-[minmax(260px,430px)_minmax(0,1fr)]";

  return (
    <section data-chart-surface className={cn(wrapperClassName, className)}>
      {eyebrow || title || description || toolbar || meta ? (
        <header className="border-b border-white/8 px-5 py-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-3xl">
              {eyebrow ? (
                <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
                  {eyebrow}
                </div>
              ) : null}
              {title ? (
                <h3 className="mt-2 text-xl font-semibold tracking-[-0.03em] text-white">
                  {title}
                </h3>
              ) : null}
              {description ? (
                <p className="mt-2 text-sm leading-6 text-[var(--color-text-soft)]">
                  {description}
                </p>
              ) : null}
            </div>
            {(toolbar || meta) ? (
              <div className="flex shrink-0 flex-col items-start gap-3 lg:items-end">
                {toolbar}
                {meta ? (
                  <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                    {meta}
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        </header>
      ) : null}

      {!hasData && emptyState ? (
        <div className="px-5 py-6">{emptyState}</div>
      ) : sparseMode && insight ? (
        <div className={cn("grid gap-6 px-4 py-4 lg:px-5", insightGridClass)}>
          <div className="mx-auto w-full max-w-[430px]">
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
      ) : sparseMode ? (
        <div className="px-4 py-4 lg:px-5">
          <div className="mx-auto w-full max-w-[520px]">
            <ReactECharts
              option={option}
              style={{ height: `${resolvedHeight}px`, width: "100%" }}
              opts={{ renderer: "svg" }}
              notMerge
              lazyUpdate
            />
          </div>
        </div>
      ) : (
        <div className="px-3 py-3 sm:px-4 sm:py-4">
          <ReactECharts
            option={option}
            style={{ height: `${resolvedHeight}px`, width: "100%" }}
            opts={{ renderer: "svg" }}
            notMerge
            lazyUpdate
          />
        </div>
      )}

      {footer ? <div className="border-t border-white/8 px-5 py-4">{footer}</div> : null}
    </section>
  );
}

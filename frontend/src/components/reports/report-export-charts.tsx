"use client";

import { ChartSurface } from "@/components/charts/chart-surface";
import {
  CHART_PALETTE,
  makeBacktestOption,
  makeGroupedBarOption,
  makeLineOption,
} from "@/lib/chart-options";
import type { BacktestSeriesPoint, TimeSeriesPoint } from "@/lib/view-models";

export function ReportBacktestChart({
  points,
  height = 420,
  title,
  description,
  meta,
}: {
  points: BacktestSeriesPoint[];
  height?: number;
  title: string;
  description: string;
  meta: string;
}) {
  return (
    <ChartSurface
      option={makeBacktestOption(points)}
      height={height}
      mode="trace"
      dataCount={points.length}
      eyebrow="Analytics"
      title={title}
      description={description}
      showDescription
      meta={meta}
    />
  );
}

export function ReportCapitalChart({
  points,
  height = 420,
  title,
  description,
  meta,
}: {
  points: TimeSeriesPoint[];
  height?: number;
  title: string;
  description: string;
  meta: string;
}) {
  return (
    <ChartSurface
      option={makeLineOption(points, CHART_PALETTE.green, { mode: "standard" })}
      height={height}
      mode="standard"
      dataCount={points.length}
      eyebrow="Capital"
      title={title}
      description={description}
      showDescription
      meta={meta}
    />
  );
}

export function ReportDecisionContinuityChart({
  labels,
  requested,
  approved,
  height = 360,
  title,
  description,
}: {
  labels: string[];
  requested: number[];
  approved: number[];
  height?: number;
  title: string;
  description: string;
}) {
  return (
    <ChartSurface
      option={makeGroupedBarOption(
        labels,
        [
          {
            name: "Requested",
            data: requested,
            color: CHART_PALETTE.gold,
          },
          {
            name: "Approved",
            data: approved,
            color: CHART_PALETTE.green,
          },
        ],
        { mode: "comparison" },
      )}
      height={height}
      mode="comparison"
      dataCount={labels.length}
      eyebrow="Decision continuity"
      title={title}
      description={description}
      showDescription
    />
  );
}

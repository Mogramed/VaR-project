"use client";

import ReactECharts from "echarts-for-react";

export function ChartCore({
  option,
  height,
}: {
  option: Record<string, unknown>;
  height: number;
}) {
  return (
    <ReactECharts
      option={option}
      style={{ height: `${height}px`, width: "100%" }}
      opts={{ renderer: "svg" }}
      notMerge
      lazyUpdate
    />
  );
}

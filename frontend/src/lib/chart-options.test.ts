import { describe, expect, it } from "vitest";
import {
  CHART_PALETTE,
  makeBacktestOption,
  makeBarOption,
  makeGroupedBarOption,
  makeLineOption,
} from "@/lib/chart-options";

function readXAxisLabels(option: Record<string, unknown>): string[] {
  const xAxis = option.xAxis as { data?: string[] };
  return Array.isArray(xAxis?.data) ? xAxis.data : [];
}

function readSeriesData(option: Record<string, unknown>, index: number): unknown[] {
  const series = option.series as Array<{ data?: unknown[] }>;
  if (!Array.isArray(series) || !Array.isArray(series[index]?.data)) {
    return [];
  }
  return series[index].data as unknown[];
}

function readYAxis(option: Record<string, unknown>): { min: number; max: number } {
  const yAxis = option.yAxis as { min?: number; max?: number };
  return {
    min: Number(yAxis?.min),
    max: Number(yAxis?.max),
  };
}

describe("chart option resilience", () => {
  it("normalizes and sorts line series when timestamps are unordered", () => {
    const option = makeLineOption(
      [
        { label: "2026-01-03T00:00:00Z", value: 30 },
        { label: "2026-01-01T00:00:00Z", value: 10 },
        { label: "2026-01-02T00:00:00Z", value: Number.NaN },
        { label: "2026-01-02T00:00:00Z", value: 20 },
      ],
      CHART_PALETTE.gold,
      { mode: "standard" },
    );

    expect(readXAxisLabels(option)).toEqual([
      "2026 01 01T00:00:00Z",
      "2026 01 02T00:00:00Z",
      "2026 01 03T00:00:00Z",
    ]);
    expect(readSeriesData(option, 0)).toEqual([10, 20, 30]);
    expect(option).toMatchSnapshot();
  });

  it("filters invalid backtest rows and keeps null for optional var values", () => {
    const option = makeBacktestOption([
      { label: "2026-03-03T00:00:00Z", pnl: 30, var_hist: 15, var_garch: Number.NaN, var_fhs: 12 },
      { label: "2026-03-01T00:00:00Z", pnl: 10, var_hist: Number.NaN, var_garch: 5, var_fhs: 4 },
      { label: "2026-03-02T00:00:00Z", pnl: Number.NaN, var_hist: 8, var_garch: 6, var_fhs: 5 },
    ]);

    expect(readXAxisLabels(option)).toEqual(["2026 03 01T00:00:00Z", "2026 03 03T00:00:00Z"]);
    expect(readSeriesData(option, 0)).toEqual([10, 30]);
    expect(readSeriesData(option, 1)).toEqual([null, 15]);
    expect(readSeriesData(option, 2)).toEqual([5, null]);
    expect(readSeriesData(option, 3)).toEqual([4, 12]);
    expect(option).toMatchSnapshot();
  });

  it("drops grouped categories where all series are invalid", () => {
    const option = makeGroupedBarOption(
      ["A", "B", "C", "D"],
      [
        { name: "Requested", data: [100, Number.NaN, 300, Number.POSITIVE_INFINITY], color: CHART_PALETTE.gold },
        { name: "Approved", data: [50, null, undefined, -20], color: CHART_PALETTE.green },
      ],
      { mode: "comparison" },
    );

    expect(readXAxisLabels(option)).toEqual(["A", "C", "D"]);
    expect(readSeriesData(option, 0)).toEqual([100, 300, null]);
    expect(readSeriesData(option, 1)).toEqual([50, null, -20]);
    expect(option).toMatchSnapshot();
  });

  it("returns a safe empty chart shape when all bar values are invalid", () => {
    const option = makeBarOption([
      { label: "first", value: Number.NaN },
      { label: "second", value: Number.POSITIVE_INFINITY },
    ]);

    expect(readXAxisLabels(option)).toEqual([]);
    expect(readSeriesData(option, 0)).toEqual([]);
    expect(readYAxis(option)).toEqual({ min: 0, max: 1 });
    expect(option).toMatchSnapshot();
  });

  it("anchors bar axis to zero when all values are positive", () => {
    const option = makeBarOption([
      { label: "alpha", value: 100 },
      { label: "beta", value: 250 },
    ]);

    const yAxis = readYAxis(option);
    expect(yAxis.min).toBe(0);
    expect(yAxis.max).toBeGreaterThan(250);
    expect(option).toMatchSnapshot();
  });

  it("sorts valid timestamps first and appends invalid labels in line series", () => {
    const option = makeLineOption(
      [
        { label: "not-a-date", value: 1 },
        { label: "2026-01-02T00:00:00Z", value: 2 },
        { label: "2026-01-01T00:00:00Z", value: 3 },
      ],
      CHART_PALETTE.blue,
      { mode: "standard" },
    );

    expect(readXAxisLabels(option)).toEqual(["2026 01 01T00:00:00Z", "2026 01 02T00:00:00Z", "1"]);
    expect(readSeriesData(option, 0)).toEqual([3, 2, 1]);
    expect(option).toMatchSnapshot();
  });

  it("treats numeric placeholder labels as undated during time sorting", () => {
    const option = makeLineOption(
      [
        { label: "2026-01-02T00:00:00Z", value: 20 },
        { label: "1", value: 10 },
        { label: "2026-01-01T00:00:00Z", value: 30 },
      ],
      CHART_PALETTE.blue,
      { mode: "standard" },
    );

    expect(readXAxisLabels(option)).toEqual(["2026 01 01T00:00:00Z", "2026 01 02T00:00:00Z", "2"]);
    expect(readSeriesData(option, 0)).toEqual([30, 20, 10]);
  });
});

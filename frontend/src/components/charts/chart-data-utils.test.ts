import { describe, expect, it } from "vitest";
import { countRenderablePoints, isRenderableDatum } from "@/components/charts/chart-data-utils";

describe("chart data utils", () => {
  it("recognizes renderable datum shapes used by ECharts", () => {
    expect(isRenderableDatum(12)).toBe(true);
    expect(isRenderableDatum([0, 12])).toBe(true);
    expect(isRenderableDatum({ value: 12 })).toBe(true);
    expect(isRenderableDatum({ value: [0, 12] })).toBe(true);
    expect(isRenderableDatum({ value: null })).toBe(false);
    expect(isRenderableDatum(Number.NaN)).toBe(false);
    expect(isRenderableDatum(undefined)).toBe(false);
  });

  it("counts renderable points from mixed series payloads", () => {
    const option = {
      series: [
        {
          type: "line",
          data: [1, null, Number.NaN, [0, 2], { value: [1, 3] }],
        },
        {
          type: "bar",
          data: [{ value: 4 }, { value: null }, { value: Number.POSITIVE_INFINITY }],
        },
        {
          type: "line",
        },
      ],
    };

    expect(countRenderablePoints(option)).toBe(4);
  });

  it("returns zero for non-series or fully invalid payload", () => {
    expect(countRenderablePoints({})).toBe(0);
    expect(
      countRenderablePoints({
        series: [{ data: [null, Number.NaN, { value: null }] }],
      }),
    ).toBe(0);
  });
});

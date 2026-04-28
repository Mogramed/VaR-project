import { describe, expect, it } from "vitest";
import type { BacktestFrameResponse } from "@/lib/api/types";
import { buildBacktestSeries } from "@/lib/view-models";

describe("buildBacktestSeries", () => {
  it("keeps pnl undefined when source rows do not provide pnl", () => {
    const frame = {
      rows: [
        { timestamp: "2026-03-01T00:00:00Z", pnl: null, var_hist: 10, var_alpha: 9.5 },
        { timestamp: "2026-03-02T00:00:00Z", pnl: "12.5", var_hist: 11, var_alpha: "10.7" },
      ],
    } as unknown as BacktestFrameResponse;

    const points = buildBacktestSeries(frame);

    expect(points).toHaveLength(2);
    expect(points[0]).toMatchObject({
      label: "2026-03-01T00:00:00.000Z",
      pnl: undefined,
      var_hist: 10,
      var_alpha: 9.5,
    });
    expect(points[1]).toMatchObject({
      label: "2026-03-02T00:00:00.000Z",
      pnl: 12.5,
      var_hist: 11,
      var_alpha: 10.7,
    });
  });
});

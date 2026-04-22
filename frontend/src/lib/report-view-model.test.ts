import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MT5LiveStateResponse, ReportContentResponse } from "@/lib/api/types";

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    safeHealth: vi.fn(),
    mt5LiveState: vi.fn(),
    latestSnapshot: vi.fn(),
    latestReport: vi.fn(),
    reportDecisionHistory: vi.fn(),
    reportCapitalHistory: vi.fn(),
    recentAudit: vi.fn(),
    latestModelComparison: vi.fn(),
    latestValidation: vi.fn(),
    latestBacktestFrame: vi.fn(),
    deskOverview: vi.fn(),
    latestCapital: vi.fn(),
  },
}));

vi.mock("@/lib/api/client", () => ({
  api: apiMock,
}));

import { loadDeskReportViewModel, resolveReportContract } from "@/lib/report-view-model";

describe("report contract parity", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.safeHealth.mockResolvedValue({
      status: "ok",
      repo_root: ".",
      database_url: "sqlite:///test.db",
      portfolio_slug: "fx_eur_20k",
      portfolio_mode: null,
      portfolio_count: 1,
      desk_slug: "main",
      latest_artifacts: {},
      defaults: {},
      dependencies: {},
    });
    apiMock.mt5LiveState.mockResolvedValue(null);
    apiMock.latestSnapshot.mockResolvedValue({
      payload: {
        var: { hist: 9999 },
        es: { hist: 9999 },
      },
      source: "mt5_live_bridge",
    });
    apiMock.latestReport.mockResolvedValue({
      report_id: 42,
      report_markdown: "/tmp/report.md",
      portfolio_slug: "fx_eur_20k",
      account_id: "demo",
      content: "# Risk Report\n\n## Portfolio Snapshot",
      chart_paths: [],
      report_contract: {
        version: "report.v1",
        timezone: "UTC",
        generated_at_utc: "2026-04-22T10:00:00+00:00",
        selected_model: "hist",
        snapshot_source: "mt5_live_bridge",
        snapshot_timestamp_utc: "2026-04-22T09:55:00+00:00",
        rounding: {
          money_decimals: 2,
          percent_decimals: 1,
        },
        metrics: {
          var: { value: 1234.56, display: "1,234.56", as_of_utc: "2026-04-22T09:55:00+00:00" },
          es: { value: 2345.67, display: "2,345.67", as_of_utc: "2026-04-22T09:55:00+00:00" },
          pnl: { value: -45.67, display: "-45.67", as_of_utc: "2026-04-22T09:50:00+00:00" },
        },
      },
    });
    apiMock.reportDecisionHistory.mockResolvedValue([]);
    apiMock.reportCapitalHistory.mockResolvedValue([]);
    apiMock.recentAudit.mockResolvedValue([]);
    apiMock.latestModelComparison.mockResolvedValue({ champion_model: "hist", score_gap: 0.2, ranking: [] });
    apiMock.latestValidation.mockResolvedValue({ summary: {} });
    apiMock.latestBacktestFrame.mockResolvedValue({ compare_csv: "/tmp/compare.csv", rows: [] });
    apiMock.deskOverview.mockResolvedValue(null);
    apiMock.latestCapital.mockResolvedValue({
      base_currency: "EUR",
      headroom_ratio: 0.25,
      total_capital_remaining_eur: 10000,
      total_capital_consumed_eur: 5000,
      total_capital_budget_eur: 15000,
      status: "ok",
      reference_model: "hist",
    });
  });

  it("parses a versioned contract payload safely", () => {
    const resolved = resolveReportContract({
      report_contract: {
        version: "report.v1",
        timezone: "UTC",
        selected_model: "hist",
        snapshot_source: "mt5_live_bridge",
        rounding: { money_decimals: 2, percent_decimals: 1 },
        metrics: {
          var: { value: 10, display: "10.00", as_of_utc: "2026-04-22T00:00:00+00:00" },
          es: { value: 12, display: "12.00", as_of_utc: "2026-04-22T00:00:00+00:00" },
          pnl: { value: 1, display: "1.00", as_of_utc: "2026-04-22T00:00:00+00:00" },
        },
      },
    } as unknown as Pick<ReportContentResponse, "report_contract">);

    expect(resolved?.version).toBe("report.v1");
    expect(resolved?.timezone).toBe("UTC");
    expect(resolved?.selectedModel).toBe("hist");
    expect(resolved?.moneyDecimals).toBe(2);
    expect(resolved?.metrics.var.display).toBe("10.00");
  });

  it("prioritizes API contract values for report metrics and export parity", async () => {
    const view = await loadDeskReportViewModel("fx_eur_20k", {
      liveState: {
        risk_summary: {
          reference_model: "hist",
          var: { hist: 99999 },
          es: { hist: 88888 },
        },
      } as unknown as MT5LiveStateResponse,
      accountId: "demo",
    });

    expect(view.selectedModel).toBe("hist");
    expect(view.varValue).toBeCloseTo(1234.56, 6);
    expect(view.esValue).toBeCloseTo(2345.67, 6);
    expect(view.pnlValue).toBeCloseTo(-45.67, 6);
    expect(view.varDisplay).toBe("1,234.56");
    expect(view.esDisplay).toBe("2,345.67");
    expect(view.pnlDisplay).toBe("-45.67");
    expect(view.meta.preferredSnapshotSource).toBe("mt5_live_bridge");
    expect(view.meta.reportContractVersion).toBe("report.v1");
    expect(view.executiveSummary).toMatchSnapshot();
  });
});

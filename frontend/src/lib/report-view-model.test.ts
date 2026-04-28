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

import { loadDeskReportViewModel, normalizeReportContent, resolveReportContract } from "@/lib/report-view-model";

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
    expect(apiMock.latestModelComparison).toHaveBeenCalledWith("fx_eur_20k", 42);
    expect(apiMock.latestValidation).toHaveBeenCalledWith("fx_eur_20k", 42);
    expect(apiMock.latestBacktestFrame).toHaveBeenCalledWith("fx_eur_20k", 260, 42);
    expect(view.executiveSummary).toMatchSnapshot();
  });

  it("falls back to live latest endpoints when no persisted report exists", async () => {
    apiMock.latestReport.mockResolvedValue(null);

    await loadDeskReportViewModel("fx_eur_20k", {
      liveState: null,
      accountId: "demo",
    });

    expect(apiMock.latestModelComparison).toHaveBeenCalledWith("fx_eur_20k", undefined);
    expect(apiMock.latestValidation).toHaveBeenCalledWith("fx_eur_20k", undefined);
    expect(apiMock.latestBacktestFrame).toHaveBeenCalledWith("fx_eur_20k", 260, undefined);
  });

  it("falls back to computed displays when contract display placeholders are returned", async () => {
    apiMock.latestReport.mockResolvedValue({
      report_id: 43,
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
          var: { value: null, display: "n/a", as_of_utc: "2026-04-22T09:55:00+00:00" },
          es: { value: null, display: "NA", as_of_utc: "2026-04-22T09:55:00+00:00" },
          pnl: { value: null, display: "-", as_of_utc: "2026-04-22T09:50:00+00:00" },
        },
      },
    });
    apiMock.latestBacktestFrame.mockResolvedValue({
      compare_csv: "/tmp/compare.csv",
      rows: [{ timestamp: "2026-04-22T09:50:00+00:00", pnl: -12.34 }],
    });

    const view = await loadDeskReportViewModel("fx_eur_20k", {
      liveState: {
        risk_summary: {
          reference_model: "hist",
          var: { hist: 77.7 },
          es: { hist: 88.8 },
        },
      } as unknown as MT5LiveStateResponse,
      accountId: "demo",
    });

    expect(view.varValue).toBeCloseTo(9999, 6);
    expect(view.esValue).toBeCloseTo(9999, 6);
    expect(view.pnlValue).toBeCloseTo(0, 6);
    expect(view.varDisplay).toBe("9,999.00");
    expect(view.esDisplay).toBe("9,999.00");
    expect(view.pnlDisplay).toBe("n/a");
  });

  it("uses report-scoped capital and decisions instead of newer live values", async () => {
    apiMock.reportDecisionHistory.mockResolvedValue([
      {
        symbol: "EURUSD",
        decision: "ACCEPT",
        requested_exposure_change: 1000,
        approved_exposure_change: 500,
        resulting_exposure: 10500,
        model_used: "hist",
        reasons: [],
        pre_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
        post_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
        created_at: "2026-04-22T09:40:00+00:00",
      },
      {
        symbol: "EURUSD",
        decision: "ACCEPT",
        requested_exposure_change: 1000,
        approved_exposure_change: 1000,
        resulting_exposure: 11500,
        model_used: "hist",
        reasons: [],
        pre_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
        post_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
        created_at: "2026-04-22T11:40:00+00:00",
      },
    ]);
    apiMock.reportCapitalHistory.mockResolvedValue([
      {
        portfolio_slug: "fx_eur_20k",
        base_currency: "EUR",
        reference_model: "hist",
        snapshot_source: "mt5_live_bridge",
        snapshot_timestamp: "2026-04-22T09:45:00+00:00",
        total_capital_budget_eur: 1000,
        total_capital_consumed_eur: 500,
        total_capital_reserved_eur: 80,
        total_capital_remaining_eur: 420,
        headroom_ratio: 0.42,
        status: "ok",
        budget: { total_var: 0, total_es: 0, total_var_budget: 0, total_es_budget: 0, utilization_var: 0, utilization_es: 0, status: "ok" },
      },
      {
        portfolio_slug: "fx_eur_20k",
        base_currency: "EUR",
        reference_model: "hist",
        snapshot_source: "mt5_live_bridge",
        snapshot_timestamp: "2026-04-22T11:45:00+00:00",
        total_capital_budget_eur: 1000,
        total_capital_consumed_eur: 880,
        total_capital_reserved_eur: 80,
        total_capital_remaining_eur: 40,
        headroom_ratio: 0.04,
        status: "warn",
        budget: { total_var: 0, total_es: 0, total_var_budget: 0, total_es_budget: 0, utilization_var: 0, utilization_es: 0, status: "warn" },
      },
    ]);
    apiMock.recentAudit.mockResolvedValue([
      {
        actor: "system",
        action_type: "report.run",
        object_type: "report",
        object_id: "42",
        metadata: null,
        created_at: "2026-04-22T10:00:00+00:00",
      },
      {
        actor: "operator",
        action_type: "execution.submit",
        object_type: "execution",
        object_id: "901",
        metadata: null,
        created_at: "2026-04-22T11:10:00+00:00",
      },
    ]);

    const view = await loadDeskReportViewModel("fx_eur_20k", {
      liveState: {
        capital_usage: {
          portfolio_slug: "fx_eur_20k",
          base_currency: "EUR",
          reference_model: "hist",
          snapshot_source: "mt5_live_bridge",
          total_capital_budget_eur: 1000,
          total_capital_consumed_eur: 950,
          total_capital_reserved_eur: 40,
          total_capital_remaining_eur: 10,
          headroom_ratio: 0.01,
          status: "breach",
          budget: { total_var: 0, total_es: 0, total_var_budget: 0, total_es_budget: 0, utilization_var: 0, utilization_es: 0, status: "breach" },
        },
      } as unknown as MT5LiveStateResponse,
      accountId: "demo",
    });

    expect(view.decisions).toHaveLength(1);
    expect(view.fillRatio).toBeCloseTo(0.5, 6);
    expect(view.capital?.headroom_ratio).toBeCloseTo(0.42, 6);
    expect(view.capitalHistory).toHaveLength(1);
    expect(view.audit).toHaveLength(1);
    expect(view.audit[0]?.action_type).toBe("report.run");
    expect(view.executiveSummary.find((item) => item.label === "Capital headroom")?.value).toBe("42%");
  });

  it("anchors report cutoff to the matching report.run audit event", async () => {
    apiMock.latestReport.mockResolvedValue({
      report_id: 44,
      report_markdown: "/tmp/report-44.md",
      portfolio_slug: "fx_eur_20k",
      account_id: "demo",
      content: "# Risk Report\n\n## Portfolio Snapshot",
      chart_paths: [],
      report_contract: {
        version: "report.v1",
        timezone: "UTC",
        selected_model: "hist",
        snapshot_source: "mt5_live_bridge",
        metrics: {
          var: { value: 100, display: "100.00", as_of_utc: null },
          es: { value: 120, display: "120.00", as_of_utc: null },
          pnl: { value: -5, display: "-5.00", as_of_utc: null },
        },
      },
    });
    apiMock.recentAudit.mockResolvedValue([
      {
        actor: "api",
        action_type: "report.run",
        object_type: "daily_report",
        object_id: 45,
        payload: { report_markdown: "/tmp/report-45.md", account_id: "demo" },
        created_at: "2026-04-22T11:00:00+00:00",
      },
      {
        actor: "api",
        action_type: "report.run",
        object_type: "daily_report",
        object_id: 44,
        payload: { report_markdown: "/tmp/report-44.md", account_id: "demo" },
        created_at: "2026-04-22T10:00:00+00:00",
      },
    ]);
    apiMock.reportDecisionHistory.mockResolvedValue([
      {
        symbol: "EURUSD",
        decision: "ACCEPT",
        requested_exposure_change: 1000,
        approved_exposure_change: 1000,
        resulting_exposure: 11500,
        model_used: "hist",
        reasons: [],
        pre_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
        post_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
        created_at: "2026-04-22T10:05:00+00:00",
      },
      {
        symbol: "EURUSD",
        decision: "ACCEPT",
        requested_exposure_change: 500,
        approved_exposure_change: 250,
        resulting_exposure: 10250,
        model_used: "hist",
        reasons: [],
        pre_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
        post_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
        created_at: "2026-04-22T09:55:00+00:00",
      },
    ]);

    const view = await loadDeskReportViewModel("fx_eur_20k", {
      liveState: null,
      accountId: "demo",
    });

    expect(view.latestReportEvent?.created_at).toBe("2026-04-22T10:00:00+00:00");
    expect(view.decisions).toHaveLength(1);
    expect(view.decisions[0]?.created_at).toBe("2026-04-22T09:55:00+00:00");
  });

  it("expands report history windows when initial slices are too recent for cutoff filtering", async () => {
    apiMock.latestReport.mockResolvedValue({
      report_id: 46,
      report_markdown: "/tmp/report-46.md",
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
        metrics: {
          var: { value: 100, display: "100.00", as_of_utc: null },
          es: { value: 120, display: "120.00", as_of_utc: null },
          pnl: { value: -5, display: "-5.00", as_of_utc: null },
        },
      },
    });

    const recentDecisionTemplate = {
      symbol: "EURUSD",
      decision: "ACCEPT",
      requested_exposure_change: 1000,
      approved_exposure_change: 1000,
      resulting_exposure: 12000,
      model_used: "hist",
      reasons: [],
      pre_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
      post_trade: { var: 1, es: 2, headroom_var: 0, headroom_es: 0, gross_exposure: 0, symbol_exposure: 0, status: "ok" },
    };
    const initialRecentDecisions = Array.from({ length: 12 }, (_, index) => ({
      ...recentDecisionTemplate,
      created_at: `2026-04-23T12:${String(index).padStart(2, "0")}:00+00:00`,
    }));
    const reportScopedDecision = {
      ...recentDecisionTemplate,
      requested_exposure_change: 600,
      approved_exposure_change: 300,
      created_at: "2026-04-21T09:59:00+00:00",
    };
    apiMock.reportDecisionHistory
      .mockResolvedValueOnce(initialRecentDecisions)
      .mockResolvedValueOnce([...initialRecentDecisions, reportScopedDecision]);

    const initialRecentCapital = Array.from({ length: 8 }, (_, index) => ({
      portfolio_slug: "fx_eur_20k",
      base_currency: "EUR",
      reference_model: "hist",
      snapshot_source: "mt5_live_bridge",
      snapshot_timestamp: `2026-04-23T10:${String(index).padStart(2, "0")}:00+00:00`,
      total_capital_budget_eur: 1000,
      total_capital_consumed_eur: 900,
      total_capital_reserved_eur: 50,
      total_capital_remaining_eur: 50,
      headroom_ratio: 0.05,
      status: "warn",
      budget: { total_var: 0, total_es: 0, total_var_budget: 0, total_es_budget: 0, utilization_var: 0, utilization_es: 0, status: "warn" },
    }));
    const reportScopedCapital = {
      portfolio_slug: "fx_eur_20k",
      base_currency: "EUR",
      reference_model: "hist",
      snapshot_source: "mt5_live_bridge",
      snapshot_timestamp: "2026-04-21T09:58:00+00:00",
      total_capital_budget_eur: 1000,
      total_capital_consumed_eur: 650,
      total_capital_reserved_eur: 50,
      total_capital_remaining_eur: 300,
      headroom_ratio: 0.3,
      status: "ok",
      budget: { total_var: 0, total_es: 0, total_var_budget: 0, total_es_budget: 0, utilization_var: 0, utilization_es: 0, status: "ok" },
    };
    apiMock.reportCapitalHistory
      .mockResolvedValueOnce(initialRecentCapital)
      .mockResolvedValueOnce([...initialRecentCapital, reportScopedCapital]);

    const view = await loadDeskReportViewModel("fx_eur_20k", {
      liveState: null,
      accountId: "demo",
    });

    expect(apiMock.reportDecisionHistory).toHaveBeenNthCalledWith(1, "fx_eur_20k", 12, "demo");
    expect(apiMock.reportDecisionHistory).toHaveBeenNthCalledWith(2, "fx_eur_20k", 200, "demo");
    expect(apiMock.reportCapitalHistory).toHaveBeenNthCalledWith(1, "fx_eur_20k", 8, "mt5_live_bridge");
    expect(apiMock.reportCapitalHistory).toHaveBeenNthCalledWith(2, "fx_eur_20k", 200, "mt5_live_bridge");
    expect(view.decisions).toHaveLength(1);
    expect(view.decisions[0]?.created_at).toBe("2026-04-21T09:59:00+00:00");
    expect(view.capital?.headroom_ratio).toBeCloseTo(0.3, 6);
    expect(view.capitalHistory).toHaveLength(1);
  });

  it("repairs legacy validation markdown tables with ES note injected inside rows", () => {
    const raw = [
      "## Model Validation",
      "| Model | Rank |",
      "|---|---:|",
      "_ES tail diagnostics are measured on VaR exceedance observations (tail observations where portfolio loss is greater than VaR)._",
      "",
      "| FHS | 1 |",
      "",
    ].join("\n");
    const normalized = normalizeReportContent(raw);
    expect(normalized).toContain("|---|---:|\n| FHS | 1 |");
    expect(normalized).toContain("| FHS | 1 |\n\n_ES tail diagnostics are measured on VaR exceedance observations");
  });

  it("inserts spacing before markdown tables following attribution bullets", () => {
    const raw = [
      "### Risk Contributions",
      "- Attribution model: **FHS**",
      "| Symbol | Asset class |",
      "|---|---|",
      "| EURUSD | fx |",
    ].join("\n");
    const normalized = normalizeReportContent(raw);
    expect(normalized).toContain("- Attribution model: **FHS**\n\n| Symbol | Asset class |");
  });
});

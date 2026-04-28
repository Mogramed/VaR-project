import { describe, expect, it } from "vitest";

import { __apiClientTestables } from "@/lib/api/client";

describe("API client timeout policy", () => {
  it("uses extended timeout for long-running execution and stress endpoints", () => {
    expect(__apiClientTestables.resolveRequestTimeoutMs("POST", "/execution/submit")).toBe(120_000);
    expect(__apiClientTestables.resolveRequestTimeoutMs("POST", "/execution/preview")).toBe(90_000);
    expect(__apiClientTestables.resolveRequestTimeoutMs("POST", "/decisions/evaluate")).toBe(90_000);
    expect(__apiClientTestables.resolveRequestTimeoutMs("POST", "/snapshots/stress")).toBe(180_000);
  });

  it("keeps strict operator contract for enqueue and status polling", () => {
    expect(__apiClientTestables.resolveRequestTimeoutMs("POST", "/operator/actions/sync")).toBe(12_000);
    expect(__apiClientTestables.resolveRequestTimeoutMs("GET", "/operator/runs")).toBe(10_000);
  });

  it("returns actionable timeout hints", () => {
    expect(__apiClientTestables.resolveTimeoutHint("/execution/submit").toLowerCase()).toContain("blotter");
    expect(__apiClientTestables.resolveTimeoutHint("/decisions/evaluate").toLowerCase()).toContain("risk service");
    expect(__apiClientTestables.resolveTimeoutHint("/snapshots/stress").toLowerCase()).toContain("long-running");
  });

  it("returns contextual network failure hints", () => {
    expect(__apiClientTestables.resolveNetworkFailureHint("/operator/actions/report")).toContain("/operator/runs");
    expect(__apiClientTestables.resolveNetworkFailureHint("/execution/submit")).toContain("API/worker/MT5");
    expect(__apiClientTestables.resolveNetworkFailureHint("/snapshots/stress")).toContain("Stress endpoint");
  });

  it("builds browser-facing export hrefs through the frontend proxy", () => {
    expect(
      __apiClientTestables.buildBrowserHref("/mt5/history/transactions/export", {
        portfolio_slug: "mt5_live_portfolio",
        account_id: "default",
        symbol: "EURUSD",
        max_rows: 5000,
      }),
    ).toBe(
      "/api/proxy/mt5/history/transactions/export?portfolio_slug=mt5_live_portfolio&account_id=default&symbol=EURUSD&max_rows=5000",
    );
  });
});

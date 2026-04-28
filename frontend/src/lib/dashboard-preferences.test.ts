import { describe, expect, it } from "vitest";

import {
  PRESETS,
  horizonDays,
  normalizeDashboardPreferences,
  resolveModelPreference,
  symbolFilterTokens,
  symbolMatchesFilter,
} from "@/lib/dashboard-preferences";

describe("dashboard preferences", () => {
  it("normalizes model aliases and detects preset state", () => {
    const normalized = normalizeDashboardPreferences(
      JSON.parse("{\"model\":\"parametric\"}"),
    );

    expect(normalized.model).toBe("param");

    const preset = normalizeDashboardPreferences(PRESETS["risk-monitoring"]);
    expect(preset.activePreset).toBe("risk-monitoring");
  });

  it("keeps overview page visible", () => {
    const normalized = normalizeDashboardPreferences({
      visiblePages: ["models"],
    });

    expect(normalized.visiblePages).toEqual(["overview", "models"]);
  });

  it("auto-includes alpha pages for legacy full sidebar preferences", () => {
    const normalized = normalizeDashboardPreferences({
      visiblePages: [
        "overview",
        "live",
        "incidents",
        "universe",
        "models",
        "attribution",
        "capital",
        "decisions",
        "execution",
        "stress",
        "blotter",
        "reports",
      ],
    });

    expect(normalized.visiblePages).toContain("alpha-features");
    expect(normalized.visiblePages).toContain("alpha-performance");
  });

  it("tokenizes and matches symbol filters in a case-insensitive way", () => {
    expect(symbolFilterTokens(" eurusd, usdjpy  ")).toEqual(["EURUSD", "USDJPY"]);
    expect(symbolMatchesFilter("eurusd.a", "EURUSD,GBPUSD")).toBe(true);
    expect(symbolMatchesFilter("xauusd", "EURUSD,GBPUSD")).toBe(false);
    expect(symbolMatchesFilter(null, "EURUSD")).toBe(false);
    expect(symbolMatchesFilter("xauusd", "")).toBe(true);
  });

  it("resolves preferred model and horizon days", () => {
    expect(resolveModelPreference("auto", "cornish_fisher")).toBe("fhs");
    expect(resolveModelPreference("garch", "hist")).toBe("garch");
    expect(resolveModelPreference("auto", null)).toBeNull();

    expect(horizonDays("1d")).toBe(1);
    expect(horizonDays("5d")).toBe(5);
    expect(horizonDays("10d")).toBe(10);
  });
});

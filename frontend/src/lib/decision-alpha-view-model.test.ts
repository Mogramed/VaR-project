import { describe, expect, it } from "vitest";
import type {
  DecisionBacktestTrajectoryResponse,
  DecisionForecastResponse,
  DecisionIntelligenceResponse,
  DecisionPortfolioForecastResponse,
  DecisionReplayResponse,
  RiskDecisionResponse,
} from "@/lib/api/types";
import {
  decisionAlphaFeatureRows,
  extractDecisionIntelligence,
  forecastChartSeries,
  portfolioPnlScenarioSeries,
  projectionHistoryForecastSeries,
  replayChartSeries,
  trajectoryChartSeries,
} from "@/lib/decision-alpha-view-model";

describe("decision alpha view model", () => {
  it("extracts and orders feature rows by contribution magnitude", () => {
    const intelligence = {
      signal: "BUY",
      score: 38,
      confidence: 0.66,
      size_multiplier: 0.52,
      top_drivers: ["momentum:+0.44"],
      model_version: "decision_alpha_v1",
      guardrail_applied: false,
      features: {
        momentum_short_term: 0.42,
        volatility_recent: 0.05,
      },
      feature_contributions: {
        momentum_short_term: 0.44,
        volatility_recent: -0.10,
      },
      calculations: {},
    } as DecisionIntelligenceResponse;

    const rows = decisionAlphaFeatureRows(intelligence);

    expect(rows).toHaveLength(2);
    expect(rows[0].key).toBe("momentum_short_term");
    expect(rows[0].label).toContain("Momentum");
    expect(rows[0].contribution).toBeCloseTo(0.44, 5);
  });

  it("marks unavailable features as n/a-ready rows", () => {
    const intelligence = {
      signal: "HOLD",
      score: 0,
      confidence: 0.5,
      size_multiplier: 0,
      top_drivers: [],
      model_version: "decision_alpha_v1",
      guardrail_applied: false,
      features: {
        slippage_points: 0,
        spread_cost_norm: 0,
      },
      feature_contributions: {
        slippage_points: -0.05,
        spread_cost_norm: -0.1,
      },
      calculations: {
        feature_available_slippage_points: 0,
        feature_available_spread_cost_norm: 1,
      },
    } as DecisionIntelligenceResponse;

    const rows = decisionAlphaFeatureRows(intelligence);
    const slippage = rows.find((row) => row.key === "slippage_points");
    const spread = rows.find((row) => row.key === "spread_cost_norm");

    expect(slippage?.available).toBe(false);
    expect(slippage?.value).toBeNull();
    expect(slippage?.contribution).toBeNull();
    expect(spread?.available).toBe(true);
    expect(spread?.value).toBe(0);
  });

  it("builds replay and forecast chart series", () => {
    const replay = {
      model_version: "decision_alpha_v1",
      generated_at: "2026-04-25T09:00:00Z",
      sample_size: 2,
      hit_rate: 0.5,
      cum_pnl: 8,
      comparables: 2,
      predicted_vs_realized: [
        {
          timestamp: "2026-04-24T09:00:00Z",
          symbol: "EURUSD",
          predicted_score: 25,
          predicted_signal: "BUY",
          realized_pnl: 10,
          hit: true,
          cum_pnl: 10,
        },
        {
          timestamp: "2026-04-25T09:00:00Z",
          symbol: "EURUSD",
          predicted_score: -12,
          predicted_signal: "SELL",
          realized_pnl: -2,
          hit: true,
          cum_pnl: 8,
        },
      ],
    } as DecisionReplayResponse;
    const forecast = {
      model_version: "decision_alpha_v1",
      generated_at: "2026-04-25T09:00:00Z",
      symbol: "EURUSD",
      horizon_days: 2,
      current_price: 1.1,
      score: 11,
      probability_up: 0.56,
      features: {},
      scenarios: [
        {
          name: "bear",
          probability: 0.3,
          projected_return: -0.01,
          path: [
            { day: 0, date: "2026-04-25", price: 1.1 },
            { day: 1, date: "2026-04-26", price: 1.09 },
            { day: 2, date: "2026-04-27", price: 1.08 },
          ],
        },
        {
          name: "base",
          probability: 0.4,
          projected_return: 0.002,
          path: [
            { day: 0, date: "2026-04-25", price: 1.1 },
            { day: 1, date: "2026-04-26", price: 1.101 },
            { day: 2, date: "2026-04-27", price: 1.102 },
          ],
        },
        {
          name: "bull",
          probability: 0.3,
          projected_return: 0.01,
          path: [
            { day: 0, date: "2026-04-25", price: 1.1 },
            { day: 1, date: "2026-04-26", price: 1.11 },
            { day: 2, date: "2026-04-27", price: 1.12 },
          ],
        },
      ],
    } as DecisionForecastResponse;

    const replaySeries = replayChartSeries(replay);
    const forecastSeries = forecastChartSeries(forecast);

    expect(replaySeries.labels).toEqual([
      "2026-04-24T09:00:00Z",
      "2026-04-25T09:00:00Z",
    ]);
    expect(replaySeries.realized).toEqual([10, -2]);
    expect(forecastSeries.labels).toEqual([
      "2026-04-25",
      "2026-04-26",
      "2026-04-27",
    ]);
    expect(forecastSeries.bear).toEqual([1.1, 1.09, 1.08]);
    expect(forecastSeries.base).toEqual([1.1, 1.101, 1.102]);
    expect(forecastSeries.bull).toEqual([1.1, 1.11, 1.12]);
  });

  it("handles missing intelligence safely", () => {
    const decision = {
      decision_intelligence: null,
    } as unknown as Pick<RiskDecisionResponse, "decision_intelligence">;

    expect(extractDecisionIntelligence(decision)).toBeNull();
    expect(decisionAlphaFeatureRows(null)).toEqual([]);
  });

  it("builds trajectory and portfolio pnl series", () => {
    const trajectory = {
      model_version: "decision_alpha_v1",
      generated_at: "2026-04-25T09:00:00Z",
      symbol: "EURUSD",
      lookback_days: 90,
      sample_size: 2,
      hit_rate: 0.5,
      mean_abs_error: 0.002,
      predicted_vs_actual: [
        {
          timestamp: "2026-04-24T09:00:00Z",
          predicted_price: 1.1,
          actual_price: 1.101,
          predicted_return: 0.001,
          realized_return: 0.0012,
          predicted_score: 10,
          hit: true,
        },
        {
          timestamp: "2026-04-25T09:00:00Z",
          predicted_price: 1.102,
          actual_price: 1.099,
          predicted_return: 0.001,
          realized_return: -0.001,
          predicted_score: 9,
          hit: false,
        },
      ],
    } as DecisionBacktestTrajectoryResponse;
    const portfolioForecast = {
      model_version: "decision_alpha_v1",
      generated_at: "2026-04-25T09:00:00Z",
      portfolio_slug: "fx_eur_20k",
      horizon_days: 150,
      symbol_count: 1,
      current_notional_eur: 100000,
      symbols: [],
      pnl_scenarios: [
        {
          name: "bear",
          probability: 0.3,
          projected_return: -0.02,
          path: [
            { day: 0, date: "2026-04-25", pnl: 0 },
            { day: 1, date: "2026-04-26", pnl: -120 },
          ],
        },
        {
          name: "base",
          probability: 0.4,
          projected_return: 0.0,
          path: [
            { day: 0, date: "2026-04-25", pnl: 0 },
            { day: 1, date: "2026-04-26", pnl: 20 },
          ],
        },
        {
          name: "bull",
          probability: 0.3,
          projected_return: 0.03,
          path: [
            { day: 0, date: "2026-04-25", pnl: 0 },
            { day: 1, date: "2026-04-26", pnl: 190 },
          ],
        },
      ],
    } as DecisionPortfolioForecastResponse;

    const trajectorySeries = trajectoryChartSeries(trajectory);
    const portfolioSeries = portfolioPnlScenarioSeries(portfolioForecast);

    expect(trajectorySeries.labels).toEqual([
      "2026-04-24T09:00:00Z",
      "2026-04-25T09:00:00Z",
    ]);
    expect(trajectorySeries.predicted).toEqual([1.1, 1.102]);
    expect(trajectorySeries.actual).toEqual([1.101, 1.099]);
    expect(portfolioSeries.labels).toEqual(["2026-04-25", "2026-04-26"]);
    expect(portfolioSeries.bear).toEqual([0, -120]);
    expect(portfolioSeries.base).toEqual([0, 20]);
    expect(portfolioSeries.bull).toEqual([0, 190]);
  });

  it("builds a merged history + forecast projection timeline", () => {
    const trajectory = {
      model_version: "decision_alpha_v1",
      generated_at: "2026-04-25T09:00:00Z",
      symbol: "EURUSD",
      lookback_days: 30,
      sample_size: 3,
      hit_rate: 0.66,
      mean_abs_error: 0.0015,
      predicted_vs_actual: [
        {
          timestamp: "2026-04-23T09:00:00Z",
          predicted_price: 1.1,
          actual_price: 1.101,
          predicted_return: 0.001,
          realized_return: 0.0012,
          predicted_score: 10,
          hit: true,
        },
        {
          timestamp: "2026-04-24T09:00:00Z",
          predicted_price: 1.102,
          actual_price: 1.103,
          predicted_return: 0.001,
          realized_return: 0.001,
          predicted_score: 9,
          hit: true,
        },
        {
          timestamp: "2026-04-25T09:00:00Z",
          predicted_price: 1.104,
          actual_price: 1.105,
          predicted_return: 0.001,
          realized_return: 0.001,
          predicted_score: 8,
          hit: true,
        },
      ],
    } as DecisionBacktestTrajectoryResponse;
    const forecast = {
      model_version: "decision_alpha_v1",
      generated_at: "2026-04-25T10:00:00Z",
      symbol: "EURUSD",
      horizon_days: 150,
      current_price: 1.105,
      score: 12,
      probability_up: 0.57,
      features: {},
      scenarios: [
        {
          name: "base",
          probability: 0.4,
          projected_return: 0.03,
          path: [
            { day: 0, date: "2026-04-25T09:00:00Z", price: 1.105 },
            { day: 1, date: "2026-04-26T09:00:00Z", price: 1.106 },
            { day: 2, date: "2026-04-27T09:00:00Z", price: 1.107 },
          ],
        },
      ],
    } as DecisionForecastResponse;

    const merged = projectionHistoryForecastSeries(trajectory, forecast, { historyWindowDays: 30 });

    expect(merged.historyCount).toBe(3);
    expect(merged.forecastCount).toBe(2);
    expect(merged.labels).toEqual([
      "2026-04-23T09:00:00Z",
      "2026-04-24T09:00:00Z",
      "2026-04-25T09:00:00Z",
      "2026-04-26T09:00:00Z",
      "2026-04-27T09:00:00Z",
    ]);
    expect(merged.actual).toEqual([1.101, 1.103, 1.105, null, null]);
    expect(merged.predicted).toEqual([1.1, 1.102, 1.104, 1.106, 1.107]);
    expect(merged.splitLabel).toBe("2026-04-25T09:00:00Z");
  });

  it("rebases projection path when forecast anchor is stale", () => {
    const trajectory = {
      model_version: "decision_alpha_v1",
      generated_at: "2026-04-25T09:00:00Z",
      symbol: "EURUSD",
      lookback_days: 30,
      sample_size: 2,
      hit_rate: 0.5,
      mean_abs_error: 0.0015,
      predicted_vs_actual: [
        {
          timestamp: "2026-04-24T09:00:00Z",
          predicted_price: 1.17,
          actual_price: 1.171,
          predicted_return: 0.0,
          realized_return: 0.0,
          predicted_score: 2,
          hit: null,
        },
        {
          timestamp: "2026-04-25T09:00:00Z",
          predicted_price: 1.17,
          actual_price: 1.172,
          predicted_return: 0.0,
          realized_return: 0.0,
          predicted_score: 2,
          hit: null,
        },
      ],
    } as DecisionBacktestTrajectoryResponse;
    const staleForecast = {
      model_version: "decision_alpha_v1",
      generated_at: "2026-04-25T10:00:00Z",
      symbol: "EURUSD",
      horizon_days: 150,
      current_price: 1.095,
      score: -15,
      probability_up: 0.44,
      features: {},
      scenarios: [
        {
          name: "base",
          probability: 0.4,
          projected_return: 0.01,
          path: [
            { day: 0, date: "2026-04-25T09:00:00Z", price: 1.095 },
            { day: 1, date: "2026-04-26T09:00:00Z", price: 1.096 },
            { day: 2, date: "2026-04-27T09:00:00Z", price: 1.097 },
          ],
        },
      ],
    } as DecisionForecastResponse;

    const merged = projectionHistoryForecastSeries(trajectory, staleForecast, { historyWindowDays: 30 });
    const firstFuture = merged.predicted[merged.historyCount];

    expect(firstFuture).not.toBeNull();
    expect(Number(firstFuture)).toBeGreaterThan(1.16);
    expect(Number(firstFuture)).toBeLessThan(1.19);
  });
});

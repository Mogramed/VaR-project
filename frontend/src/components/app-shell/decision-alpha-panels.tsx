"use client";

import { ChartSurface } from "@/components/charts/chart-surface";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { CHART_PALETTE, makeGroupedBarOption, makeLineOption } from "@/lib/chart-options";
import type {
  DecisionBacktestTrajectoryResponse,
  DecisionForecastResponse,
  DecisionIntelligenceResponse,
  DecisionPortfolioForecastResponse,
  DecisionReplayResponse,
} from "@/lib/api/types";
import {
  decisionAlphaFeatureRows,
  forecastChartSeries,
  portfolioPnlScenarioSeries,
  projectionHistoryForecastSeries,
  replayChartSeries,
  trajectoryChartSeries,
} from "@/lib/decision-alpha-view-model";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

function intelligenceTone(signal: string | null | undefined) {
  const normalized = String(signal ?? "").toUpperCase();
  if (normalized === "BUY") {
    return "success" as const;
  }
  if (normalized === "SELL") {
    return "warning" as const;
  }
  if (normalized === "HOLD") {
    return "neutral" as const;
  }
  return "neutral" as const;
}

function resolveTightYAxisBounds(values: number[]) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return null;
  }
  const rawMin = Math.min(...finite);
  const rawMax = Math.max(...finite);
  const span = rawMax - rawMin;
  const base = Math.max(Math.abs(rawMax), Math.abs(rawMin), 1e-9);
  const padding = span <= 1e-9 ? Math.max(base * 0.015, 0.0002) : Math.max(span * 0.15, 0.0002);
  return {
    min: rawMin - padding,
    max: rawMax + padding,
    span: Math.max(span, padding),
  };
}

function axisDecimalsFromSpan(span: number) {
  if (span < 0.005) return 5;
  if (span < 0.05) return 4;
  if (span < 0.5) return 3;
  return 2;
}

function forecastScenarioOption(forecast: DecisionForecastResponse | null | undefined) {
  const series = forecastChartSeries(forecast);
  return {
    animationDuration: 650,
    animationDurationUpdate: 320,
    tooltip: { trigger: "axis" as const },
    legend: {
      top: 2,
      textStyle: { color: "rgba(234,236,239,0.6)", fontSize: 10 },
      data: ["Bear", "Base", "Bull"],
    },
    grid: { left: 16, right: 20, top: 34, bottom: 42, containLabel: true },
    xAxis: {
      type: "category" as const,
      data: series.labels,
      axisLabel: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" as const } },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    series: [
      {
        name: "Bear",
        type: "line",
        smooth: 0.25,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.red },
        data: series.bear,
      },
      {
        name: "Base",
        type: "line",
        smooth: 0.25,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.gold },
        data: series.base,
      },
      {
        name: "Bull",
        type: "line",
        smooth: 0.25,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.green },
        data: series.bull,
      },
    ],
  };
}

function trajectoryOption(trajectory: DecisionBacktestTrajectoryResponse | null | undefined) {
  const series = trajectoryChartSeries(trajectory);
  const yBounds = resolveTightYAxisBounds([...series.predicted, ...series.actual]);
  const yDecimals = axisDecimalsFromSpan(yBounds?.span ?? 1);
  return {
    animationDuration: 650,
    animationDurationUpdate: 320,
    tooltip: { trigger: "axis" as const },
    legend: {
      top: 2,
      textStyle: { color: "rgba(234,236,239,0.6)", fontSize: 10 },
      data: ["Predicted", "Actual"],
    },
    grid: { left: 16, right: 20, top: 34, bottom: 60, containLabel: true },
    xAxis: {
      type: "category" as const,
      data: series.labels,
      axisLabel: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
    },
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: 0,
        filterMode: "none",
        zoomOnMouseWheel: "shift",
        moveOnMouseWheel: true,
      },
      {
        type: "slider",
        xAxisIndex: 0,
        height: 14,
        bottom: 8,
        borderColor: "rgba(255,255,255,0.12)",
        backgroundColor: "rgba(255,255,255,0.03)",
        fillerColor: "rgba(240,185,11,0.16)",
        handleSize: 14,
        handleStyle: {
          color: "rgba(240,185,11,0.85)",
          borderColor: "rgba(240,185,11,0.95)",
        },
      },
    ],
    yAxis: {
      type: "value" as const,
      min: yBounds?.min,
      max: yBounds?.max,
      scale: true,
      axisLabel: {
        color: "rgba(234,236,239,0.55)",
        fontSize: 10,
        formatter: (value: number) => Number(value).toFixed(yDecimals),
      },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" as const } },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    series: [
      {
        name: "Predicted",
        type: "line",
        smooth: 0.2,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.gold },
        data: series.predicted,
      },
      {
        name: "Actual",
        type: "line",
        smooth: 0.2,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.blue },
        data: series.actual,
      },
    ],
  };
}

function projectionTimelineOption({
  labels,
  actual,
  predicted,
  historyCount,
  forecastCount,
  splitLabel,
}: ReturnType<typeof projectionHistoryForecastSeries>) {
  const yBounds = resolveTightYAxisBounds([
    ...actual.filter((value): value is number => value != null),
    ...predicted.filter((value): value is number => value != null),
  ]);
  const yDecimals = axisDecimalsFromSpan(yBounds?.span ?? 1);
  const projectionStartLabel = historyCount > 0 ? labels[Math.max(historyCount - 1, 0)] : null;
  const projectionEndLabel = labels.length > 0 ? labels[labels.length - 1] : null;
  const projectionZoneActive = forecastCount > 0 && projectionStartLabel != null && projectionEndLabel != null;

  return {
    animationDuration: 650,
    animationDurationUpdate: 320,
    tooltip: { trigger: "axis" as const },
    legend: {
      top: 2,
      textStyle: { color: "rgba(234,236,239,0.6)", fontSize: 10 },
      data: ["Predicted", "Actual"],
    },
    grid: { left: 16, right: 20, top: 34, bottom: 60, containLabel: true },
    xAxis: {
      type: "category" as const,
      data: labels,
      axisLabel: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
    },
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: 0,
        filterMode: "none",
        zoomOnMouseWheel: "shift",
        moveOnMouseWheel: true,
      },
      {
        type: "slider",
        xAxisIndex: 0,
        height: 14,
        bottom: 8,
        borderColor: "rgba(255,255,255,0.12)",
        backgroundColor: "rgba(255,255,255,0.03)",
        fillerColor: "rgba(240,185,11,0.16)",
        handleSize: 14,
        handleStyle: {
          color: "rgba(240,185,11,0.85)",
          borderColor: "rgba(240,185,11,0.95)",
        },
      },
    ],
    yAxis: {
      type: "value" as const,
      min: yBounds?.min,
      max: yBounds?.max,
      scale: true,
      axisLabel: {
        color: "rgba(234,236,239,0.55)",
        fontSize: 10,
        formatter: (value: number) => Number(value).toFixed(yDecimals),
      },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" as const } },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    series: [
      {
        name: "Predicted",
        type: "line",
        smooth: 0.22,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.gold },
        data: predicted,
        markLine: splitLabel
          ? {
              silent: true,
              symbol: "none",
              data: [{ xAxis: splitLabel }],
              lineStyle: { color: "rgba(255,255,255,0.2)", type: "dashed", width: 1 },
              label: { show: false },
            }
          : undefined,
        markArea: projectionZoneActive
          ? {
              silent: true,
              itemStyle: { color: "rgba(240,185,11,0.06)" },
              data: [[{ xAxis: projectionStartLabel }, { xAxis: projectionEndLabel }]],
            }
          : undefined,
      },
      {
        name: "Actual",
        type: "line",
        smooth: 0.22,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.blue },
        data: actual,
      },
    ],
  };
}

function portfolioPnlOption(portfolioForecast: DecisionPortfolioForecastResponse | null | undefined) {
  const series = portfolioPnlScenarioSeries(portfolioForecast);
  return {
    animationDuration: 650,
    animationDurationUpdate: 320,
    tooltip: { trigger: "axis" as const },
    legend: {
      top: 2,
      textStyle: { color: "rgba(234,236,239,0.6)", fontSize: 10 },
      data: ["Bear", "Base", "Bull"],
    },
    grid: { left: 16, right: 20, top: 34, bottom: 42, containLabel: true },
    xAxis: {
      type: "category" as const,
      data: series.labels,
      axisLabel: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)", type: "dashed" as const } },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    series: [
      {
        name: "Bear",
        type: "line",
        smooth: 0.25,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.red },
        data: series.bear,
      },
      {
        name: "Base",
        type: "line",
        smooth: 0.25,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.gold },
        data: series.base,
      },
      {
        name: "Bull",
        type: "line",
        smooth: 0.25,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.green },
        data: series.bull,
      },
    ],
  };
}

export function DecisionAlphaIntelligencePanel({
  intelligence,
  title = "Decision Alpha v1",
}: {
  intelligence: DecisionIntelligenceResponse | null | undefined;
  title?: string;
}) {
  if (!intelligence) {
    return (
      <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
          {title}
        </h4>
        <p className="mt-1.5 text-xs text-[var(--color-text-muted)]">
          Decision intelligence not available yet.
        </p>
      </div>
    );
  }

  const rows = decisionAlphaFeatureRows(intelligence);
  return (
    <div className="space-y-3">
      <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
        <div className="flex items-center justify-between gap-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            {title}
          </h4>
          <StatusBadge label={String(intelligence.signal)} tone={intelligenceTone(intelligence.signal)} />
        </div>
        <div className="mt-2 grid gap-2 sm:grid-cols-4">
          <MetricBlock label="Score" value={Number(intelligence.score).toFixed(1)} tone="accent" />
          <MetricBlock label="Confidence" value={formatPercent(intelligence.confidence, 0)} />
          <MetricBlock label="Size multiplier" value={formatPercent(intelligence.size_multiplier, 0)} tone="success" />
          <MetricBlock
            label="Guardrail"
            value={intelligence.guardrail_applied ? "Applied" : "None"}
            tone={intelligence.guardrail_applied ? "warning" : "neutral"}
          />
        </div>
        <div className="mt-2 text-[11px] text-[var(--color-text-muted)]">
          Model {intelligence.model_version}
        </div>
        {intelligence.model_runtime ? (
          <div className="mt-1 text-[11px] text-[var(--color-text-muted)]">
            Runtime: {intelligence.model_runtime.trained_model ? "trained" : "fallback"} | samples{" "}
            {Number(intelligence.model_runtime.sample_count ?? 0)}
          </div>
        ) : null}
        {intelligence.top_drivers?.length ? (
          <div className="mt-2 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-2.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Top drivers
            </div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {intelligence.top_drivers.map((driver) => (
                <span
                  key={driver}
                  className="rounded-full border border-[var(--color-border)] px-2 py-0.5 text-[10px] text-[var(--color-text-soft)]"
                >
                  {driver}
                </span>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
          Features and contributions
        </div>
        {rows.length === 0 ? (
          <p className="text-xs text-[var(--color-text-muted)]">No features available.</p>
        ) : (
          <div className="space-y-1.5">
            {rows.map((row) => (
              <div key={row.key} className="grid grid-cols-[minmax(0,1fr)_90px_90px] gap-2 text-[11px]">
                <span className="truncate text-[var(--color-text-soft)]">{row.label}</span>
                <span className="mono text-right text-[var(--color-text)]">
                  {row.value == null ? "n/a" : row.value.toFixed(4)}
                </span>
                <span className="mono text-right text-[var(--color-text-muted)]">
                  {row.contribution == null ? "n/a" : row.contribution.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function DecisionAlphaReplayPanel({
  replay,
}: {
  replay: DecisionReplayResponse | null | undefined;
}) {
  const points = replay?.predicted_vs_realized ?? [];
  const series = replayChartSeries(replay);
  return (
    <div className="space-y-3">
      <section className="grid gap-3 sm:grid-cols-4">
        <MetricBlock label="Replay sample" value={String(replay?.sample_size ?? 0)} />
        <MetricBlock label="Hit rate" value={formatPercent(replay?.hit_rate ?? 0, 1)} tone="accent" />
        <MetricBlock label="Cum PnL" value={formatCurrency(replay?.cum_pnl ?? 0, 2)} tone={(replay?.cum_pnl ?? 0) >= 0 ? "success" : "warning"} />
        <MetricBlock
          label="Runtime"
          value={replay?.model_runtime?.trained_model ? "Trained" : "Fallback"}
          tone={replay?.model_runtime?.trained_model ? "success" : "warning"}
        />
      </section>
      <ChartSurface
        option={makeGroupedBarOption(
          series.labels,
          [
            { name: "Predicted score", data: series.predicted, color: CHART_PALETTE.gold },
            { name: "Realized PnL", data: series.realized, color: CHART_PALETTE.blue },
          ],
          { mode: "comparison" },
        )}
        mode="comparison"
        dataCount={series.labels.length}
        title="Replay: predicted vs realized"
        meta={`Updated ${formatTimestamp(replay?.generated_at ?? null)}`}
        emptyState={<p className="text-xs text-[var(--color-text-muted)]">No replay data yet.</p>}
      />
      <ChartSurface
        option={makeLineOption(
          series.labels.map((label, index) => ({ label, value: series.cumulative[index] ?? 0 })),
          CHART_PALETTE.green,
          { mode: "standard" },
        )}
        mode="standard"
        dataCount={series.labels.length}
        title="Replay cumulative PnL"
        emptyState={<p className="text-xs text-[var(--color-text-muted)]">No cumulative path yet.</p>}
      />
      {points.length > 0 ? (
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Latest replay point
          </div>
          <div className="text-xs text-[var(--color-text-soft)]">
            {points[points.length - 1].symbol} | score {points[points.length - 1].predicted_score.toFixed(1)} |
            pnl {formatCurrency(points[points.length - 1].realized_pnl, 2)}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function DecisionAlphaForecastPanel({
  forecast,
}: {
  forecast: DecisionForecastResponse | null | undefined;
}) {
  const scenarios = forecast?.scenarios ?? [];
  return (
    <div className="space-y-3">
      <section className="grid gap-3 sm:grid-cols-4">
        <MetricBlock label="Symbol" value={forecast?.symbol ?? "n/a"} />
        <MetricBlock label="Horizon" value={`${forecast?.horizon_days ?? 0}d`} />
        <MetricBlock label="Prob up" value={formatPercent(forecast?.probability_up ?? 0, 1)} tone="accent" />
        <MetricBlock
          label="Runtime"
          value={forecast?.model_runtime?.trained_model ? "Trained" : "Fallback"}
          tone={forecast?.model_runtime?.trained_model ? "success" : "warning"}
        />
      </section>
      <ChartSurface
        option={forecastScenarioOption(forecast)}
        mode="comparison"
        dataCount={Math.max(...scenarios.map((scenario) => scenario.path?.length ?? 0), 0)}
        title="Forecast scenarios"
        meta={`Generated ${formatTimestamp(forecast?.generated_at ?? null)}`}
        emptyState={<p className="text-xs text-[var(--color-text-muted)]">No forecast scenarios yet.</p>}
      />
      <div className="grid gap-2 sm:grid-cols-3">
        {scenarios.map((scenario) => (
          <div key={scenario.name} className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-2.5">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              {scenario.name}
            </div>
            <div className="mt-1 text-[12px] font-semibold text-[var(--color-text)]">
              {formatPercent(scenario.probability, 1)}
            </div>
            <div className="text-[11px] text-[var(--color-text-muted)]">
              return {formatPercent(scenario.projected_return, 2)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function DecisionAlphaTrajectoryPanel({
  trajectory,
}: {
  trajectory: DecisionBacktestTrajectoryResponse | null | undefined;
}) {
  const points = trajectory?.predicted_vs_actual ?? [];
  const comparableCount = points.filter((point) => point.hit != null).length;
  return (
    <div className="space-y-3">
      <section className="grid gap-3 sm:grid-cols-4">
        <MetricBlock label="Sample" value={String(trajectory?.sample_size ?? 0)} />
        <MetricBlock label="Hit rate" value={formatPercent(trajectory?.hit_rate ?? 0, 1)} tone="accent" />
        <MetricBlock label="MAE" value={Number(trajectory?.mean_abs_error ?? 0).toFixed(5)} />
        <MetricBlock label="Comparables" value={String(comparableCount)} />
      </section>
      <ChartSurface
        option={trajectoryOption(trajectory)}
        mode="comparison"
        dataCount={points.length}
        title="Past trajectory: predicted vs actual"
        meta={`Updated ${formatTimestamp(trajectory?.generated_at ?? null)}`}
        emptyState={<p className="text-xs text-[var(--color-text-muted)]">No trajectory history yet.</p>}
      />
    </div>
  );
}

export function DecisionAlphaProjectionPanel({
  trajectory,
  forecast,
  historyWindowDays = 30,
}: {
  trajectory: DecisionBacktestTrajectoryResponse | null | undefined;
  forecast: DecisionForecastResponse | null | undefined;
  historyWindowDays?: number;
}) {
  const series = projectionHistoryForecastSeries(trajectory, forecast, { historyWindowDays });
  const lastActual = [...series.actual].reverse().find((value): value is number => value != null) ?? null;
  return (
    <div className="space-y-3">
      <section className="grid gap-3 sm:grid-cols-4">
        <MetricBlock label="History window" value={`${historyWindowDays}d`} />
        <MetricBlock label="Forecast horizon" value={`${forecast?.horizon_days ?? 0}d`} />
        <MetricBlock label="History/Future points" value={`${series.historyCount} / ${series.forecastCount}`} />
        <MetricBlock label="Last actual" value={lastActual == null ? "n/a" : Number(lastActual).toFixed(5)} tone="accent" />
      </section>
      <ChartSurface
        option={projectionTimelineOption(series)}
        mode="comparison"
        dataCount={series.labels.length}
        title="1M history + 5M model projection"
        meta={`Updated ${formatTimestamp(forecast?.generated_at ?? trajectory?.generated_at ?? null)}`}
        emptyState={<p className="text-xs text-[var(--color-text-muted)]">No projection data yet.</p>}
      />
    </div>
  );
}

export function DecisionAlphaPortfolioForecastPanel({
  portfolioForecast,
}: {
  portfolioForecast: DecisionPortfolioForecastResponse | null | undefined;
}) {
  const scenarios = portfolioForecast?.pnl_scenarios ?? [];
  const symbols = portfolioForecast?.symbols ?? [];
  return (
    <div className="space-y-3">
      <section className="grid gap-3 sm:grid-cols-4">
        <MetricBlock label="Symbols" value={String(portfolioForecast?.symbol_count ?? symbols.length)} />
        <MetricBlock label="Notional" value={formatCurrency(portfolioForecast?.current_notional_eur ?? 0, 0)} />
        <MetricBlock label="Horizon" value={`${portfolioForecast?.horizon_days ?? 0}d`} />
        <MetricBlock
          label="Runtime"
          value={portfolioForecast?.model_runtime?.trained_model ? "Trained" : "Fallback"}
          tone={portfolioForecast?.model_runtime?.trained_model ? "success" : "warning"}
        />
      </section>
      <ChartSurface
        option={portfolioPnlOption(portfolioForecast)}
        mode="comparison"
        dataCount={Math.max(...scenarios.map((scenario) => scenario.path?.length ?? 0), 0)}
        title="Portfolio PnL scenarios"
        meta={`Generated ${formatTimestamp(portfolioForecast?.generated_at ?? null)}`}
        emptyState={<p className="text-xs text-[var(--color-text-muted)]">No portfolio scenarios yet.</p>}
      />
      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
        {symbols.map((item) => {
          const baseScenario = item.forecast.scenarios?.find((scenario) => scenario.name === "base");
          return (
            <div key={item.symbol} className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-2.5">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-[var(--color-text)]">{item.symbol}</span>
                <span className="mono text-[10px] text-[var(--color-text-muted)]">{formatPercent(item.weight, 1)}</span>
              </div>
              <div className="mt-1 text-[11px] text-[var(--color-text-muted)]">
                Exposure {formatCurrency(item.exposure_eur, 0)}
              </div>
              <div className="text-[11px] text-[var(--color-text-muted)]">
                Base return {formatPercent(baseScenario?.projected_return ?? 0, 2)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

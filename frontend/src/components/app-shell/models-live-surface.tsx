"use client";

import { startTransition, useEffect, useState } from "react";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { ModelRankingTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { BacktestFrameResponse, MT5LiveStateResponse, ModelComparisonResponse, ValidationRunSummary } from "@/lib/api/types";
import { CHART_PALETTE, makeBacktestOption, makeBarOption } from "@/lib/chart-options";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { preferredHeadlineRisk } from "@/lib/risk-surface";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";
import { buildBacktestSeries, buildModelScoreSeries } from "@/lib/view-models";

export function ModelsLiveSurface({
  portfolioSlug, initialLiveState, initialComparison, initialValidation, initialFrame,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialComparison: ModelComparisonResponse | null;
  initialValidation: ValidationRunSummary | null;
  initialFrame: BacktestFrameResponse | null;
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [comparison, setComparison] = useState(initialComparison);
  const [validation, setValidation] = useState(initialValidation);
  const [frame, setFrame] = useState(initialFrame);

  useEffect(() => {
    let c = false;
    const t = window.setInterval(() => {
      Promise.all([
        api.latestModelComparison(portfolioSlug).catch(() => null),
        api.latestValidation(portfolioSlug).catch(() => null),
        api.latestBacktestFrame(portfolioSlug, 240).catch(() => null),
      ]).then(([nc, nv, nf]) => {
        if (!c) startTransition(() => { setComparison(nc); setValidation(nv); setFrame(nf); });
      });
    }, 30000);
    return () => { c = true; window.clearInterval(t); };
  }, [portfolioSlug]);

  const ranking = comparison?.ranking ?? [];
  const backtestSeries = frame ? buildBacktestSeries(frame) : [];
  const scoreSeries = comparison ? buildModelScoreSeries(comparison) : [];
  const selectedModel = liveState?.risk_summary?.reference_model ?? comparison?.champion_model ?? validation?.best_model ?? "hist";
  const liveRisk95 = preferredHeadlineRisk(liveState?.risk_summary?.headline_risk, ["live_1d_95"]);
  const liveRisk99 = preferredHeadlineRisk(liveState?.risk_summary?.headline_risk, ["live_1d_99"]);
  const liveCapital = liveState?.capital_usage ?? null;
  const validationSurface = (comparison?.validation_surface ?? null) as { points?: unknown[] } | null;
  const validationSurfacePoints = Array.isArray(validationSurface?.points)
    ? validationSurface.points
    : [];

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Models" title="Champion, challenger and statistical credibility"
        aside={<>
          {comparison?.champion_model ? <StatusBadge label={comparison.champion_model.toUpperCase()} tone="accent" /> : null}
          <StatusBadge label={transport} tone={transport === "stream" ? "success" : "warning"} />
        </>}
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricBlock label="Champion" value={(comparison?.champion_model ?? "n/a").toUpperCase()} />
        <MetricBlock label="Reporting champion" value={(comparison?.champion_model_reporting ?? comparison?.challenger_model ?? "n/a").toUpperCase()} />
        <MetricBlock label="Exception rate" value={validation ? formatPercent(validation.expected_rate) : "n/a"} tone="warning" />
        <MetricBlock label="Score gap" value={comparison?.score_gap != null ? comparison.score_gap.toFixed(1) : "n/a"} tone="accent" />
        <MetricBlock
          label="Live VaR / ES"
          value={liveRisk95 == null ? "n/a" : formatCurrency(liveRisk95.var)}
          hint={liveRisk99 ? `99% ES ${formatCurrency(liveRisk99.es)}` : liveState?.risk_summary?.latest_observation ? formatTimestamp(liveState.risk_summary.latest_observation) : undefined}
          tone="warning"
        />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_300px]">
        <ChartSurface
          option={makeBacktestOption(backtestSeries)}
          mode="trace" dataCount={backtestSeries.length}
          title="Backtest trace" meta={backtestSeries.length ? `${backtestSeries.length} rows` : undefined}
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No backtest frame available.</p>}
        />
        <div className="space-y-3">
          <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Validation</div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Best model</span>
                <span className="mono font-semibold text-[var(--color-text)]">{(validation?.best_model ?? selectedModel).toUpperCase()}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Reporting champion</span>
                <span className="mono text-[var(--color-text)]">{(comparison?.champion_model_reporting ?? "n/a").toUpperCase()}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Alpha</span>
                <span className="mono text-[var(--color-text)]">{validation ? formatPercent(validation.alpha, 0) : "n/a"}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Headroom</span>
                <span className="mono text-[var(--color-green)]">{liveCapital ? formatPercent(liveCapital.headroom_ratio ?? 0, 0) : "n/a"}</span>
              </div>
            </div>
          </div>
          {ranking.slice(0, 3).map((row) => (
            <div key={row.model} className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--color-text)]">{row.model.toUpperCase()}</span>
                <span className="mono text-sm text-[var(--color-accent)]">{row.score.toFixed(1)}</span>
              </div>
              <div className="mt-1.5 flex gap-4 text-[11px] text-[var(--color-text-muted)]">
                <span>Rate {formatPercent(row.actual_rate)}</span>
                <span>VaR {formatCurrency(row.current_var)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(280px,0.85fr)_minmax(0,1.15fr)]">
        <ChartSurface
          option={makeBarOption(scoreSeries, { color: CHART_PALETTE.gold, negativeColor: CHART_PALETTE.green, mode: ranking.length <= 5 ? "sparse" : "standard" })}
          mode={ranking.length <= 5 ? "sparse" : "standard"} dataCount={ranking.length}
          title="Model scores"
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No ranking available.</p>}
        />
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Ranking table</h4>
          <ModelRankingTable rows={ranking} />
          {validationSurfacePoints.length > 0 ? (
            <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3 text-[11px] text-[var(--color-text-soft)]">
              Validation surface loaded with {validationSurfacePoints.length} model / alpha / horizon points.
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

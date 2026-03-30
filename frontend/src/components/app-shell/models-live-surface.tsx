"use client";

import { startTransition, useEffect, useState } from "react";

import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { ModelRankingTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type {
  BacktestFrameResponse,
  MT5LiveStateResponse,
  ModelComparisonResponse,
  ValidationRunSummary,
} from "@/lib/api/types";
import { makeBacktestOption, makeBarOption } from "@/lib/chart-options";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";
import { buildBacktestSeries, buildModelScoreSeries } from "@/lib/view-models";

export function ModelsLiveSurface({
  portfolioSlug,
  initialLiveState,
  initialComparison,
  initialValidation,
  initialFrame,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialComparison: ModelComparisonResponse | null;
  initialValidation: ValidationRunSummary | null;
  initialFrame: BacktestFrameResponse | null;
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [comparison, setComparison] = useState<ModelComparisonResponse | null>(initialComparison);
  const [validation, setValidation] = useState<ValidationRunSummary | null>(initialValidation);
  const [frame, setFrame] = useState<BacktestFrameResponse | null>(initialFrame);

  useEffect(() => {
    let cancelled = false;

    const refreshModels = async () => {
      try {
        const [nextComparison, nextValidation, nextFrame] = await Promise.all([
          api.latestModelComparison(portfolioSlug).catch(() => null),
          api.latestValidation(portfolioSlug).catch(() => null),
          api.latestBacktestFrame(portfolioSlug, 240).catch(() => null),
        ]);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setComparison(nextComparison);
          setValidation(nextValidation);
          setFrame(nextFrame);
        });
      } catch {
        // Keep the current model monitor state on transient failures.
      }
    };

    const timer = window.setInterval(() => {
      void refreshModels();
    }, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [portfolioSlug]);

  const ranking = comparison?.ranking ?? [];
  const backtestSeries = frame ? buildBacktestSeries(frame) : [];
  const scoreSeries = comparison ? buildModelScoreSeries(comparison) : [];
  const selectedModel =
    liveState?.risk_summary?.reference_model ??
    comparison?.champion_model ??
    validation?.best_model ??
    "hist";
  const liveVarValue =
    liveState?.risk_summary?.var?.[selectedModel] ??
    Object.values(liveState?.risk_summary?.var ?? {})[0] ??
    null;
  const liveCapital = liveState?.capital_usage ?? null;

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Model Monitor"
        title="Champion, challenger and statistical credibility."
        description="The model surface now stays tied to the MT5 bridge: historical ranking and backtest evidence remain visible, while live VaR and capital posture expose how the current portfolio is actually being read."
        aside={
          <div className="flex flex-wrap items-center gap-2">
            {comparison?.champion_model ? (
              <StatusBadge label={comparison.champion_model.toUpperCase()} tone="accent" />
            ) : null}
            <StatusBadge
              label={transport === "stream" ? "stream" : transport === "polling" ? "polling" : "connecting"}
              tone={transport === "stream" ? "success" : transport === "polling" ? "warning" : "neutral"}
            />
          </div>
        }
      />

      <section className="grid gap-4 xl:grid-cols-5">
        <MetricBlock
          label="Champion"
          value={(comparison?.champion_model ?? "n/a").toUpperCase()}
          hint="Current best performer"
        />
        <MetricBlock
          label="Challenger"
          value={(comparison?.challenger_model ?? "n/a").toUpperCase()}
          hint="Second-ranked model"
        />
        <MetricBlock
          label="Expected exceptions"
          value={validation ? formatPercent(validation.expected_rate) : "n/a"}
          hint="Target hit-rate"
          tone="warning"
        />
        <MetricBlock
          label="Score gap"
          value={comparison?.score_gap != null ? comparison.score_gap.toFixed(1) : "n/a"}
          hint="Champion vs challenger"
          tone="accent"
        />
        <MetricBlock
          label="Live VaR"
          value={liveVarValue == null ? "n/a" : formatCurrency(liveVarValue)}
          hint={
            liveState?.risk_summary?.latest_observation
              ? `${selectedModel.toUpperCase()} at ${formatTimestamp(liveState.risk_summary.latest_observation)}`
              : "No live risk summary"
          }
          tone="warning"
        />
      </section>

      <LiveOperatorAlerts
        alerts={liveState?.operator_alerts ?? []}
        title="Model watchlist"
        copy="Bridge health, live risk pressure and reconciliation incidents remain visible here while you read champion and challenger diagnostics."
      />

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_340px]">
        <ChartSurface
          option={makeBacktestOption(backtestSeries)}
          mode="trace"
          dataCount={backtestSeries.length}
          eyebrow="Backtest trace"
          title="Portfolio PnL versus selected VaR series"
          description="Backtest evidence stays central, while the live panel on the side shows how the current MT5 posture maps to the active model."
          meta={backtestSeries.length ? `${backtestSeries.length} rows` : "No backtest frame"}
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No backtest frame available yet.
            </div>
          }
        />

        <div className="space-y-6">
          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Validation pulse
            </div>
            <div className="mt-5 grid gap-4">
              <MetricBlock
                label="Best model"
                value={(validation?.best_model ?? selectedModel).toUpperCase()}
                hint="Latest validation or live reference"
                className="bg-transparent"
              />
              <MetricBlock
                label="Alpha"
                value={validation ? formatPercent(validation.alpha, 0) : "n/a"}
                hint="Confidence level"
                tone="accent"
                className="bg-transparent"
              />
              <MetricBlock
                label="Capital headroom"
                value={liveCapital ? formatPercent(liveCapital.headroom_ratio ?? 0, 0) : "n/a"}
                hint={
                  liveCapital
                    ? `Remaining ${formatCurrency(liveCapital.total_capital_remaining_eur)}`
                    : "No live capital posture"
                }
                tone="success"
                className="bg-transparent"
              />
            </div>
            {ranking.length > 0 ? (
              <div className="mt-5 space-y-3">
                {ranking.slice(0, 3).map((row) => (
                  <div
                    key={row.model}
                    className="rounded-[1.3rem] border border-white/8 bg-black/18 p-4"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-base font-semibold text-white">
                        {row.model.toUpperCase()}
                      </div>
                      <div className="mono text-sm text-[var(--color-accent)]">
                        {row.score.toFixed(1)}
                      </div>
                    </div>
                    <div className="mt-3 flex items-center justify-between text-sm text-[var(--color-text-soft)]">
                      <span>Actual rate</span>
                      <span>{formatPercent(row.actual_rate)}</span>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-sm text-[var(--color-text-soft)]">
                      <span>Current VaR</span>
                      <span>{formatCurrency(row.current_var)}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(320px,0.88fr)_minmax(0,1.12fr)]">
        <ChartSurface
          option={makeBarOption(scoreSeries, {
            color: "#d89b49",
            negativeColor: "#5fd4a6",
            mode: ranking.length <= 5 ? "sparse" : "standard",
          })}
          mode={ranking.length <= 5 ? "sparse" : "standard"}
          dataCount={ranking.length}
          eyebrow="Ranking surface"
          title="Model score spread"
          description="The ranking surface stays historical and statistical, while the page itself remains anchored to the current live portfolio read."
          insight={
            ranking.length > 0 ? (
              <div className="grid gap-4">
                <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
                  <div className="text-sm text-[var(--color-text-soft)]">Champion edge</div>
                  <div className="mt-2 text-2xl font-semibold text-white">
                    {comparison?.score_gap != null ? comparison.score_gap.toFixed(1) : "n/a"}
                  </div>
                  <div className="mt-2 text-sm leading-6 text-[var(--color-text-soft)]">
                    Gap between champion and challenger under the latest validation snapshot.
                  </div>
                </div>
                <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
                  <div className="text-sm text-[var(--color-text-soft)]">Bridge posture</div>
                  <div className="mt-2 text-2xl font-semibold text-white">
                    {(liveState?.status ?? "pending").toUpperCase()}
                  </div>
                  <div className="mt-2 text-sm leading-6 text-[var(--color-text-soft)]">
                    {liveState?.generated_at
                      ? `Updated ${formatTimestamp(liveState.generated_at)}.`
                      : "Waiting for MT5 live state."}
                  </div>
                </div>
              </div>
            ) : undefined
          }
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No ranking snapshot is available yet.
            </div>
          }
        />

        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Ranking table
          </div>
          <ModelRankingTable rows={ranking} />
        </div>
      </section>
    </div>
  );
}

"use client";

import { startTransition, useEffect, useState } from "react";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { DecisionHistoryTable } from "@/components/data/risk-tables";
import { TradeDecisionPanel } from "@/components/forms/trade-decision-panel";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { MT5LiveStateResponse, RiskDecisionResponse } from "@/lib/api/types";
import { CHART_PALETTE, makeBarOption, makeGroupedBarOption } from "@/lib/chart-options";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { averageDecisionFillRatio, buildDecisionDeltaComparison, buildDecisionImpactSeries, buildDecisionVerdictCounts } from "@/lib/view-models";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

export function DecisionsLiveSurface({
  portfolioSlug, initialLiveState, initialDecisions,
}: { portfolioSlug: string; initialLiveState: MT5LiveStateResponse | null; initialDecisions: RiskDecisionResponse[] }) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [decisions, setDecisions] = useState(initialDecisions);

  useEffect(() => {
    let c = false;
    api.recentDecisions(portfolioSlug, 12).then((n) => { if (!c) startTransition(() => setDecisions(n)); }).catch(() => {});
    return () => { c = true; };
  }, [portfolioSlug, liveState?.sequence]);

  useEffect(() => {
    let c = false;
    const t = window.setInterval(() => {
      api.recentDecisions(portfolioSlug, 12).then((n) => { if (!c) startTransition(() => setDecisions(n)); }).catch(() => {});
    }, 15000);
    return () => { c = true; window.clearInterval(t); };
  }, [portfolioSlug]);

  const verdicts = buildDecisionVerdictCounts(decisions);
  const fillRatio = averageDecisionFillRatio(decisions);
  const sizeComparison = buildDecisionDeltaComparison(decisions);
  const impactSeries = buildDecisionImpactSeries(decisions);
  const latest = decisions[0] ?? null;
  const liveCapital = liveState?.capital_usage ?? null;

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Decisions" title="Accept, reduce or reject before execution"
        aside={<>
          <StatusBadge label={portfolioSlug} tone="accent" />
          <StatusBadge label={transport} tone={transport === "stream" ? "success" : "warning"} />
        </>}
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricBlock label="Accepted" value={String(verdicts.ACCEPT ?? 0)} tone="success" />
        <MetricBlock label="Reduced" value={String(verdicts.REDUCE ?? 0)} tone="warning" />
        <MetricBlock label="Rejected" value={String(verdicts.REJECT ?? 0)} tone="danger" />
        <MetricBlock label="Avg fill" value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)} />
        <MetricBlock label="Headroom" value={liveCapital ? formatPercent(liveCapital.headroom_ratio ?? 0, 0) : "n/a"} tone="accent" />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />
      <TradeDecisionPanel portfolioSlug={portfolioSlug} onEvaluated={(r) => startTransition(() => setDecisions((c) => [r, ...c].slice(0, 12)))} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_280px]">
        <ChartSurface
          option={makeGroupedBarOption(sizeComparison.labels, [
            { name: "Requested", data: sizeComparison.requested, color: CHART_PALETTE.gold },
            { name: "Approved", data: sizeComparison.approved, color: CHART_PALETTE.green },
          ], { mode: "comparison" })}
          mode="comparison" dataCount={sizeComparison.labels.length}
          title="Requested vs approved" meta={`${decisions.length} decisions`}
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No decision history.</p>}
        />
        <div className="space-y-3">
          {latest ? (
            <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Latest verdict</div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-[var(--color-text)]">{latest.symbol}</span>
                <StatusBadge label={latest.decision} />
              </div>
              <div className="mt-2 text-xs text-[var(--color-text-muted)]">
                Approved {formatCurrency(latest.approved_exposure_change)} · {formatTimestamp(latest.created_at ?? latest.time_utc)}
              </div>
            </div>
          ) : null}
          <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Live posture</div>
            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between"><span className="text-[var(--color-text-muted)]">Manual events</span><span className="mono text-[var(--color-text)]">{liveState?.reconciliation?.manual_event_count ?? 0}</span></div>
              <div className="flex justify-between"><span className="text-[var(--color-text-muted)]">Unmatched</span><span className="mono text-[var(--color-text)]">{liveState?.reconciliation?.unmatched_execution_count ?? 0}</span></div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(280px,0.78fr)_minmax(0,1.22fr)]">
        <ChartSurface
          option={makeBarOption(impactSeries, { color: CHART_PALETTE.gold, negativeColor: CHART_PALETTE.green, mode: impactSeries.length <= 5 ? "sparse" : "standard" })}
          mode={impactSeries.length <= 5 ? "sparse" : "standard"} dataCount={impactSeries.length}
          title="Post-trade VaR delta"
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No impact data.</p>}
        />
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Decision history</h4>
          <DecisionHistoryTable rows={decisions} />
        </div>
      </div>
    </div>
  );
}

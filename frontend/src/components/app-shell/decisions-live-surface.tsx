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
import { makeBarOption, makeGroupedBarOption } from "@/lib/chart-options";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import {
  averageDecisionFillRatio,
  buildDecisionDeltaComparison,
  buildDecisionImpactSeries,
  buildDecisionVerdictCounts,
} from "@/lib/view-models";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

export function DecisionsLiveSurface({
  portfolioSlug,
  initialLiveState,
  initialDecisions,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialDecisions: RiskDecisionResponse[];
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [decisions, setDecisions] = useState<RiskDecisionResponse[]>(initialDecisions);

  useEffect(() => {
    let cancelled = false;

    const refreshDecisions = async () => {
      try {
        const nextDecisions = await api.recentDecisions(portfolioSlug, 12);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setDecisions(nextDecisions);
        });
      } catch {
        // Keep the current decision history on transient failures.
      }
    };

    void refreshDecisions();
    return () => {
      cancelled = true;
    };
  }, [portfolioSlug, liveState?.sequence]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const nextDecisions = await api.recentDecisions(portfolioSlug, 12);
          if (cancelled) {
            return;
          }
          startTransition(() => {
            setDecisions(nextDecisions);
          });
        } catch {
          // Keep the current decision history on transient failures.
        }
      })();
    }, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [portfolioSlug]);

  const verdicts = buildDecisionVerdictCounts(decisions);
  const fillRatio = averageDecisionFillRatio(decisions);
  const sizeComparison = buildDecisionDeltaComparison(decisions);
  const impactSeries = buildDecisionImpactSeries(decisions);
  const latestDecision = decisions[0] ?? null;
  const liveRisk = liveState?.risk_summary ?? null;
  const liveCapital = liveState?.capital_usage ?? null;
  const reconciliation = liveState?.reconciliation ?? null;
  const liveVarValue = liveRisk
    ? liveRisk.var?.[liveRisk.reference_model] ??
      Object.values(liveRisk.var ?? {})[0] ??
      0
    : null;

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Decision Layer"
        title="Accept, reduce or reject before anything reaches execution."
        description="Decisioning now stays attached to the MT5 bridge: live risk posture, capital headroom and reconciliation drift remain visible while recent advisory outcomes refresh from the desk history."
        aside={
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge label={portfolioSlug} tone="accent" />
            <StatusBadge
              label={transport === "stream" ? "stream" : transport === "polling" ? "polling" : "connecting"}
              tone={transport === "stream" ? "success" : transport === "polling" ? "warning" : "neutral"}
            />
          </div>
        }
      />

      <section className="grid gap-4 xl:grid-cols-5">
        <MetricBlock
          label="Accepted"
          value={String(verdicts.ACCEPT ?? 0)}
          hint="Recent approvals"
          tone="success"
        />
        <MetricBlock
          label="Reduced"
          value={String(verdicts.REDUCE ?? 0)}
          hint="Size was clipped"
          tone="warning"
        />
        <MetricBlock
          label="Rejected"
          value={String(verdicts.REJECT ?? 0)}
          hint="No admissible fill"
          tone="danger"
        />
        <MetricBlock
          label="Average fill"
          value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)}
          hint="Approved vs requested"
        />
        <MetricBlock
          label="Live headroom"
          value={liveCapital ? formatPercent(liveCapital.headroom_ratio ?? 0, 0) : "n/a"}
          hint={
            liveRisk?.latest_observation
              ? `Sample ${formatTimestamp(liveRisk.latest_observation)}`
              : "Waiting for live posture"
          }
          tone="accent"
        />
      </section>

      <LiveOperatorAlerts
        alerts={liveState?.operator_alerts ?? []}
        title="Decision watchlist"
        copy="Live MT5 drift, manual activity and budget pressure stay visible here while you evaluate or revisit trade proposals."
      />

      <TradeDecisionPanel
        portfolioSlug={portfolioSlug}
        onEvaluated={(result) => {
          startTransition(() => {
            setDecisions((current) => [result, ...current].slice(0, 12));
          });
        }}
      />

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.08fr)_340px]">
        <ChartSurface
          option={makeGroupedBarOption(
            sizeComparison.labels,
            [
              {
                name: "Requested",
                data: sizeComparison.requested,
                color: "#d89b49",
              },
              {
                name: "Approved",
                data: sizeComparison.approved,
                color: "#5fd4a6",
              },
            ],
            { mode: "comparison" },
          )}
          mode="comparison"
          dataCount={sizeComparison.labels.length}
          eyebrow="Sizing posture"
          title="Requested versus approved size"
          description="Sizing friction remains visible immediately, while the live posture on the side reflects what the MT5 bridge sees right now."
          meta={`${decisions.length} recent decisions`}
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No decision history is available yet.
            </div>
          }
        />

        <div className="space-y-6">
          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Latest verdict
            </div>
            {latestDecision ? (
              <div className="mt-5 space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-lg font-semibold text-white">
                      {latestDecision.symbol}
                    </div>
                    <div className="mt-1 text-sm text-[var(--color-text-soft)]">
                      {formatTimestamp(latestDecision.created_at ?? latestDecision.time_utc)}
                    </div>
                  </div>
                  <StatusBadge label={latestDecision.decision} />
                </div>
                <MetricBlock
                  label="Approved exposure"
                  value={formatCurrency(latestDecision.approved_exposure_change)}
                  hint={`Requested ${formatCurrency(latestDecision.requested_exposure_change)}`}
                  className="bg-transparent"
                />
                <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4 text-sm leading-7 text-[var(--color-text-soft)]">
                  {latestDecision.reasons[0] ?? "No persisted rationale available."}
                </div>
              </div>
            ) : (
              <div className="mt-5 text-sm text-[var(--color-text-muted)]">
                No decision has been persisted yet.
              </div>
            )}
          </div>

          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Live posture
            </div>
            <div className="mt-5 grid gap-4">
              <MetricBlock
                label="Current VaR"
                value={liveVarValue == null ? "n/a" : formatCurrency(liveVarValue)}
                hint={liveRisk?.reference_model?.toUpperCase() ?? "No live model"}
                tone="warning"
                className="bg-transparent"
              />
              <MetricBlock
                label="Manual MT5 events"
                value={String(reconciliation?.manual_event_count ?? 0)}
                hint="Detected outside desk execution"
                className="bg-transparent"
              />
              <MetricBlock
                label="Unmatched executions"
                value={String(reconciliation?.unmatched_execution_count ?? 0)}
                hint="Desk vs broker drift"
                tone={(reconciliation?.unmatched_execution_count ?? 0) > 0 ? "warning" : "success"}
                className="bg-transparent"
              />
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(320px,0.78fr)_minmax(0,1.22fr)]">
        <ChartSurface
          option={makeBarOption(impactSeries, {
            color: "#d89b49",
            negativeColor: "#5fd4a6",
            mode: impactSeries.length <= 5 ? "sparse" : "standard",
          })}
          mode={impactSeries.length <= 5 ? "sparse" : "standard"}
          dataCount={impactSeries.length}
          insightLayout="stack"
          eyebrow="Risk shift"
          title="Post-trade VaR delta"
          description="Positive bars indicate additional pressure introduced after the decision outcome, negative bars indicate net de-risking."
          insight={
            latestDecision ? (
              <div className="grid gap-4">
                <MetricBlock
                  label="Pre-trade VaR"
                  value={formatCurrency(latestDecision.pre_trade.var)}
                  hint={latestDecision.model_used.toUpperCase()}
                  tone="warning"
                  className="bg-transparent"
                />
                <MetricBlock
                  label="Post-trade VaR"
                  value={formatCurrency(latestDecision.post_trade.var)}
                  hint="Resulting state"
                  tone="accent"
                  className="bg-transparent"
                />
              </div>
            ) : undefined
          }
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No impact series is available yet.
            </div>
          }
        />

        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Recent decisions
          </div>
          <DecisionHistoryTable rows={decisions} />
        </div>
      </section>
    </div>
  );
}

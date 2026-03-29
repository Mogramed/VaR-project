import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { DecisionHistoryTable } from "@/components/data/risk-tables";
import { TradeDecisionPanel } from "@/components/forms/trade-decision-panel";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { makeBarOption, makeGroupedBarOption } from "@/lib/chart-options";
import {
  averageDecisionFillRatio,
  buildDecisionDeltaComparison,
  buildDecisionImpactSeries,
  buildDecisionVerdictCounts,
} from "@/lib/view-models";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

export default async function DeskDecisionsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const resolvedPortfolio = portfolioSlug ?? (await api.health()).portfolio_slug;

  const decisions = await api.recentDecisions(resolvedPortfolio, 12).catch(() => []);
  const verdicts = buildDecisionVerdictCounts(decisions);
  const fillRatio = averageDecisionFillRatio(decisions);
  const sizeComparison = buildDecisionDeltaComparison(decisions);
  const impactSeries = buildDecisionImpactSeries(decisions);
  const latestDecision = decisions[0] ?? null;

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Decision Layer"
        title="Accept, reduce or reject before anything reaches execution."
        description="The advisory layer is now denser and more operator-friendly: verdict, fill ratio, resulting posture and the latest rationale stay visible without oversized empty panels."
        aside={<StatusBadge label={resolvedPortfolio} tone="accent" />}
      />

      <section className="grid gap-4 xl:grid-cols-4">
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
      </section>

      <TradeDecisionPanel portfolioSlug={resolvedPortfolio} />

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
          description="Sizing friction is visible immediately, while denser series keep their zoom and shorter series give the space back to the page."
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
                  label="Approved notional"
                  value={formatCurrency(latestDecision.approved_delta_position_eur)}
                  hint={`Requested ${formatCurrency(latestDecision.requested_delta_position_eur)}`}
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

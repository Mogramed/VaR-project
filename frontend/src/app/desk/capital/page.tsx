import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { CapitalAllocationTable } from "@/components/data/risk-tables";
import { CapitalRebalancePanel } from "@/components/forms/capital-rebalance-panel";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { makeLineOption } from "@/lib/chart-options";
import { formatCurrency, formatPercent } from "@/lib/utils";
import {
  buildCapitalHistorySeries,
  flattenCapitalAllocations,
} from "@/lib/view-models";

export default async function DeskCapitalPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;

  const [capital, history] = await Promise.all([
    api.latestCapital(portfolioSlug).catch(() => null),
    api.capitalHistory(portfolioSlug, 18).catch(() => []),
  ]);

  const allocations = capital ? flattenCapitalAllocations(capital) : [];
  const topAllocation = allocations[0];
  const topRecommendation = capital?.recommendations?.[0];
  const capitalSeries = buildCapitalHistorySeries(history);

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Capital Management"
        title="Budget usage, headroom and rebalance recommendations."
        description="The capital surface is now tighter: history, allocation pressure and the next rebalance move sit close enough to support an actual desk decision."
        aside={capital ? <StatusBadge label={capital.status} tone="accent" /> : null}
      />

      {capital ? (
        <section className="grid gap-4 xl:grid-cols-4">
          <MetricBlock
            label="Budget"
            value={formatCurrency(capital.total_capital_budget_eur)}
            hint="Total allowed capital"
          />
          <MetricBlock
            label="Consumed"
            value={formatCurrency(capital.total_capital_consumed_eur)}
            hint="Capital currently in use"
            tone="warning"
          />
          <MetricBlock
            label="Reserved"
            value={formatCurrency(capital.total_capital_reserved_eur)}
            hint="Capital held back"
          />
          <MetricBlock
            label="Headroom"
            value={formatPercent(capital.headroom_ratio ?? 0, 0)}
            hint={capital.reference_model.toUpperCase()}
            tone="success"
          />
        </section>
      ) : null}

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.12fr)_360px]">
        <ChartSurface
          option={makeLineOption(capitalSeries, "#5fd4a6", { mode: "standard" })}
          mode="standard"
          dataCount={capitalSeries.length}
          eyebrow="Capital history"
          title="Consumed capital through time"
          description="History stays readable on longer sequences while compact enough not to isolate itself in a giant empty panel."
          meta={
            capital ? capital.reference_model.toUpperCase() : "No capital snapshot"
          }
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No capital history is available yet.
            </div>
          }
        />

        <div className="space-y-6">
          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Allocation pressure
            </div>
            {topAllocation ? (
              <div className="mt-5 space-y-4">
                <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
                  <div className="text-sm text-[var(--color-text-soft)]">
                    Most utilized symbol
                  </div>
                  <div className="mt-2 text-2xl font-semibold text-white">
                    {topAllocation.symbol}
                  </div>
                  <div className="mt-2 text-sm leading-6 text-[var(--color-text-soft)]">
                    {formatPercent(topAllocation.utilization)} utilized with{" "}
                    {formatCurrency(topAllocation.consumedCapital)} consumed.
                  </div>
                </div>
                <MetricBlock
                  label="Remaining capital"
                  value={
                    capital
                      ? formatCurrency(capital.total_capital_remaining_eur)
                      : "n/a"
                  }
                  hint="Budget still available"
                  tone="success"
                  className="bg-transparent"
                />
              </div>
            ) : (
              <div className="mt-5 text-sm text-[var(--color-text-muted)]">
                No allocation pressure data available.
              </div>
            )}
          </div>

          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Rebalance signal
            </div>
            <div className="mt-5 rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
              {topRecommendation ? (
                <>
                  <div className="text-lg font-semibold text-white">
                    {topRecommendation.symbol_from} {"->"} {topRecommendation.symbol_to}
                  </div>
                  <div className="mt-2 text-sm leading-7 text-[var(--color-text-soft)]">
                    Move {formatCurrency(topRecommendation.amount_eur)}.{" "}
                    {topRecommendation.reason}
                  </div>
                </>
              ) : (
                <div className="text-sm text-[var(--color-text-muted)]">
                  No rebalance recommendation is currently persisted.
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_390px]">
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Allocations
          </div>
          {capital ? (
            <CapitalAllocationTable rows={allocations} />
          ) : (
            <div className="surface rounded-[1.7rem] p-6 text-sm text-[var(--color-text-muted)]">
              No capital snapshot available yet.
            </div>
          )}
        </div>

        {capital ? (
          <CapitalRebalancePanel
            portfolioSlug={capital.portfolio_slug}
            referenceModel={capital.reference_model}
          />
        ) : null}
      </section>
    </div>
  );
}

import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { makeBarOption } from "@/lib/chart-options";
import {
  formatCurrency,
  formatPercent,
  formatTimestamp,
  humanizeIdentifier,
} from "@/lib/utils";
import { buildAlertSeverityCounts, buildDeskConsumptionSeries } from "@/lib/view-models";

export default async function DeskOverviewPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;

  const [health, alerts, capital, comparison, snapshot] = await Promise.all([
    api.health(),
    api.recentAlerts(12).catch(() => []),
    api.latestCapital(portfolioSlug).catch(() => null),
    api.latestModelComparison(portfolioSlug).catch(() => null),
    api.latestSnapshot(portfolioSlug).catch(() => null),
  ]);
  const [marketStatus, reconciliation] = await Promise.all([
    api.marketDataStatus(portfolioSlug).catch(() => null),
    api.reconciliationSummary(portfolioSlug).catch(() => null),
  ]);

  const deskSlug = health.desk_slug ?? "main";
  const desk = await api.deskOverview(deskSlug).catch(() => null);
  const payload = (snapshot?.payload ?? {}) as {
    var?: Record<string, number>;
    es?: Record<string, number>;
  };
  const selectedModel =
    comparison?.champion_model ?? capital?.reference_model ?? "hist";
  const varValue = Number(
    payload.var?.[selectedModel] ?? Object.values(payload.var ?? {})[0] ?? 0,
  );
  const esValue = Number(
    payload.es?.[selectedModel] ?? Object.values(payload.es ?? {})[0] ?? 0,
  );
  const alertCounts = buildAlertSeverityCounts(alerts);
  const deskSeries = desk ? buildDeskConsumptionSeries(desk) : [];
  const deskPortfolios = desk?.portfolios ?? [];
  const topPortfolio =
    deskPortfolios
      .slice()
      .sort(
        (left, right) =>
          right.total_capital_consumed_eur - left.total_capital_consumed_eur,
      )[0] ?? null;

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Desk Overview"
        title="Current risk posture across the FX desk."
        description="A denser operator view: portfolio load, model leadership, capital headroom and alert pressure all read in a few seconds."
        aside={capital ? <StatusBadge label={capital.status} tone="accent" /> : null}
      />

      <section className="grid gap-4 xl:grid-cols-4">
        <MetricBlock
          label={`VaR / ${selectedModel.toUpperCase()}`}
          value={formatCurrency(varValue)}
          hint="Current portfolio risk"
          tone="accent"
        />
        <MetricBlock
          label={`ES / ${selectedModel.toUpperCase()}`}
          value={formatCurrency(esValue)}
          hint="Tail loss expectation"
          tone="warning"
        />
        <MetricBlock
          label="Capital headroom"
          value={capital ? formatPercent(capital.headroom_ratio ?? 0, 0) : "n/a"}
          hint={
            capital
              ? `Remaining ${formatCurrency(capital.total_capital_remaining_eur)}`
              : "No capital snapshot yet"
          }
          tone="success"
        />
        <MetricBlock
          label="Market sync"
          value={(marketStatus?.status ?? "n/a").toUpperCase()}
          hint={
            marketStatus?.latest_sync_at
              ? `Last sync ${formatTimestamp(marketStatus.latest_sync_at)}`
              : "No MT5 market sync yet"
          }
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_360px]">
        <ChartSurface
          option={makeBarOption(
            deskSeries,
            {
              color: "#d89b49",
              negativeColor: "#5fd4a6",
              mode: deskSeries.length <= 3 ? "sparse" : "comparison",
            },
          )}
          mode={deskSeries.length <= 3 ? "sparse" : "comparison"}
          dataCount={deskSeries.length}
          insightLayout="stack"
          eyebrow="Desk capital load"
          title="Capital consumed by portfolio"
          description="The overview expands or compresses based on how many portfolio slices exist, so sparse desks do not leave a giant dead canvas."
          meta={
            desk?.generated_at ? formatTimestamp(desk.generated_at) : "Live view"
          }
          insight={
            desk ? (
              <div className="grid gap-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <MetricBlock
                    label="Consumed"
                    value={formatCurrency(desk.total_capital_consumed_eur)}
                    hint="Across the desk"
                    tone="warning"
                    className="h-full bg-transparent"
                  />
                  <MetricBlock
                    label="Remaining"
                    value={formatCurrency(desk.total_capital_remaining_eur)}
                    hint="Budget still available"
                    tone="success"
                    className="h-full bg-transparent"
                  />
                </div>
                {topPortfolio ? (
                  <div className="rounded-[1.4rem] border border-white/8 bg-black/18 px-4 py-4">
                    <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                      Leading pressure point
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-3">
                      <div>
                        <div className="text-lg font-semibold text-white">
                          {humanizeIdentifier(topPortfolio.portfolio_name)}
                        </div>
                        <div className="mt-1 text-sm text-[var(--color-text-soft)]">
                          {formatPercent(topPortfolio.utilization ?? 0)} utilized with{" "}
                          {topPortfolio.alert_count} alerts.
                        </div>
                      </div>
                      <StatusBadge label={topPortfolio.status} />
                    </div>
                  </div>
                ) : null}
              </div>
            ) : undefined
          }
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              Desk snapshot unavailable.
            </div>
          }
        />

        <div className="space-y-6">
          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Alert posture
            </div>
            <div className="mt-5 grid gap-4 sm:grid-cols-3 xl:grid-cols-1 2xl:grid-cols-3">
              <MetricBlock
                label="Warn"
                value={String(alertCounts.warn ?? 0)}
                tone="warning"
                className="bg-transparent"
              />
              <MetricBlock
                label="Breach"
                value={String(alertCounts.breach ?? 0)}
                tone="danger"
                className="bg-transparent"
              />
              <MetricBlock
                label="Info"
                value={String(alertCounts.info ?? 0)}
                tone="success"
                className="bg-transparent"
              />
            </div>
          </div>

          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Desk alignment
            </div>
            <div className="mt-5 grid gap-4">
              <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
                <div className="text-sm text-[var(--color-text-soft)]">Champion</div>
                <div className="mt-2 text-2xl font-semibold text-white">
                  {(comparison?.champion_model ?? "n/a").toUpperCase()}
                </div>
              </div>
              <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
                <div className="text-sm text-[var(--color-text-soft)]">Manual MT5 events</div>
                <div className="mt-2 text-2xl font-semibold text-white">
                  {String(reconciliation?.manual_event_count ?? 0)}
                </div>
              </div>
              <div className="text-sm leading-7 text-[var(--color-text-soft)]">
                Latest snapshot{" "}
                {snapshot?.created_at ? formatTimestamp(snapshot.created_at) : "not available"}
                . {reconciliation?.unmatched_execution_count ?? 0} unmatched execution attempt
                {(reconciliation?.unmatched_execution_count ?? 0) === 1 ? "" : "s"}.
              </div>
            </div>
          </div>
        </div>
      </section>

      {desk ? (
        <section className="surface rounded-[1.8rem] p-6">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
                Portfolio slices
              </div>
              <h2 className="mt-3 text-2xl font-semibold text-white">
                Desk distribution at a glance
              </h2>
            </div>
            <div className="text-sm text-[var(--color-text-soft)]">
              {deskPortfolios.length} active portfolio
              {deskPortfolios.length > 1 ? "s" : ""}
            </div>
          </div>
          <div className="mt-6 grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
            {deskPortfolios.map((portfolio) => (
              <div
                key={portfolio.portfolio_slug}
                className="rounded-[1.45rem] border border-white/8 bg-black/18 p-4"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="text-base font-semibold text-white">
                    {humanizeIdentifier(portfolio.portfolio_name)}
                  </div>
                  <StatusBadge label={portfolio.status} />
                </div>
                <div className="mt-4 space-y-3 text-sm text-[var(--color-text-soft)]">
                  <div className="flex items-center justify-between">
                    <span>Consumed</span>
                    <span className="mono text-white">
                      {formatCurrency(portfolio.total_capital_consumed_eur)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Remaining</span>
                    <span className="mono text-white">
                      {formatCurrency(portfolio.total_capital_remaining_eur)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Utilization</span>
                    <span className="mono text-white">
                      {formatPercent(portfolio.utilization ?? 0)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Alerts</span>
                    <span className="mono text-white">{portfolio.alert_count}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}

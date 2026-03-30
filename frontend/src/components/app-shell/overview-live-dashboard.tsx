"use client";

import { startTransition, useEffect, useState } from "react";

import { OverviewLiveStripPanel } from "@/components/app-shell/overview-live-strip";
import { ChartSurface } from "@/components/charts/chart-surface";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { makeBarOption } from "@/lib/chart-options";
import type { CapitalUsageSnapshotResponse, DeskSnapshotResponse, MT5LiveStateResponse } from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatPercent, formatTimestamp, humanizeIdentifier } from "@/lib/utils";
import { buildDeskConsumptionSeries } from "@/lib/view-models";

export function OverviewLiveDashboard({
  deskSlug,
  portfolioSlug,
  initialDesk,
  initialLiveState,
  initialCapital,
  fallbackSelectedModel,
  fallbackVarValue,
  fallbackEsValue,
  alertCounts,
  championModel,
  snapshotCreatedAt,
}: {
  deskSlug: string;
  portfolioSlug: string;
  initialDesk: DeskSnapshotResponse | null;
  initialLiveState: MT5LiveStateResponse | null;
  initialCapital: CapitalUsageSnapshotResponse | null;
  fallbackSelectedModel: string;
  fallbackVarValue: number;
  fallbackEsValue: number;
  alertCounts: Record<string, number>;
  championModel: string | null;
  snapshotCreatedAt: string | null;
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [desk, setDesk] = useState<DeskSnapshotResponse | null>(initialDesk);

  useEffect(() => {
    let cancelled = false;

    const refreshDesk = async () => {
      try {
        const nextDesk = await api.deskOverview(deskSlug);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setDesk(nextDesk);
        });
      } catch {
        // Keep current desk snapshot on transient failures.
      }
    };

    void refreshDesk();
    return () => {
      cancelled = true;
    };
  }, [deskSlug, liveState?.sequence]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const nextDesk = await api.deskOverview(deskSlug);
          if (cancelled) {
            return;
          }
          startTransition(() => {
            setDesk(nextDesk);
          });
        } catch {
          // Keep current desk snapshot on transient failures.
        }
      })();
    }, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [deskSlug]);

  const reconciliation = liveState?.reconciliation ?? null;
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
    <div className="space-y-8">
      <OverviewLiveStripPanel
        liveState={liveState}
        transport={transport}
        initialCapital={initialCapital}
        fallbackSelectedModel={fallbackSelectedModel}
        fallbackVarValue={fallbackVarValue}
        fallbackEsValue={fallbackEsValue}
      />

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
          description="The desk chart now follows the MT5 bridge cadence, so portfolio slices refresh as live capital posture changes."
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
                  {(championModel ?? "n/a").toUpperCase()}
                </div>
              </div>
              <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
                <div className="text-sm text-[var(--color-text-soft)]">Manual MT5 events</div>
                <div className="mt-2 text-2xl font-semibold text-white">
                  {String(reconciliation?.manual_event_count ?? 0)}
                </div>
              </div>
              <div className="text-sm leading-7 text-[var(--color-text-soft)]">
                Latest snapshot {snapshotCreatedAt ? formatTimestamp(snapshotCreatedAt) : "not available"}
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

"use client";

import { startTransition, useEffect, useState } from "react";

import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { CapitalAllocationTable } from "@/components/data/risk-tables";
import { CapitalRebalancePanel } from "@/components/forms/capital-rebalance-panel";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { makeLineOption } from "@/lib/chart-options";
import type {
  CapitalUsageSnapshotResponse,
  MT5LiveStateResponse,
} from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";
import {
  buildCapitalHistorySeries,
  flattenCapitalAllocations,
} from "@/lib/view-models";

function preferredHistorySource(liveState: MT5LiveStateResponse | null) {
  return liveState?.capital_usage?.snapshot_source === "mt5_live_bridge"
    ? "mt5_live_bridge"
    : undefined;
}

export function CapitalLiveSurface({
  portfolioSlug,
  initialLiveState,
  initialCapital,
  initialHistory,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialCapital: CapitalUsageSnapshotResponse | null;
  initialHistory: CapitalUsageSnapshotResponse[];
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [capital, setCapital] = useState<CapitalUsageSnapshotResponse | null>(initialCapital);
  const [history, setHistory] = useState<CapitalUsageSnapshotResponse[]>(initialHistory);
  const historySource = preferredHistorySource(liveState);

  useEffect(() => {
    const nextCapital = liveState?.capital_usage;
    if (!nextCapital) {
      return;
    }
    startTransition(() => {
      setCapital(nextCapital);
    });
  }, [liveState?.sequence, liveState?.capital_usage]);

  useEffect(() => {
    let cancelled = false;

    const refreshHistory = async () => {
      try {
        let nextHistory = await api.capitalHistory(portfolioSlug, 18, historySource);
        if (historySource && nextHistory.length === 0) {
          nextHistory = await api.capitalHistory(portfolioSlug, 18);
        }
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setHistory(nextHistory);
        });
      } catch {
        // Keep the current capital history on transient failures.
      }
    };

    void refreshHistory();
    return () => {
      cancelled = true;
    };
  }, [historySource, portfolioSlug, liveState?.sequence]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setInterval(() => {
      void (async () => {
        const source = preferredHistorySource(liveState);
        try {
          let nextHistory = await api.capitalHistory(portfolioSlug, 18, source);
          if (source && nextHistory.length === 0) {
            nextHistory = await api.capitalHistory(portfolioSlug, 18);
          }
          if (cancelled) {
            return;
          }
          startTransition(() => {
            setHistory(nextHistory);
          });
        } catch {
          // Keep the current capital history on transient failures.
        }
      })();
    }, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [portfolioSlug, liveState]);

  const resolvedCapital = liveState?.capital_usage ?? capital;
  const allocations = resolvedCapital ? flattenCapitalAllocations(resolvedCapital) : [];
  const topAllocation = allocations[0];
  const topRecommendation = resolvedCapital?.recommendations?.[0];
  const capitalSeries = buildCapitalHistorySeries(history);
  const operatorAlerts = liveState?.operator_alerts ?? [];

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Capital Management"
        title="Budget usage, headroom and rebalance recommendations."
        description="This surface now follows the MT5 bridge cadence: live capital posture stays front and center, while the persisted history updates from mt5_live_bridge snapshots instead of feeling frozen between manual runs."
        aside={
          <div className="flex flex-wrap items-center gap-2">
            {resolvedCapital ? (
              <StatusBadge label={resolvedCapital.status} tone="accent" />
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
          label="Budget"
          value={resolvedCapital ? formatCurrency(resolvedCapital.total_capital_budget_eur) : "n/a"}
          hint="Total allowed capital"
        />
        <MetricBlock
          label="Consumed"
          value={resolvedCapital ? formatCurrency(resolvedCapital.total_capital_consumed_eur) : "n/a"}
          hint="Capital currently in use"
          tone="warning"
        />
        <MetricBlock
          label="Reserved"
          value={resolvedCapital ? formatCurrency(resolvedCapital.total_capital_reserved_eur) : "n/a"}
          hint="Capital held back"
        />
        <MetricBlock
          label="Headroom"
          value={resolvedCapital ? formatPercent(resolvedCapital.headroom_ratio ?? 0, 0) : "n/a"}
          hint={resolvedCapital?.reference_model?.toUpperCase() ?? "No capital snapshot"}
          tone="success"
        />
        <MetricBlock
          label="Bridge"
          value={(liveState?.status ?? "pending").toUpperCase()}
          hint={
            liveState?.generated_at
              ? `Seq ${liveState.sequence} · ${formatTimestamp(liveState.generated_at)}`
              : "Waiting for MT5 live state"
          }
          tone={
            liveState?.status === "ok"
              ? "success"
              : liveState?.degraded || liveState?.stale
                ? "warning"
                : "neutral"
          }
        />
      </section>

      <LiveOperatorAlerts
        alerts={operatorAlerts}
        title="Capital watchlist"
        copy="Capital pressure, bridge health and reconciliation incidents are surfaced here while the budget history keeps updating from live snapshots."
      />

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.12fr)_360px]">
        <ChartSurface
          option={makeLineOption(capitalSeries, "#5fd4a6", { mode: "standard" })}
          mode="standard"
          dataCount={capitalSeries.length}
          eyebrow="Capital history"
          title="Consumed capital through time"
          description="The chart now follows the live bridge persistence path, so the capital curve keeps moving as the desk posture changes."
          meta={
            resolvedCapital?.snapshot_timestamp
              ? `Live ${formatTimestamp(resolvedCapital.snapshot_timestamp)}`
              : resolvedCapital
                ? resolvedCapital.reference_model.toUpperCase()
                : "No capital snapshot"
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
                    resolvedCapital
                      ? formatCurrency(resolvedCapital.total_capital_remaining_eur)
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
            <div className="mt-4 text-sm leading-7 text-[var(--color-text-soft)]">
              {liveState?.generated_at
                ? `Bridge refreshed ${formatTimestamp(liveState.generated_at)}.`
                : "Bridge refresh pending."}{" "}
              {resolvedCapital?.snapshot_source
                ? `Current capital source: ${resolvedCapital.snapshot_source}.`
                : "No live capital source available yet."}
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_390px]">
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Allocations
          </div>
          {resolvedCapital ? (
            <CapitalAllocationTable rows={allocations} />
          ) : (
            <div className="surface rounded-[1.7rem] p-6 text-sm text-[var(--color-text-muted)]">
              No capital snapshot available yet.
            </div>
          )}
        </div>

        {resolvedCapital ? (
          <CapitalRebalancePanel
            portfolioSlug={resolvedCapital.portfolio_slug}
            referenceModel={resolvedCapital.reference_model}
            onRebalanced={(nextCapital) => {
              startTransition(() => {
                setCapital(nextCapital);
                setHistory((current) => [nextCapital, ...current].slice(0, 18));
              });
            }}
          />
        ) : null}
      </section>
    </div>
  );
}

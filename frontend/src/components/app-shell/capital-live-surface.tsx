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
import { CHART_PALETTE, makeLineOption } from "@/lib/chart-options";
import type { CapitalUsageSnapshotResponse, MT5LiveStateResponse } from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";
import { buildCapitalHistorySeries, flattenCapitalAllocations } from "@/lib/view-models";

function preferredSource(ls: MT5LiveStateResponse | null) {
  return ls?.capital_usage?.snapshot_source === "mt5_live_bridge" ? "mt5_live_bridge" : undefined;
}

export function CapitalLiveSurface({
  portfolioSlug, initialLiveState, initialCapital, initialHistory,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialCapital: CapitalUsageSnapshotResponse | null;
  initialHistory: CapitalUsageSnapshotResponse[];
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [capital, setCapital] = useState(initialCapital);
  const [history, setHistory] = useState(initialHistory);
  const src = preferredSource(liveState);

  useEffect(() => {
    const next = liveState?.capital_usage;
    if (next) startTransition(() => setCapital(next));
  }, [liveState?.sequence, liveState?.capital_usage]);

  useEffect(() => {
    let c = false;
    (async () => {
      try {
        let h = await api.capitalHistory(portfolioSlug, 18, src);
        if (src && h.length === 0) h = await api.capitalHistory(portfolioSlug, 18);
        if (!c) startTransition(() => setHistory(h));
      } catch { /* keep */ }
    })();
    return () => { c = true; };
  }, [src, portfolioSlug, liveState?.sequence]);

  useEffect(() => {
    let c = false;
    const t = window.setInterval(() => {
      (async () => {
        try {
          let h = await api.capitalHistory(portfolioSlug, 18, preferredSource(liveState));
          if (preferredSource(liveState) && h.length === 0) h = await api.capitalHistory(portfolioSlug, 18);
          if (!c) startTransition(() => setHistory(h));
        } catch { /* keep */ }
      })();
    }, 15000);
    return () => { c = true; window.clearInterval(t); };
  }, [portfolioSlug, liveState]);

  const resolved = liveState?.capital_usage ?? capital;
  const allocations = resolved ? flattenCapitalAllocations(resolved) : [];
  const capitalSeries = buildCapitalHistorySeries(history);

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Capital" title="Budget usage, headroom and rebalance"
        aside={<>
          {resolved ? <StatusBadge label={resolved.status} tone="accent" /> : null}
          <StatusBadge label={transport} tone={transport === "stream" ? "success" : "warning"} />
        </>}
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricBlock label="Budget" value={resolved ? formatCurrency(resolved.total_capital_budget_eur) : "n/a"} />
        <MetricBlock label="Consumed" value={resolved ? formatCurrency(resolved.total_capital_consumed_eur) : "n/a"} tone="warning" />
        <MetricBlock label="Reserved" value={resolved ? formatCurrency(resolved.total_capital_reserved_eur) : "n/a"} />
        <MetricBlock label="Headroom" value={resolved ? formatPercent(resolved.headroom_ratio ?? 0, 0) : "n/a"} tone="success" />
        <MetricBlock label="Bridge" value={(liveState?.status ?? "pending").toUpperCase()} hint={liveState?.generated_at ? formatTimestamp(liveState.generated_at) : undefined}
          tone={liveState?.status === "ok" ? "success" : liveState?.degraded ? "warning" : "neutral"} />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_300px]">
        <ChartSurface
          option={makeLineOption(capitalSeries, CHART_PALETTE.green, { mode: "standard" })}
          mode="standard" dataCount={capitalSeries.length}
          title="Capital history" meta={resolved?.snapshot_timestamp ? formatTimestamp(resolved.snapshot_timestamp) : undefined}
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No capital history available.</p>}
        />
        <div className="space-y-3">
          {allocations[0] ? (
            <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Top allocation</div>
              <div className="text-lg font-semibold text-[var(--color-text)]">{allocations[0].symbol}</div>
              <div className="mt-1 text-xs text-[var(--color-text-muted)]">
                {formatPercent(allocations[0].utilization)} util · {formatCurrency(allocations[0].consumedCapital)} consumed
              </div>
            </div>
          ) : null}
          {resolved?.recommendations?.[0] ? (
            <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Rebalance signal</div>
              <div className="text-xs font-semibold text-[var(--color-text)]">
                {resolved.recommendations[0].symbol_from} → {resolved.recommendations[0].symbol_to}
              </div>
              <div className="mt-1 text-[11px] text-[var(--color-text-muted)]">
                {formatCurrency(resolved.recommendations[0].amount_eur)} · {resolved.recommendations[0].reason}
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Allocations</h4>
          {resolved ? <CapitalAllocationTable rows={allocations} /> : <p className="text-xs text-[var(--color-text-muted)]">No capital snapshot.</p>}
        </div>
        {resolved ? <CapitalRebalancePanel portfolioSlug={resolved.portfolio_slug} referenceModel={resolved.reference_model}
          onRebalanced={(r) => startTransition(() => { setCapital(r); setHistory((h) => [r, ...h].slice(0, 18)); })} /> : null}
      </div>
    </div>
  );
}

"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";

import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { LivePostureBanner } from "@/components/app-shell/live-posture-banner";
import { LiveRuntimeBadgeGroup } from "@/components/app-shell/live-runtime-badge-group";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { CapitalAllocationTable } from "@/components/data/risk-tables";
import { CapitalRebalancePanel } from "@/components/forms/capital-rebalance-panel";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { CHART_PALETTE, makeLineOption } from "@/lib/chart-options";
import type { CapitalUsageSnapshotResponse } from "@/lib/api/types";
import {
  deskArtifactQueryKey,
  deskArtifactQueryOptions,
} from "@/components/app-shell/desk-artifact-query";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";
import { buildCapitalHistorySeries, flattenCapitalAllocations } from "@/lib/view-models";

function preferredSource(ls: ReturnType<typeof useDeskLive>["liveState"]) {
  return ls?.capital_usage?.snapshot_source === "mt5_live_bridge" ? "mt5_live_bridge" : "auto";
}

export function CapitalLiveSurface({
  portfolioSlug, initialCapital, initialHistory,
}: {
  portfolioSlug: string;
  initialCapital: CapitalUsageSnapshotResponse | null;
  initialHistory: CapitalUsageSnapshotResponse[];
}) {
  const { liveState, transport } = useDeskLive();
  const queryClient = useQueryClient();
  const src = preferredSource(liveState);
  const capitalQueryKey = deskArtifactQueryKey("capital", "latest", portfolioSlug, src);
  const historyQueryKey = deskArtifactQueryKey("capital", "history", portfolioSlug, 18, src);
  const capitalQuery = useQuery({
    queryKey: capitalQueryKey,
    queryFn: () => api.latestCapital(portfolioSlug, src),
    initialData: initialCapital,
    ...deskArtifactQueryOptions,
  });
  const historyQuery = useQuery({
    queryKey: historyQueryKey,
    queryFn: () => api.capitalHistory(portfolioSlug, 18, src),
    initialData: initialHistory,
    ...deskArtifactQueryOptions,
  });

  const resolved = liveState?.capital_usage ?? capitalQuery.data ?? initialCapital;
  const history = historyQuery.data ?? initialHistory;
  const allocations = resolved ? flattenCapitalAllocations(resolved) : [];
  const capitalSeries = buildCapitalHistorySeries(history);

  return (
    <div className="desk-page space-y-4">
      <PageHeader
        eyebrow="Capital"
        title="Budget usage, headroom and rebalance"
        aside={(
          <>
            {resolved ? <StatusBadge label={resolved.status} tone="accent" /> : null}
            <LiveRuntimeBadgeGroup liveState={liveState} transport={transport} showBridge={false} />
          </>
        )}
      />
      <LivePostureBanner liveState={liveState} transport={transport} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricBlock label="Budget" value={resolved ? formatCurrency(resolved.total_capital_budget_eur) : "n/a"} />
        <MetricBlock label="Consumed" value={resolved ? formatCurrency(resolved.total_capital_consumed_eur) : "n/a"} tone="warning" />
        <MetricBlock label="Reserved" value={resolved ? formatCurrency(resolved.total_capital_reserved_eur) : "n/a"} />
        <MetricBlock label="Headroom" value={resolved ? formatPercent(resolved.headroom_ratio ?? 0, 0) : "n/a"} tone="success" />
        <MetricBlock
          label="Bridge"
          value={(liveState?.status ?? "pending").toUpperCase()}
          hint={liveState?.generated_at ? formatTimestamp(liveState.generated_at) : undefined}
          tone={liveState?.status === "ok" ? "success" : liveState?.degraded ? "warning" : "neutral"}
        />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_300px]">
        <ChartSurface
          option={makeLineOption(capitalSeries, CHART_PALETTE.green, { mode: "standard" })}
          mode="standard"
          dataCount={capitalSeries.length}
          title="Capital history"
          meta={resolved?.snapshot_timestamp ? formatTimestamp(resolved.snapshot_timestamp) : undefined}
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No capital history available.</p>}
        />
        <div className="space-y-3">
          {allocations[0] ? (
            <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Top allocation</div>
              <div className="text-lg font-semibold text-[var(--color-text)]">{allocations[0].symbol}</div>
              <div className="mt-1 text-xs text-[var(--color-text-muted)]">
                {`${formatPercent(allocations[0].utilization)} util | ${formatCurrency(allocations[0].consumedCapital)} consumed`}
              </div>
            </div>
          ) : null}
          {resolved?.recommendations?.[0] ? (
            <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Rebalance signal</div>
              <div className="text-xs font-semibold text-[var(--color-text)]">
                {`${resolved.recommendations[0].symbol_from} -> ${resolved.recommendations[0].symbol_to}`}
              </div>
              <div className="mt-1 text-[11px] text-[var(--color-text-muted)]">
                {`${formatCurrency(resolved.recommendations[0].amount_eur)} | ${resolved.recommendations[0].reason}`}
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
        {resolved ? (
          <CapitalRebalancePanel
            portfolioSlug={resolved.portfolio_slug}
            referenceModel={resolved.reference_model}
            onRebalanced={(result) => {
              queryClient.setQueryData<CapitalUsageSnapshotResponse | null>(capitalQueryKey, result);
              queryClient.setQueryData<CapitalUsageSnapshotResponse[]>(
                historyQueryKey,
                (current) => [result, ...(current ?? [])].slice(0, 18),
              );
            }}
          />
        ) : null}
      </div>
    </div>
  );
}

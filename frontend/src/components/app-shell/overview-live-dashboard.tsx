"use client";

import { startTransition, useEffect, useState } from "react";

import { OverviewLiveStripPanel } from "@/components/app-shell/overview-live-strip";
import { ChartSurface } from "@/components/charts/chart-surface";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { CHART_PALETTE, makeBarOption } from "@/lib/chart-options";
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
    const refresh = async () => {
      try {
        const next = await api.deskOverview(deskSlug);
        if (!cancelled) startTransition(() => setDesk(next));
      } catch { /* keep current */ }
    };
    void refresh();
    return () => { cancelled = true; };
  }, [deskSlug, liveState?.sequence]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const next = await api.deskOverview(deskSlug);
          if (!cancelled) startTransition(() => setDesk(next));
        } catch { /* keep current */ }
      })();
    }, 15000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [deskSlug]);

  const reconciliation = liveState?.reconciliation ?? null;
  const deskSeries = desk ? buildDeskConsumptionSeries(desk) : [];
  const deskPortfolios = desk?.portfolios ?? [];

  return (
    <div className="space-y-4">
      <OverviewLiveStripPanel
        liveState={liveState}
        transport={transport}
        initialCapital={initialCapital}
        fallbackSelectedModel={fallbackSelectedModel}
        fallbackVarValue={fallbackVarValue}
        fallbackEsValue={fallbackEsValue}
      />

      {/* Chart + side panel */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_300px]">
        <ChartSurface
          option={makeBarOption(deskSeries, {
            color: CHART_PALETTE.gold,
            negativeColor: CHART_PALETTE.green,
            mode: deskSeries.length <= 3 ? "sparse" : "comparison",
          })}
          mode={deskSeries.length <= 3 ? "sparse" : "comparison"}
          dataCount={deskSeries.length}
          title="Capital by portfolio"
          meta={desk?.generated_at ? formatTimestamp(desk.generated_at) : undefined}
          emptyState={
            <p className="text-xs text-[var(--color-text-muted)]">No desk snapshot available.</p>
          }
        />

        <div className="space-y-3">
          {/* Alert posture */}
          <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Alert posture
            </div>
            <div className="grid grid-cols-3 gap-2">
              <MiniStat label="Warn" value={alertCounts.warn ?? 0} color="var(--color-amber)" />
              <MiniStat label="Breach" value={alertCounts.breach ?? 0} color="var(--color-red)" />
              <MiniStat label="Info" value={alertCounts.info ?? 0} color="var(--color-green)" />
            </div>
          </div>

          {/* Desk alignment */}
          <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Desk alignment
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Champion</span>
                <span className="mono font-semibold text-[var(--color-accent)]">
                  {(championModel ?? "n/a").toUpperCase()}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Manual events</span>
                <span className="mono text-[var(--color-text)]">{reconciliation?.manual_event_count ?? 0}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Latest snapshot</span>
                <span className="mono text-[10px] text-[var(--color-text-muted)]">
                  {snapshotCreatedAt ? formatTimestamp(snapshotCreatedAt) : "n/a"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Portfolio slices */}
      {deskPortfolios.length > 0 ? (
        <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {deskPortfolios.map((p) => (
            <div
              key={p.portfolio_slug}
              className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--color-text)]">
                  {humanizeIdentifier(p.portfolio_name)}
                </span>
                <StatusBadge label={p.status} />
              </div>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Consumed</span>
                  <span className="mono text-[var(--color-text)]">{formatCurrency(p.total_capital_consumed_eur)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Remaining</span>
                  <span className="mono text-[var(--color-text)]">{formatCurrency(p.total_capital_remaining_eur)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Utilization</span>
                  <span className="mono text-[var(--color-text)]">{formatPercent(p.utilization ?? 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--color-text-muted)]">Alerts</span>
                  <span className="mono text-[var(--color-text)]">{p.alert_count}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function MiniStat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="text-center">
      <div className="mono text-lg font-semibold" style={{ color }}>{value}</div>
      <div className="text-[9px] uppercase tracking-wider text-[var(--color-text-muted)]">{label}</div>
    </div>
  );
}

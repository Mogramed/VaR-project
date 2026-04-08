"use client";

import { startTransition, useEffect, useState } from "react";

import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { OverviewLiveStripPanel } from "@/components/app-shell/overview-live-strip";
import { ChartSurface } from "@/components/charts/chart-surface";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { CHART_PALETTE, makeBarOption } from "@/lib/chart-options";
import type { CapitalUsageSnapshotResponse, DeskSnapshotResponse } from "@/lib/api/types";
import {
  formatCurrency,
  formatOperationalTruth,
  formatPercent,
  formatSourceLabel,
  formatTimestamp,
  formatTimestampWithSource,
  humanizeIdentifier,
  joinLabelParts,
} from "@/lib/utils";
import { buildDeskConsumptionSeries } from "@/lib/view-models";

export function OverviewLiveDashboard({
  deskSlug,
  initialDesk,
  initialCapital,
  fallbackSelectedModel,
  fallbackVarValue,
  fallbackEsValue,
  alertCounts,
  championModel,
  snapshotCreatedAt,
  snapshotSource,
}: {
  deskSlug: string;
  initialDesk: DeskSnapshotResponse | null;
  initialCapital: CapitalUsageSnapshotResponse | null;
  fallbackSelectedModel: string;
  fallbackVarValue: number;
  fallbackEsValue: number;
  alertCounts: Record<string, number>;
  championModel: string | null;
  snapshotCreatedAt: string | null;
  snapshotSource: string | null;
}) {
  const { liveState, transport, artifactVersion } = useDeskLive();
  const [desk, setDesk] = useState<DeskSnapshotResponse | null>(initialDesk);
  const liveCapitalTimestamp = liveState?.capital_usage?.snapshot_timestamp ?? null;

  useEffect(() => {
    startTransition(() => setDesk(initialDesk));
  }, [initialDesk]);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      try {
        const next = await api.deskOverview(deskSlug);
        if (!cancelled) startTransition(() => setDesk(next));
      } catch { /* keep current */ }
    };
    void refresh();
    return () => {
      cancelled = true;
    };
  }, [artifactVersion, deskSlug, liveCapitalTimestamp]);

  const reconciliation = liveState?.reconciliation ?? null;
  const deskSeries = desk ? buildDeskConsumptionSeries(desk) : [];
  const deskPortfolios = desk?.portfolios ?? [];
  const riskTimestamp =
    liveState?.risk_summary?.latest_observation ??
    liveState?.risk_summary?.generated_at ??
    initialCapital?.snapshot_timestamp ??
    initialCapital?.created_at ??
    snapshotCreatedAt ??
    null;
  const riskSource =
    liveState?.risk_summary?.source ??
    initialCapital?.snapshot_source ??
    initialCapital?.source ??
    snapshotSource ??
    null;
  const capitalTimestamp =
    liveState?.capital_usage?.snapshot_timestamp ??
    liveState?.capital_usage?.created_at ??
    initialCapital?.snapshot_timestamp ??
    initialCapital?.created_at ??
    null;
  const capitalSource =
    liveState?.capital_usage?.snapshot_source ??
    liveState?.capital_usage?.source ??
    initialCapital?.snapshot_source ??
    initialCapital?.source ??
    null;
  const marketReference = formatTimestampWithSource({
    source: reconciliation?.market_reference_source,
    timestamp: reconciliation?.market_reference_timestamp,
  });
  const persistedSnapshot = formatTimestampWithSource({
    source: snapshotSource,
    timestamp: snapshotCreatedAt,
  });
  const integrityChecks: Array<{
    id?: unknown;
    label?: unknown;
    status?: unknown;
    message?: unknown;
  }> = Array.isArray(liveState?.quality_checks)
    ? (liveState?.quality_checks as Array<Record<string, unknown>>)
    : [];
  const integrityCounts = integrityChecks.reduce<{ pass: number; warn: number; fail: number }>(
    (acc, check) => {
      const status = String(check.status ?? "").trim().toLowerCase();
      if (status === "pass") {
        acc.pass += 1;
      } else if (status === "warn") {
        acc.warn += 1;
      } else if (status === "fail") {
        acc.fail += 1;
      }
      return acc;
    },
    { pass: 0, warn: 0, fail: 0 },
  );
  const truthScore = typeof liveState?.truth_score === "number"
    ? `${Math.round(liveState.truth_score * 100)}%`
    : "n/a";

  return (
    <div className="space-y-4">
      <OverviewLiveStripPanel
        liveState={liveState}
        transport={transport}
        initialCapital={initialCapital}
        fallbackSelectedModel={fallbackSelectedModel}
        fallbackVarValue={fallbackVarValue}
        fallbackEsValue={fallbackEsValue}
        fallbackSnapshotCreatedAt={snapshotCreatedAt}
        fallbackSnapshotSource={snapshotSource}
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
          meta={
            desk?.generated_at
              ? joinLabelParts("Desk snapshot", formatTimestamp(desk.generated_at))
              : undefined
          }
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
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-[var(--color-text-muted)]">Truth source</span>
                <span className="text-right text-[var(--color-text)]">
                  {reconciliation?.operational_truth
                    ? formatOperationalTruth(reconciliation.operational_truth)
                    : (initialCapital ?? snapshotSource)
                      ? "Persisted snapshot"
                      : "n/a"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-[var(--color-text-muted)]">Risk basis</span>
                <span className="text-right text-[var(--color-text)]">{formatSourceLabel(riskSource)}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-[var(--color-text-muted)]">Risk as of</span>
                <span className="mono text-right text-[10px] text-[var(--color-text-muted)]">
                  {riskTimestamp ? formatTimestamp(riskTimestamp) : "n/a"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-[var(--color-text-muted)]">Capital basis</span>
                <span className="text-right text-[var(--color-text)]">{formatSourceLabel(capitalSource)}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-[var(--color-text-muted)]">Capital as of</span>
                <span className="mono text-right text-[10px] text-[var(--color-text-muted)]">
                  {capitalTimestamp ? formatTimestamp(capitalTimestamp) : "n/a"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-[var(--color-text-muted)]">Market ref</span>
                <span className="mono text-right text-[10px] text-[var(--color-text-muted)]">{marketReference}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Manual events</span>
                <span className="mono text-[var(--color-text)]">{reconciliation?.manual_event_count ?? 0}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="text-[var(--color-text-muted)]">Persisted snapshot</span>
                <span className="mono text-right text-[10px] text-[var(--color-text-muted)]">
                  {persistedSnapshot}
                </span>
              </div>
            </div>
          </div>

          {/* Data integrity */}
          <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Data integrity
            </div>
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Truth score</span>
              <span className="mono text-[var(--color-text)]">{truthScore}</span>
            </div>
            {integrityChecks.length > 0 ? (
              <div className="space-y-2">
                {integrityChecks.slice(0, 4).map((check) => {
                  const status = String(check.status ?? "warn").toLowerCase();
                  const label = String(check.label ?? check.id ?? "check");
                  const message = String(check.message ?? "");
                  return (
                    <div key={`${label}:${status}`} className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg)] px-2.5 py-2">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[11px] font-medium text-[var(--color-text)]">{label}</span>
                        <StatusBadge label={status} tone={integrityTone(status)} />
                      </div>
                      <p className="mt-1 text-[11px] text-[var(--color-text-muted)]">{message}</p>
                    </div>
                  );
                })}
                <div className="grid grid-cols-3 gap-2 pt-1">
                  <MiniStat label="Pass" value={integrityCounts.pass} color="var(--color-green)" />
                  <MiniStat label="Warn" value={integrityCounts.warn} color="var(--color-amber)" />
                  <MiniStat label="Fail" value={integrityCounts.fail} color="var(--color-red)" />
                </div>
              </div>
            ) : (
              <p className="text-[11px] text-[var(--color-text-muted)]">
                No live integrity checks yet.
              </p>
            )}
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

function integrityTone(status: string): "neutral" | "accent" | "success" | "warning" | "danger" {
  const normalized = status.trim().toLowerCase();
  if (normalized === "pass") {
    return "success";
  }
  if (normalized === "warn") {
    return "warning";
  }
  if (normalized === "fail") {
    return "danger";
  }
  return "neutral";
}

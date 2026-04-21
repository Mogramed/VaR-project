"use client";

import { startTransition, useEffect, useState } from "react";

import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { deriveLiveRuntimeDiagnostics } from "@/components/app-shell/live-runtime-phase";
import { LivePostureBanner } from "@/components/app-shell/live-posture-banner";
import { LiveRuntimeBadgeGroup } from "@/components/app-shell/live-runtime-badge-group";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import {
  DealHistoryTable,
  ExecutionHistoryTable,
  HoldingsTable,
  MT5OrdersTable,
  OrderHistoryTable,
  ReconciliationTable,
} from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { CHART_PALETTE } from "@/lib/chart-options";
import { formatCurrency, formatPercent, formatTimestamp, formatTimestampWithSource } from "@/lib/utils";

type AnalyticsPoint = {
  timestamp: string;
  balance?: number | null;
  equity?: number | null;
  margin_free?: number | null;
  margin_level?: number | null;
  profit?: number | null;
  avg_spread_bps?: number | null;
  tick_age_seconds?: number | null;
};

function downloadJsonFile(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function utcLabel(timestamp: string | null | undefined) {
  if (!timestamp) return "n/a";
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.valueOf()) || parsed.getUTCFullYear() < 2000) return "n/a";
  const hh = String(parsed.getUTCHours()).padStart(2, "0");
  const mm = String(parsed.getUTCMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

function lineOption(
  labels: string[],
  series: Array<{ name: string; data: Array<number | null>; color: string; yAxisIndex?: number }>,
  yAxisNames: string[],
) {
  const yAxis = yAxisNames.map((name) => ({
    type: "value",
    name,
    nameTextStyle: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
    axisLabel: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
    splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)" } },
    axisLine: { show: false },
    axisTick: { show: false },
  }));
  return {
    animationDuration: 450,
    grid: { left: 18, right: 20, top: 26, bottom: 24, containLabel: true },
    legend: {
      top: 0,
      textStyle: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
    },
    tooltip: { trigger: "axis", confine: true },
    xAxis: {
      type: "category",
      data: labels,
      axisLabel: { color: "rgba(234,236,239,0.55)", fontSize: 10 },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
      axisTick: { show: false },
    },
    yAxis,
    series: series.map((item) => ({
      name: item.name,
      type: "line",
      smooth: 0.25,
      symbol: "none",
      yAxisIndex: item.yAxisIndex ?? 0,
      lineStyle: { width: 2, color: item.color },
      data: item.data,
    })),
  };
}

function toneForCheck(status: string) {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "pass") return "success" as const;
  if (normalized === "warn") return "warning" as const;
  if (normalized === "fail") return "danger" as const;
  return "neutral" as const;
}

function coerceNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function Mt5LiveOpsFeed() {
  const { liveState, heartbeatAt, transport, accountId } = useDeskLive();
  const ls = liveState;
  const [analyticsPoints, setAnalyticsPoints] = useState<AnalyticsPoint[]>([]);
  const analyticsRefreshMs = Math.max(
    1_000,
    Math.round((Number(ls?.poll_interval_seconds ?? 1) || 1) * 1_000),
  );

  useEffect(() => {
    const activePortfolioSlug = ls?.portfolio_slug ?? undefined;
    if (!activePortfolioSlug) return;
    let cancelled = false;
    let timer: number | null = null;

    const loop = async () => {
      try {
        const payload = await api.mt5AnalyticsSeries(activePortfolioSlug, {
          windowMinutes: 240,
          maxPoints: 240,
          accountId,
        });
        if (!cancelled) {
          startTransition(() => {
            setAnalyticsPoints(Array.isArray(payload.points) ? payload.points : []);
          });
        }
      } catch {
        // Keep previous points if endpoint is temporarily unavailable.
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(() => {
            void loop();
          }, analyticsRefreshMs);
        }
      }
    };

    void loop();
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, [accountId, analyticsRefreshMs, ls?.portfolio_slug]);

  if (ls == null) {
    return (
      <div className="desk-page space-y-4">
        <PageHeader eyebrow="MT5 Ops" title="Live telemetry from the MT5 bridge" />
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-[11px] text-[var(--color-text-muted)]">
          Waiting for the shared MT5 live feed to connect.
        </div>
      </div>
    );
  }

  const reconciliation = ls.reconciliation;
  const holdings = ls.holdings ?? [];
  const pendingOrders = ls.pending_orders ?? [];
  const orderHistory = ls.order_history ?? [];
  const dealHistory = ls.deal_history ?? [];
  const pollIntervalSeconds = Number(ls.poll_interval_seconds);
  const pollHint = Number.isFinite(pollIntervalSeconds) && pollIntervalSeconds > 0
    ? `Poll ${pollIntervalSeconds.toFixed(1)}s`
    : "Poll n/a";
  const liveExposure = ls.exposure?.gross_exposure_base_ccy
    ?? holdings.reduce((sum, item) => {
      const exposure = Number(item.signed_exposure_base_ccy);
      return sum + (Number.isFinite(exposure) ? Math.abs(exposure) : 0);
    }, 0);
  const marketReference = formatTimestampWithSource({
    source: reconciliation?.market_reference_source,
    timestamp: reconciliation?.market_reference_timestamp,
  });
  const analyticsAsOf = ls.analytics_generated_at ? formatTimestamp(ls.analytics_generated_at) : "n/a";
  const runtimeDiagnostics = deriveLiveRuntimeDiagnostics(ls, transport);
  const runtimePhase = runtimeDiagnostics.phase;
  const bridgeLabel = runtimePhase === "live"
    ? "Live"
    : runtimePhase === "recovering"
      ? runtimeDiagnostics.isRetrying
        ? "Retrying"
        : "Recovering"
      : runtimePhase === "degraded"
        ? "Delayed"
        : runtimePhase === "offline"
          ? "Offline"
          : "Pending";
  const bridgeTone = runtimePhase === "live"
    ? "success"
    : runtimePhase === "recovering"
      ? "accent"
      : runtimePhase === "degraded"
        ? "warning"
        : runtimePhase === "offline"
          ? "danger"
          : "neutral";
  const retryDelayLabel = runtimeDiagnostics.retryInSeconds == null
    ? null
    : runtimeDiagnostics.retryInSeconds >= 10
      ? `${Math.round(runtimeDiagnostics.retryInSeconds)}s`
      : `${runtimeDiagnostics.retryInSeconds.toFixed(1)}s`;
  const transportValue = transport === "stream" ? "Streaming" : transport === "polling" ? "Polling" : "Starting";
  const inspectorPayload = {
    exported_at: new Date().toISOString(),
    transport,
    live_state: ls,
    terminal_status: ls.terminal_status,
    account: ls.account,
    portfolio: {
      portfolio_slug: ls.portfolio_slug,
      portfolio_mode: ls.portfolio_mode,
      symbols: ls.symbols ?? [],
    },
    tick_quality: ls.tick_quality,
    microstructure: ls.microstructure,
    reconciliation: ls.reconciliation,
    counts: {
      holdings: holdings.length,
      pending_orders: pendingOrders.length,
      order_history: orderHistory.length,
      deal_history: dealHistory.length,
    },
  };

  const fallbackPoint: AnalyticsPoint | null = ls.account
    ? {
      timestamp: ls.account.timestamp_utc ?? ls.generated_at,
      balance: ls.account.balance,
      equity: ls.account.equity,
      margin_free: ls.account.margin_free,
      margin_level: ls.account.margin_level,
      profit: ls.account.profit,
      avg_spread_bps: coerceNumber(ls.microstructure?.avg_spread_bps),
      tick_age_seconds: null,
    }
    : null;

  const seriesPoints = analyticsPoints.length > 0
    ? analyticsPoints
    : (fallbackPoint ? [fallbackPoint] : []);
  const labels = seriesPoints.map((point) => utcLabel(point.timestamp));

  const equityBalanceOption = lineOption(
    labels,
    [
      {
        name: "Balance",
        data: seriesPoints.map((point) => point.balance ?? null),
        color: CHART_PALETTE.gold,
      },
      {
        name: "Equity",
        data: seriesPoints.map((point) => point.equity ?? null),
        color: CHART_PALETTE.green,
      },
    ],
    ["EUR"],
  );
  const marginOption = lineOption(
    labels,
    [
      {
        name: "Margin free",
        data: seriesPoints.map((point) => point.margin_free ?? null),
        color: CHART_PALETTE.blue,
        yAxisIndex: 0,
      },
      {
        name: "Margin level",
        data: seriesPoints.map((point) => point.margin_level ?? null),
        color: CHART_PALETTE.teal,
        yAxisIndex: 1,
      },
    ],
    ["EUR", "%"],
  );
  const latestPoint = seriesPoints.length > 0 ? seriesPoints[seriesPoints.length - 1] : null;
  const truthScore = ls.truth_score == null ? "n/a" : formatPercent(ls.truth_score, 0);
  const avgSpreadBps = coerceNumber(latestPoint?.avg_spread_bps ?? ls.microstructure?.avg_spread_bps ?? null);
  const tickAgeSeconds = latestPoint?.tick_age_seconds ?? null;

  return (
    <div className="desk-page space-y-4">
      <PageHeader
        eyebrow="MT5 Ops"
        title="Live telemetry from the MT5 bridge"
        aside={(
          <LiveRuntimeBadgeGroup
            liveState={ls}
            heartbeatAt={heartbeatAt}
            transport={transport}
            showFreshness
          />
        )}
      />
      <LivePostureBanner liveState={ls} transport={transport} />
      {reconciliation?.market_closed ? (
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-[11px] text-[var(--color-text-soft)]">
          Reference market snapshot: <span className="mono">{marketReference}</span>.
        </div>
      ) : null}

      <LiveOperatorAlerts alerts={ls.operator_alerts ?? []} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricBlock
          label="Bridge"
          value={bridgeLabel}
          hint={[
            `Seq ${ls.sequence}`,
            retryDelayLabel && runtimeDiagnostics.isRetrying ? `retry ${retryDelayLabel}` : null,
            runtimeDiagnostics.failureCount > 0 ? `fail ${runtimeDiagnostics.failureCount}` : null,
            formatTimestamp(ls.generated_at),
          ].filter(Boolean).join(" | ")}
          tone={bridgeTone}
        />
        <MetricBlock
          label="Transport"
          value={transportValue}
          hint={pollHint}
        />
        <MetricBlock label="Exposure" value={formatCurrency(liveExposure)} hint={`${holdings.length} positions`} tone="accent" />
        <MetricBlock
          label="Pending"
          value={String(pendingOrders.length)}
          hint={`${reconciliation?.manual_event_count ?? 0} manual`}
          tone={pendingOrders.length > 0 ? "warning" : "success"}
        />
      </section>

      {ls.account ? (
        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricBlock label="Equity" value={formatCurrency(ls.account.equity)} hint={`Balance ${formatCurrency(ls.account.balance)}`} tone="accent" />
          <MetricBlock label="Free margin" value={formatCurrency(ls.account.margin_free)} hint={ls.account.server ?? "MT5"} tone="success" />
          <MetricBlock label="Profit" value={formatCurrency(ls.account.profit, 2)} tone={ls.account.profit >= 0 ? "success" : "danger"} />
          <MetricBlock label="Updated" value={formatTimestamp(ls.account.timestamp_utc ?? ls.generated_at)} hint={`Lev ${ls.account.leverage ?? "n/a"}`} />
        </section>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartSurface
          option={equityBalanceOption}
          mode="standard"
          dataCount={seriesPoints.length}
          title="MT5 Health & Equity"
          meta={seriesPoints.length > 0 ? `${seriesPoints.length} points` : undefined}
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No analytics series available.</p>}
          insight={(
            <div className="grid gap-3">
              <MetricBlock label="Truth score" value={truthScore} hint={ls.operational_truth ? ls.operational_truth.replaceAll("_", " ") : "n/a"} className="bg-transparent" />
              <MetricBlock label="Tick age" value={tickAgeSeconds == null ? "n/a" : `${Math.round(tickAgeSeconds)}s`} hint={`Spread ${avgSpreadBps == null ? "n/a" : `${avgSpreadBps.toFixed(2)} bps`}`} className="bg-transparent" />
            </div>
          )}
        />
        <ChartSurface
          option={marginOption}
          mode="standard"
          dataCount={seriesPoints.length}
          title="Margin & Liquidity"
          meta={ls.analytics_generated_at ? `Analytics as of ${analyticsAsOf}` : undefined}
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No margin series available.</p>}
        />
      </div>

      <section className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
        <div className="mb-3 flex items-center justify-between gap-2">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Data Integrity
          </div>
          <StatusBadge label={truthScore} tone={ls.truth_score != null && ls.truth_score >= 0.8 ? "success" : "warning"} />
        </div>
        {(ls.quality_checks ?? []).length === 0 ? (
          <p className="text-xs text-[var(--color-text-muted)]">No quality checks reported by backend.</p>
        ) : (
          <div className="grid gap-2 md:grid-cols-2">
            {(ls.quality_checks ?? []).map((check: Record<string, unknown>, index: number) => (
              <div
                key={`${String(check.id ?? index)}:${String(check.status ?? "")}`}
                className="rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface-strong)] p-2.5"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-semibold text-[var(--color-text)]">
                    {String(check.label ?? check.id ?? "check")}
                  </span>
                  <StatusBadge label={String(check.status ?? "unknown")} tone={toneForCheck(String(check.status ?? ""))} />
                </div>
                <p className="mt-1 text-[11px] text-[var(--color-text-muted)]">{String(check.message ?? "")}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
        <div className="mb-3 flex items-center justify-between gap-2">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            MT5 Inspector
          </div>
          <button
            type="button"
            onClick={() => downloadJsonFile(`mt5-inspector-${ls.portfolio_slug ?? "portfolio"}-${Date.now()}.json`, inspectorPayload)}
            className="rounded-[var(--radius-sm)] border border-[var(--color-border)] px-2 py-1 text-[11px] text-[var(--color-text-soft)] transition-colors hover:border-[var(--color-border-strong)]"
          >
            Export JSON
          </button>
        </div>
        <div className="grid gap-2 text-xs md:grid-cols-2 xl:grid-cols-4">
          <div className="flex items-center justify-between"><span className="text-[var(--color-text-muted)]">Tick quality</span><span className="mono">{String(ls.tick_quality?.status ?? "n/a")}</span></div>
          <div className="flex items-center justify-between"><span className="text-[var(--color-text-muted)]">Regime</span><span className="mono">{String(ls.microstructure?.regime ?? "n/a")}</span></div>
          <div className="flex items-center justify-between"><span className="text-[var(--color-text-muted)]">Analytics as of</span><span className="mono">{analyticsAsOf}</span></div>
          <div className="flex items-center justify-between"><span className="text-[var(--color-text-muted)]">Analytics stale</span><span className="mono">{ls.analytics_stale ? "yes" : "no"}</span></div>
        </div>
        <pre className="mt-3 max-h-52 overflow-auto rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface-strong)] p-2 text-[10px] text-[var(--color-text-soft)]">
          {JSON.stringify(
            {
              terminal_status: ls.terminal_status,
              account: ls.account,
              ticks: ls.ticks,
              reconciliation: ls.reconciliation,
            },
            null,
            2,
          )}
        </pre>
      </section>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Holdings</h4>
          <HoldingsTable rows={holdings} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Pending orders</h4>
          <MT5OrdersTable rows={pendingOrders} />
        </div>
      </div>

      <div className="space-y-2">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Reconciliation</h4>
        <ReconciliationTable rows={reconciliation?.mismatches ?? []} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Order blotter</h4>
          <OrderHistoryTable rows={orderHistory} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Deal blotter</h4>
          <DealHistoryTable rows={dealHistory} />
        </div>
      </div>

      <div className="space-y-2">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Execution trail</h4>
        <ExecutionHistoryTable rows={reconciliation?.recent_execution_attempts ?? []} />
      </div>
    </div>
  );
}

"use client";

import { DealHistoryTable, ExecutionHistoryTable, HoldingsTable, MT5OrdersTable, OrderHistoryTable, ReconciliationTable } from "@/components/data/risk-tables";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import type { MT5LiveStateResponse } from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatTimestamp } from "@/lib/utils";

export function Mt5LiveOpsFeed({ initialState }: { initialState: MT5LiveStateResponse }) {
  const { liveState: maybe, transport } = useMt5LiveState(initialState.portfolio_slug ?? undefined, initialState);
  const ls = maybe ?? initialState;

  const reconciliation = ls.reconciliation;
  const holdings = ls.holdings ?? [];
  const pendingOrders = ls.pending_orders ?? [];
  const orderHistory = ls.order_history ?? [];
  const dealHistory = ls.deal_history ?? [];
  const liveExposure = ls.exposure?.gross_exposure_base_ccy ?? holdings.reduce((s, i) => s + Math.abs(i.signed_exposure_base_ccy), 0);

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="MT5 Ops" title="Live telemetry from the MT5 bridge"
        aside={<>
          <StatusBadge label={ls.status === "ok" ? "Live" : ls.status} tone={ls.status === "ok" ? "success" : "warning"} />
          <StatusBadge label={transport === "stream" ? "SSE" : transport} tone={transport === "stream" ? "success" : "warning"} />
        </>}
      />

      {ls.last_error ? (
        <div className="rounded-[var(--radius-md)] border border-[var(--color-amber)]/20 bg-[var(--color-amber-soft)] px-3 py-2 text-[11px] text-[var(--color-text-soft)]">{ls.last_error}</div>
      ) : null}

      <LiveOperatorAlerts alerts={ls.operator_alerts ?? []} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricBlock label="Bridge" value={ls.connected ? "Connected" : "Degraded"} hint={`Seq ${ls.sequence} · ${formatTimestamp(ls.generated_at)}`} tone={ls.status === "ok" ? "success" : "warning"} />
        <MetricBlock label="Transport" value={transport === "stream" ? "Streaming" : transport === "polling" ? "Polling" : "Starting"} hint={`Poll ${ls.poll_interval_seconds.toFixed(1)}s`} />
        <MetricBlock label="Exposure" value={formatCurrency(liveExposure)} hint={`${holdings.length} positions`} tone="accent" />
        <MetricBlock label="Pending" value={String(pendingOrders.length)} hint={`${reconciliation?.manual_event_count ?? 0} manual`} tone={pendingOrders.length > 0 ? "warning" : "success"} />
      </section>

      {ls.account ? (
        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricBlock label="Equity" value={formatCurrency(ls.account.equity)} hint={`Balance ${formatCurrency(ls.account.balance)}`} tone="accent" />
          <MetricBlock label="Free margin" value={formatCurrency(ls.account.margin_free)} hint={ls.account.server ?? "MT5"} tone="success" />
          <MetricBlock label="Profit" value={formatCurrency(ls.account.profit, 2)} tone={ls.account.profit >= 0 ? "success" : "danger"} />
          <MetricBlock label="Updated" value={formatTimestamp(ls.account.timestamp_utc ?? ls.generated_at)} hint={`Lev ${ls.account.leverage ?? "n/a"}`} />
        </section>
      ) : null}

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

"use client";

import {
  DealHistoryTable,
  ExecutionHistoryTable,
  HoldingsTable,
  MT5OrdersTable,
  OrderHistoryTable,
  ReconciliationTable,
} from "@/components/data/risk-tables";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import type { MT5LiveStateResponse } from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatTimestamp } from "@/lib/utils";

export function Mt5LiveOpsFeed({
  initialState,
}: {
  initialState: MT5LiveStateResponse;
}) {
  const { liveState: maybeLiveState, transport } = useMt5LiveState(
    initialState.portfolio_slug ?? undefined,
    initialState,
  );
  const liveState = maybeLiveState ?? initialState;

  const reconciliation = liveState.reconciliation;
  const holdings = liveState.holdings ?? [];
  const pendingOrders = liveState.pending_orders ?? [];
  const orderHistory = liveState.order_history ?? [];
  const dealHistory = liveState.deal_history ?? [];
  const liveExposure =
    liveState.exposure?.gross_exposure_base_ccy ??
    holdings.reduce((sum, item) => sum + Math.abs(item.signed_exposure_base_ccy), 0);
  const transportTone =
    transport === "stream" ? "success" : transport === "polling" ? "warning" : "neutral";
  const bridgeTone = liveState.status === "ok" ? "success" : liveState.degraded ? "warning" : "neutral";

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="MT5 Ops"
        title="Operate the desk from MT5 telemetry, with a live bridge instead of one-shot snapshots."
        description="The feed now stays attached to the MT5 bridge, streams cached account telemetry into the desk, and keeps holdings, blotter and reconciliation aligned with the same live source."
        aside={
          <div className="flex flex-wrap items-center gap-3">
            <StatusBadge
              label={liveState.status === "ok" ? "Bridge live" : liveState.status}
              tone={bridgeTone}
            />
            <StatusBadge
              label={transport === "stream" ? "SSE stream" : transport === "polling" ? "Polling fallback" : "Connecting"}
              tone={transportTone}
            />
          </div>
        }
      />

      {liveState.last_error ? (
        <div className="rounded-[1.6rem] border border-amber-300/16 bg-amber-300/8 px-5 py-5 text-sm leading-7 text-[var(--color-text-soft)]">
          {liveState.last_error}
        </div>
      ) : null}

      <LiveOperatorAlerts
        alerts={liveState.operator_alerts ?? []}
        title="Ops watchlist"
        copy="Bridge health, manual MT5 activity, pending broker actions and reconciliation drift are derived directly from the live feed."
      />

      <section className="grid gap-4 xl:grid-cols-4">
        <MetricBlock
          label="Bridge"
          value={liveState.connected ? "Connected" : "Degraded"}
          hint={`Seq ${liveState.sequence} · ${formatTimestamp(liveState.generated_at)}`}
          tone={bridgeTone}
        />
        <MetricBlock
          label="Feed mode"
          value={transport === "stream" ? "Streaming" : transport === "polling" ? "Polling" : "Starting"}
          hint={`Poll ${liveState.poll_interval_seconds.toFixed(1)}s · History ${liveState.history_poll_interval_seconds.toFixed(0)}s`}
          tone={transportTone}
        />
        <MetricBlock
          label="Live exposure"
          value={formatCurrency(liveExposure)}
          hint={`${holdings.length} open positions`}
          tone="accent"
        />
        <MetricBlock
          label="Pending orders"
          value={String(pendingOrders.length)}
          hint={`${reconciliation?.manual_event_count ?? 0} manual MT5 events`}
          tone={pendingOrders.length > 0 ? "warning" : "success"}
        />
      </section>

      {liveState.account ? (
        <section className="grid gap-4 xl:grid-cols-4">
          <MetricBlock
            label="Equity"
            value={formatCurrency(liveState.account.equity)}
            hint={`Balance ${formatCurrency(liveState.account.balance)}`}
            tone="accent"
          />
          <MetricBlock
            label="Free margin"
            value={formatCurrency(liveState.account.margin_free)}
            hint={liveState.account.server ?? "MT5 account"}
            tone="success"
          />
          <MetricBlock
            label="Profit"
            value={formatCurrency(liveState.account.profit, 2)}
            hint={liveState.account.name ?? "Account"}
            tone={liveState.account.profit >= 0 ? "success" : "danger"}
          />
          <MetricBlock
            label="Updated"
            value={formatTimestamp(liveState.account.timestamp_utc ?? liveState.generated_at)}
            hint={`Leverage ${liveState.account.leverage ?? "n/a"}`}
          />
        </section>
      ) : null}

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Live holdings
          </div>
          <HoldingsTable rows={holdings} />
        </div>
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Pending orders
          </div>
          <MT5OrdersTable rows={pendingOrders} />
        </div>
      </section>

      <section className="space-y-3">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Intraday reconciliation
        </div>
        <ReconciliationTable rows={reconciliation?.mismatches ?? []} />
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Order blotter
          </div>
          <OrderHistoryTable rows={orderHistory} />
        </div>
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Deal blotter
          </div>
          <DealHistoryTable rows={dealHistory} />
        </div>
      </section>

      <section className="space-y-3">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Desk execution trail
        </div>
        <ExecutionHistoryTable rows={reconciliation?.recent_execution_attempts ?? []} />
      </section>
    </div>
  );
}

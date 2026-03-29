import { PageHeader } from "@/components/app-shell/page-header";
import {
  HoldingsTable,
  MT5OrdersTable,
  ReconciliationTable,
} from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { formatCurrency, formatTimestamp } from "@/lib/utils";

export default async function DeskMt5OpsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const health = await api.health();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;

  const [status, account, marketStatus, reconciliation] = await Promise.all([
    api.mt5Status(),
    api.mt5Account().catch(() => null),
    api.marketDataStatus(resolvedPortfolio).catch(() => null),
    api.reconciliationSummary(resolvedPortfolio).catch(() => null),
  ]);

  const holdings = reconciliation?.holdings ?? [];
  const pendingOrders = marketStatus?.pending_orders ?? [];
  const mismatches = reconciliation?.mismatches ?? [];
  const liveExposure = holdings.reduce(
    (sum, item) => sum + Math.abs(item.signed_position_eur),
    0,
  );

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="MT5 Ops"
        title="Operate the desk from MT5 telemetry, not from fixture-era assumptions."
        description="This surface now fuses terminal health, synchronized market data, live holdings and intraday reconciliation so the operator can see whether the desk and MT5 still agree."
        aside={
          <StatusBadge
            label={status.ready ? "MT5 ready" : "Guarded"}
            tone={status.ready ? "success" : "warning"}
          />
        }
      />

      {!status.ready ? (
        <div className="rounded-[1.6rem] border border-amber-300/16 bg-amber-300/8 px-5 py-5 text-sm leading-7 text-[var(--color-text-soft)]">
          {status.message}
        </div>
      ) : null}

      <section className="grid gap-4 xl:grid-cols-4">
        <MetricBlock
          label="Terminal"
          value={status.connected ? "Connected" : "Offline"}
          hint={status.company ?? "MetaTrader 5"}
          tone={status.connected ? "success" : "danger"}
        />
        <MetricBlock
          label="Market data"
          value={marketStatus?.status ?? "unknown"}
          hint={marketStatus?.latest_sync_at ? formatTimestamp(marketStatus.latest_sync_at) : "No MT5 sync yet"}
          tone={marketStatus?.status === "ok" ? "success" : "warning"}
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

      {account ? (
        <section className="grid gap-4 xl:grid-cols-4">
          <MetricBlock label="Equity" value={formatCurrency(account.equity)} hint={`Balance ${formatCurrency(account.balance)}`} tone="accent" />
          <MetricBlock label="Free margin" value={formatCurrency(account.margin_free)} hint={account.server ?? "MT5 account"} tone="success" />
          <MetricBlock label="Profit" value={formatCurrency(account.profit, 2)} hint={account.name ?? "Account"} tone={account.profit >= 0 ? "success" : "danger"} />
          <MetricBlock label="Updated" value={formatTimestamp(account.timestamp_utc)} hint={`Leverage ${account.leverage ?? "n/a"}`} />
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
        <ReconciliationTable rows={mismatches} />
      </section>
    </div>
  );
}

import { PageHeader } from "@/components/app-shell/page-header";
import {
  DealHistoryTable,
  ExecutionHistoryTable,
  OrderHistoryTable,
  ReconciliationTable,
} from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { countManualMt5Events } from "@/lib/view-models";

export default async function DeskBlotterPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const resolvedPortfolio = portfolioSlug ?? (await api.health()).portfolio_slug;

  const [orders, deals, executions, reconciliation] = await Promise.all([
    api.mt5HistoryOrders(resolvedPortfolio, 40).catch(() => []),
    api.mt5HistoryDeals(resolvedPortfolio, 40).catch(() => []),
    api.recentExecutionResults(resolvedPortfolio, 20).catch(() => []),
    api.reconciliationSummary(resolvedPortfolio).catch(() => null),
  ]);

  const manual = countManualMt5Events(orders, deals);

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="MT5 Blotter"
        title="Audit what MT5 actually did, not just what the desk intended."
        description="The former simulation surface is now a real operator blotter: order history, deal history, desk execution attempts and reconciliation mismatches live side by side."
        aside={
          <StatusBadge
            label={reconciliation?.market_data_status ?? "unknown"}
            tone={reconciliation?.market_data_status === "ok" ? "success" : "warning"}
          />
        }
      />

      <section className="grid gap-4 xl:grid-cols-4">
        <MetricBlock
          label="Orders"
          value={String(orders.length)}
          hint={`${manual.orders} manual`}
          tone="accent"
        />
        <MetricBlock
          label="Deals"
          value={String(deals.length)}
          hint={`${manual.deals} manual`}
          tone="warning"
        />
        <MetricBlock
          label="Desk attempts"
          value={String(executions.length)}
          hint={`${reconciliation?.unmatched_execution_count ?? 0} unmatched`}
          tone="success"
        />
        <MetricBlock
          label="Mismatches"
          value={String(
            (reconciliation?.mismatches ?? []).filter((item) => item.status !== "match")
              .length,
          )}
          hint="Desk vs MT5 position drift"
          tone={
            (reconciliation?.mismatches ?? []).some((item) => item.status !== "match")
              ? "warning"
              : "success"
          }
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Order history
          </div>
          <OrderHistoryTable rows={orders} />
        </div>
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Deal history
          </div>
          <DealHistoryTable rows={deals} />
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Desk execution attempts
          </div>
          <ExecutionHistoryTable rows={executions} />
        </div>
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Position reconciliation
          </div>
          <ReconciliationTable rows={reconciliation?.mismatches ?? []} />
        </div>
      </section>
    </div>
  );
}

"use client";

import { useCallback } from "react";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { DealHistoryTable, ExecutionFillsTable, ExecutionHistoryTable, OrderHistoryTable, ReconciliationTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { ExecutionFillResponse, ExecutionResultResponse, MT5LiveStateResponse } from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { useRecentExecutionActivity } from "@/lib/use-recent-execution-activity";
import { formatTimestamp } from "@/lib/utils";
import { countManualMt5Events } from "@/lib/view-models";

export function Mt5BlotterLiveSurface({
  portfolioSlug, initialLiveState, initialExecutions, initialFills,
}: { portfolioSlug: string; initialLiveState: MT5LiveStateResponse | null; initialExecutions: ExecutionResultResponse[]; initialFills: ExecutionFillResponse[] }) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const { executions, fills } = useRecentExecutionActivity({ portfolioSlug, initialExecutions, initialFills, liveSequence: liveState?.sequence, executionLimit: 20, fillLimit: 20 });
  const orders = liveState?.order_history ?? [];
  const deals = liveState?.deal_history ?? [];
  const reconciliation = liveState?.reconciliation ?? null;
  const manual = countManualMt5Events(orders, deals);
  const mismatchCount = (reconciliation?.mismatches ?? []).filter((i) => i.status !== "match").length;

  const handleAck = useCallback((symbol: string) => {
    api.acknowledgeReconciliation({ portfolio_slug: portfolioSlug, symbol, reason: "operator_acknowledged", operator_note: "" }).catch(() => null);
  }, [portfolioSlug]);

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Blotter" title="MT5 orders, deals, fills and reconciliation"
        aside={<>
          <StatusBadge label={liveState?.status ?? "unknown"} tone={liveState?.status === "ok" ? "success" : "warning"} />
          <StatusBadge label={transport} tone={transport === "stream" ? "success" : "warning"} />
        </>}
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricBlock label="Orders" value={String(orders.length)} hint={`${manual.orders} manual`} tone="accent" />
        <MetricBlock label="Deals" value={String(deals.length)} hint={`${manual.deals} manual`} tone="warning" />
        <MetricBlock label="Desk attempts" value={String(executions.length)} tone="success" />
        <MetricBlock label="Fills" value={String(fills.length)} hint={liveState?.generated_at ? formatTimestamp(liveState.generated_at) : undefined} />
        <MetricBlock label="Mismatches" value={String(mismatchCount)} tone={mismatchCount > 0 ? "warning" : "success"} />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Order history</h4>
          <OrderHistoryTable rows={orders} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Deal history</h4>
          <DealHistoryTable rows={deals} />
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Execution attempts</h4>
          <ExecutionHistoryTable rows={executions} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Reconciliation</h4>
          <ReconciliationTable rows={reconciliation?.mismatches ?? []} onAcknowledge={handleAck} />
        </div>
      </div>

      <div className="space-y-2">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Execution fills</h4>
        <ExecutionFillsTable rows={fills} />
      </div>
    </div>
  );
}

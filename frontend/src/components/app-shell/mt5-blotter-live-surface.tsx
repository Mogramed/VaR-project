"use client";

import { useCallback } from "react";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import {
  DealHistoryTable,
  ExecutionFillsTable,
  ExecutionHistoryTable,
  OrderHistoryTable,
  ReconciliationTable,
} from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type {
  ExecutionFillResponse,
  ExecutionResultResponse,
  MT5LiveStateResponse,
} from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { useRecentExecutionActivity } from "@/lib/use-recent-execution-activity";
import { formatTimestamp } from "@/lib/utils";
import { countManualMt5Events } from "@/lib/view-models";

export function Mt5BlotterLiveSurface({
  portfolioSlug,
  initialLiveState,
  initialExecutions,
  initialFills,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialExecutions: ExecutionResultResponse[];
  initialFills: ExecutionFillResponse[];
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const { executions, fills } = useRecentExecutionActivity({
    portfolioSlug,
    initialExecutions,
    initialFills,
    liveSequence: liveState?.sequence,
    executionLimit: 20,
    fillLimit: 20,
  });
  const orders = liveState?.order_history ?? [];
  const deals = liveState?.deal_history ?? [];
  const reconciliation = liveState?.reconciliation ?? null;
  const manual = countManualMt5Events(orders, deals);
  const mismatchCount = (reconciliation?.mismatches ?? []).filter((item) => item.status !== "match").length;

  const handleAcknowledge = useCallback(
    (symbol: string) => {
      api
        .acknowledgeReconciliation({
          portfolio_slug: portfolioSlug,
          symbol,
          reason: "operator_acknowledged",
          operator_note: "",
        })
        .catch(() => null);
    },
    [portfolioSlug],
  );

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="MT5 Blotter"
        title="Audit what MT5 actually did, with bridge-driven reconciliation and fills."
        description="Orders, deals, desk attempts, fills and reconciliation now refresh from the MT5 live bridge so the blotter reflects broker reality instead of a static page load."
        aside={
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge
              label={reconciliation?.market_data_status ?? liveState?.status ?? "unknown"}
              tone={(reconciliation?.market_data_status ?? liveState?.status) === "ok" ? "success" : "warning"}
            />
            <StatusBadge
              label={transport === "stream" ? "stream" : transport === "polling" ? "polling" : "connecting"}
              tone={transport === "stream" ? "success" : transport === "polling" ? "warning" : "neutral"}
            />
          </div>
        }
      />

      <section className="grid gap-4 xl:grid-cols-5">
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
          label="Recent fills"
          value={String(fills.length)}
          hint={liveState?.generated_at ? `Live ${formatTimestamp(liveState.generated_at)}` : "Awaiting bridge"}
          tone="accent"
        />
        <MetricBlock
          label="Mismatches"
          value={String(mismatchCount)}
          hint="Desk vs MT5 drift"
          tone={mismatchCount > 0 ? "warning" : "success"}
        />
      </section>

      <LiveOperatorAlerts
        alerts={liveState?.operator_alerts ?? []}
        title="Blotter watchlist"
        copy="These alerts highlight manual broker activity, stale telemetry, unmatched desk attempts and drift already visible in the live blotter."
      />

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
          <ReconciliationTable rows={reconciliation?.mismatches ?? []} onAcknowledge={handleAcknowledge} />
        </div>
      </section>

      <section className="space-y-3">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Execution fills
        </div>
        <ExecutionFillsTable rows={fills} />
      </section>
    </div>
  );
}

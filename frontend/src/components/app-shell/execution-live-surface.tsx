"use client";

import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { ExecutionFillsTable, ExecutionHistoryTable } from "@/components/data/risk-tables";
import { ExecutionPanel } from "@/components/forms/execution-panel";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import type { ExecutionFillResponse, ExecutionResultResponse, MT5LiveStateResponse, MT5TerminalStatusResponse } from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { useRecentExecutionActivity } from "@/lib/use-recent-execution-activity";


export function ExecutionLiveSurface({
  portfolioSlug, initialLiveState, initialTerminalStatus, initialExecutions, initialFills, initialSymbol, initialExposureChange, initialSide,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialTerminalStatus: MT5TerminalStatusResponse;
  initialExecutions: ExecutionResultResponse[];
  initialFills: ExecutionFillResponse[];
  initialSymbol?: string;
  initialExposureChange?: number | null;
  initialSide?: "buy" | "sell";
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const { executions, fills, pushExecutionResult } = useRecentExecutionActivity({
    portfolioSlug, initialExecutions, initialFills, liveSequence: liveState?.sequence, executionLimit: 12, fillLimit: 12,
  });
  const status = liveState?.terminal_status ?? initialTerminalStatus;
  const executedCount = executions.filter((i) => ["EXECUTED", "PLACED"].includes(i.status)).length;
  const blockedCount = executions.filter((i) => ["BLOCKED", "REJECTED", "FAILED"].includes(i.status)).length;

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Execution" title="Pre-trade dry run with MT5 routing"
        aside={<>
          <StatusBadge label={status.ready ? "MT5 ready" : "Guarded"} tone={status.ready ? "success" : "warning"} />
          <StatusBadge label={transport} tone={transport === "stream" ? "success" : "warning"} />
        </>}
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricBlock label="Terminal" value={status.connected ? "Connected" : "Offline"} tone={status.connected ? "success" : "danger"} />
        <MetricBlock label="Ready" value={status.ready ? "Yes" : "No"} tone={status.ready ? "success" : "warning"} />
        <MetricBlock label="Executed" value={String(executedCount)} tone="success" />
        <MetricBlock label="Blocked" value={String(blockedCount)} tone={blockedCount > 0 ? "warning" : "accent"} />
        <MetricBlock label="Live alerts" value={String((liveState?.operator_alerts ?? []).length)} tone={(liveState?.operator_alerts ?? []).length > 0 ? "warning" : "success"} />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} title="Execution alerts" />
      <ExecutionPanel
        portfolioSlug={portfolioSlug}
        terminalStatus={status}
        onSubmitted={pushExecutionResult}
        initialSymbol={initialSymbol}
        initialExposureChange={initialExposureChange}
        initialSide={initialSide}
      />

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Execution attempts</h4>
          <ExecutionHistoryTable rows={executions} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Recent fills</h4>
          <ExecutionFillsTable rows={fills} />
        </div>
      </section>
    </div>
  );
}

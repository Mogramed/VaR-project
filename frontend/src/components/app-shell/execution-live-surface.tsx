"use client";

import { useMemo } from "react";

import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import {
  ExecutionFillsTable,
  ExecutionHistoryTable,
} from "@/components/data/risk-tables";
import { ExecutionPanel } from "@/components/forms/execution-panel";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import type {
  ExecutionFillResponse,
  ExecutionResultResponse,
  MT5LiveStateResponse,
  MT5TerminalStatusResponse,
} from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { useRecentExecutionActivity } from "@/lib/use-recent-execution-activity";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

export function ExecutionLiveSurface({
  portfolioSlug,
  initialLiveState,
  initialTerminalStatus,
  initialExecutions,
  initialFills,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialTerminalStatus: MT5TerminalStatusResponse;
  initialExecutions: ExecutionResultResponse[];
  initialFills: ExecutionFillResponse[];
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const { executions, fills, pushExecutionResult } = useRecentExecutionActivity({
    portfolioSlug,
    initialExecutions,
    initialFills,
    liveSequence: liveState?.sequence,
    executionLimit: 12,
    fillLimit: 12,
  });
  const status = liveState?.terminal_status ?? initialTerminalStatus;
  const operatorAlerts = liveState?.operator_alerts ?? [];
  const latestExecution = executions[0] ?? null;
  const executedCount = executions.filter((item) => ["EXECUTED", "PLACED"].includes(item.status)).length;
  const blockedCount = executions.filter((item) => ["BLOCKED", "REJECTED", "FAILED"].includes(item.status)).length;
  const liveAlertCount = operatorAlerts.length;
  const liveAlertTone =
    liveAlertCount === 0
      ? "success"
      : operatorAlerts.some((item) => item.severity.toLowerCase().includes("breach"))
        ? "danger"
        : "warning";
  const routingHint = useMemo(() => {
    if (!latestExecution) {
      return "No execution has been routed yet.";
    }
    const brokerStatus = latestExecution.broker_status ?? latestExecution.status;
    const fillRatio =
      latestExecution.fill_ratio == null ? "n/a" : formatPercent(latestExecution.fill_ratio, 0);
    return `${brokerStatus} · fill ${fillRatio}`;
  }, [latestExecution]);

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Pre-Trade Dry Run"
        title="Preview the MT5 order path, then watch the routing state update from the live bridge."
        description="The dry-run panel now stays attached to the MT5 live feed: recent submissions, fills and routing drift update from bridge events instead of waiting for a full page reload."
        aside={
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge
              label={status.ready ? "MT5 ready" : "MT5 guarded"}
              tone={status.ready ? "success" : "warning"}
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
          label="Terminal"
          value={status.connected ? "Connected" : "Offline"}
          hint={liveState?.generated_at ? `Live ${formatTimestamp(liveState.generated_at)}` : status.message}
          tone={status.connected ? "success" : "danger"}
        />
        <MetricBlock
          label="Ready to submit"
          value={status.ready ? "Yes" : "No"}
          hint={status.execution_enabled ? "Kill switch on" : "Kill switch off"}
          tone={status.ready ? "success" : "warning"}
        />
        <MetricBlock
          label="Executed"
          value={String(executedCount)}
          hint="Recent MT5 submissions"
          tone="success"
        />
        <MetricBlock
          label="Blocked / failed"
          value={String(blockedCount)}
          hint="Guard or terminal prevented execution"
          tone={blockedCount > 0 ? "warning" : "accent"}
        />
        <MetricBlock
          label="Live operator alerts"
          value={String(liveAlertCount)}
          hint={liveState ? `Bridge seq ${liveState.sequence}` : "Waiting for live bridge"}
          tone={liveAlertTone}
        />
      </section>

      <LiveOperatorAlerts
        alerts={operatorAlerts}
        title="Execution watchlist"
        copy="Bridge health, manual MT5 activity, partial fills and broker drift are surfaced here while you route orders."
      />

      <ExecutionPanel
        portfolioSlug={portfolioSlug}
        terminalStatus={status}
        onSubmitted={pushExecutionResult}
      />

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Recent execution attempts
          </div>
          <ExecutionHistoryTable rows={executions} />
        </div>

        <div className="space-y-6">
          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Latest routing event
            </div>
            {latestExecution ? (
              <div className="mt-5 space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-lg font-semibold text-white">{latestExecution.symbol}</div>
                    <div className="mt-1 text-sm text-[var(--color-text-soft)]">
                      {formatTimestamp(latestExecution.created_at ?? latestExecution.time_utc)}
                    </div>
                  </div>
                  <StatusBadge label={latestExecution.status} />
                </div>
                <MetricBlock
                  label="Executed exposure"
                  value={formatCurrency(latestExecution.executed_exposure_change)}
                  hint={`Approved ${formatCurrency(latestExecution.approved_exposure_change)}`}
                  className="bg-transparent"
                />
                <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4 text-sm leading-7 text-[var(--color-text-soft)]">
                  Guard verdict {latestExecution.guard.decision} with {latestExecution.guard.volume_lots.toFixed(2)} lots on{" "}
                  {latestExecution.guard.model_used.toUpperCase()}. {routingHint}
                </div>
              </div>
            ) : (
              <div className="mt-5 text-sm text-[var(--color-text-muted)]">
                No execution has been routed yet.
              </div>
            )}
          </div>

          <div className="space-y-3">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Recent fills
            </div>
            <ExecutionFillsTable rows={fills} />
          </div>
        </div>
      </section>
    </div>
  );
}

import { PageHeader } from "@/components/app-shell/page-header";
import { ExecutionHistoryTable } from "@/components/data/risk-tables";
import { ExecutionPanel } from "@/components/forms/execution-panel";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { formatCurrency, formatTimestamp } from "@/lib/utils";

export default async function DeskExecutionPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const health = await api.health();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;

  const [status, recentExecutions] = await Promise.all([
    api.mt5Status(),
    api.recentExecutionResults(resolvedPortfolio, 12).catch(() => []),
  ]);

  const latestExecution = recentExecutions[0] ?? null;
  const executedCount = recentExecutions.filter((item) =>
    ["EXECUTED", "PLACED"].includes(item.status),
  ).length;
  const blockedCount = recentExecutions.filter((item) =>
    ["BLOCKED", "REJECTED", "FAILED"].includes(item.status),
  ).length;

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Pre-Trade Dry Run"
        title="Preview the MT5 order path before you actually submit."
        description="This page keeps the dry-run workflow focused: convert desk notional into broker lots, inspect the guard verdict, then hand off to MT5 Ops only when the route is clean."
        aside={
          <StatusBadge
            label={status.ready ? "MT5 ready" : "MT5 guarded"}
            tone={status.ready ? "success" : "warning"}
          />
        }
      />

      <section className="grid gap-4 xl:grid-cols-4">
        <MetricBlock
          label="Terminal"
          value={status.connected ? "Connected" : "Offline"}
          hint={status.message}
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
      </section>

      <ExecutionPanel portfolioSlug={resolvedPortfolio} terminalStatus={status} />

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Recent execution attempts
          </div>
          <ExecutionHistoryTable rows={recentExecutions} />
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
                    <div className="text-lg font-semibold text-white">
                      {latestExecution.symbol}
                    </div>
                    <div className="mt-1 text-sm text-[var(--color-text-soft)]">
                      {formatTimestamp(latestExecution.created_at ?? latestExecution.time_utc)}
                    </div>
                  </div>
                  <StatusBadge label={latestExecution.status} />
                </div>
                <MetricBlock
                  label="Executed notional"
                  value={formatCurrency(latestExecution.executed_delta_position_eur)}
                  hint={`Approved ${formatCurrency(latestExecution.approved_delta_position_eur)}`}
                  className="bg-transparent"
                />
                <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4 text-sm leading-7 text-[var(--color-text-soft)]">
                  Guard verdict {latestExecution.guard.decision} with {latestExecution.guard.volume_lots.toFixed(2)} lots on {latestExecution.guard.model_used.toUpperCase()}.
                </div>
              </div>
            ) : (
              <div className="mt-5 text-sm text-[var(--color-text-muted)]">
                No execution has been routed yet.
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

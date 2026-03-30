"use client";

import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import type {
  CapitalUsageSnapshotResponse,
  MT5LiveStateResponse,
} from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

function firstNumericValue(values: Record<string, number> | undefined) {
  if (!values) {
    return null;
  }
  const first = Object.values(values)[0];
  return typeof first === "number" ? first : null;
}

export function OverviewLiveStrip({
  portfolioSlug,
  initialLiveState,
  initialCapital,
  fallbackSelectedModel,
  fallbackVarValue,
  fallbackEsValue,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialCapital: CapitalUsageSnapshotResponse | null;
  fallbackSelectedModel: string;
  fallbackVarValue: number;
  fallbackEsValue: number;
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  return (
    <OverviewLiveStripPanel
      liveState={liveState}
      transport={transport}
      initialCapital={initialCapital}
      fallbackSelectedModel={fallbackSelectedModel}
      fallbackVarValue={fallbackVarValue}
      fallbackEsValue={fallbackEsValue}
    />
  );
}

export function OverviewLiveStripPanel({
  liveState,
  transport,
  initialCapital,
  fallbackSelectedModel,
  fallbackVarValue,
  fallbackEsValue,
}: {
  liveState: MT5LiveStateResponse | null;
  transport: "stream" | "polling" | "connecting";
  initialCapital: CapitalUsageSnapshotResponse | null;
  fallbackSelectedModel: string;
  fallbackVarValue: number;
  fallbackEsValue: number;
}) {
  const riskSummary = liveState?.risk_summary ?? null;
  const capital = liveState?.capital_usage ?? initialCapital;
  const reconciliation = liveState?.reconciliation ?? null;
  const selectedModel =
    riskSummary?.reference_model ?? fallbackSelectedModel;
  const varValue =
    riskSummary?.var?.[selectedModel] ??
    firstNumericValue(riskSummary?.var) ??
    fallbackVarValue;
  const esValue =
    riskSummary?.es?.[selectedModel] ??
    firstNumericValue(riskSummary?.es) ??
    fallbackEsValue;

  return (
    <div className="space-y-6">
      <section className="grid gap-4 xl:grid-cols-4">
        <MetricBlock
          label={`VaR / ${selectedModel.toUpperCase()}`}
          value={formatCurrency(varValue)}
          hint={
            riskSummary?.latest_observation
              ? `Live sample ${formatTimestamp(riskSummary.latest_observation)}`
              : "Current portfolio risk"
          }
          tone="accent"
        />
        <MetricBlock
          label={`ES / ${selectedModel.toUpperCase()}`}
          value={formatCurrency(esValue)}
          hint={
            riskSummary?.sample_size != null
              ? `${riskSummary.sample_size} observations`
              : "Tail loss expectation"
          }
          tone="warning"
        />
        <MetricBlock
          label="Capital headroom"
          value={capital ? formatPercent(capital.headroom_ratio ?? 0, 0) : "n/a"}
          hint={
            capital
              ? `Remaining ${formatCurrency(capital.total_capital_remaining_eur)}`
              : "No capital snapshot yet"
          }
          tone="success"
        />
        <MetricBlock
          label="MT5 live"
          value={(liveState?.status ?? "pending").toUpperCase()}
          hint={
            liveState?.generated_at
              ? `${transport} ${formatTimestamp(liveState.generated_at)}`
              : "Waiting for bridge"
          }
          tone={
            liveState?.status === "ok"
              ? "success"
              : liveState?.degraded || liveState?.stale
                ? "warning"
                : "accent"
          }
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
        <LiveOperatorAlerts
          alerts={liveState?.operator_alerts ?? []}
          title="Overview watchlist"
          copy="The desk overview now tracks bridge health, live risk budget pressure, manual MT5 activity and reconciliation drift."
        />

        <div className="surface rounded-[1.7rem] p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
                Live posture
              </div>
              <div className="mt-3 text-2xl font-semibold text-white">
                {selectedModel.toUpperCase()}
              </div>
              <div className="mt-2 text-sm leading-7 text-[var(--color-text-soft)]">
                Reference model currently used to read live VaR/ES and capital posture from the MT5 bridge cache.
              </div>
            </div>
            <StatusBadge
              label={transport === "stream" ? "stream" : transport === "polling" ? "polling" : "connecting"}
              tone={transport === "stream" ? "success" : transport === "polling" ? "warning" : "neutral"}
            />
          </div>

          <div className="mt-6 grid gap-4">
            <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
              <div className="text-sm text-[var(--color-text-soft)]">Manual MT5 events</div>
              <div className="mt-2 text-2xl font-semibold text-white">
                {String(reconciliation?.manual_event_count ?? 0)}
              </div>
            </div>
            <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
              <div className="text-sm text-[var(--color-text-soft)]">Unmatched execution attempts</div>
              <div className="mt-2 text-2xl font-semibold text-white">
                {String(reconciliation?.unmatched_execution_count ?? 0)}
              </div>
            </div>
            <div className="text-sm leading-7 text-[var(--color-text-soft)]">
              {liveState?.generated_at
                ? `Bridge refreshed ${formatTimestamp(liveState.generated_at)}.`
                : "Bridge refresh pending."}{" "}
              {capital
                ? `Capital status ${capital.status} with ${formatCurrency(capital.total_capital_consumed_eur)} consumed.`
                : "No live capital posture available yet."}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

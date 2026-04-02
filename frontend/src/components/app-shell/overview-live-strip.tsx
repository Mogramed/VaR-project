"use client";

import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import type {
  CapitalUsageSnapshotResponse,
  MT5LiveStateResponse,
} from "@/lib/api/types";
import { preferredHeadlineRisk } from "@/lib/risk-surface";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

function firstNumericValue(values: Record<string, number> | undefined) {
  if (!values) return null;
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
  const selectedModel = riskSummary?.reference_model ?? fallbackSelectedModel;
  const varValue = riskSummary?.var?.[selectedModel] ?? firstNumericValue(riskSummary?.var) ?? fallbackVarValue;
  const esValue = riskSummary?.es?.[selectedModel] ?? firstNumericValue(riskSummary?.es) ?? fallbackEsValue;
  const headline = riskSummary?.headline_risk ?? [];
  const live95 = preferredHeadlineRisk(headline, ["live_1d_95"]);
  const live99 = preferredHeadlineRisk(headline, ["live_1d_99"]);
  const stressed = preferredHeadlineRisk(headline, [
    "stressed_10d_975",
    "stressed_10d_99",
    "governance_10d_975",
    "governance_10d_99",
  ]);
  const dataQuality = riskSummary?.data_quality ?? null;
  const microstructure = liveState?.microstructure ?? riskSummary?.microstructure ?? null;
  const tickQuality = liveState?.tick_quality ?? riskSummary?.tick_quality ?? null;
  const nowcast = liveState?.risk_nowcast ?? riskSummary?.risk_nowcast ?? null;
  const nowcast99 = (nowcast?.live_1d_99 ?? null) as { nowcast_var?: number; nowcast_es?: number } | null;
  const nowcastRegime = typeof nowcast?.regime === "string" ? nowcast.regime : undefined;
  const marketRegime = typeof microstructure?.regime === "string" ? microstructure.regime : undefined;
  const qualityStatus = typeof dataQuality?.status === "string" ? dataQuality.status : undefined;

  return (
    <div className="space-y-4">
      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <MetricBlock
          label={
            live95
              ? `VaR/ES ${Math.round(live95.alpha * 100)}% / ${live95.horizon_days}d`
              : `VaR / ${selectedModel.toUpperCase()}`
          }
          value={formatCurrency(live95?.var ?? varValue)}
          hint={
            live95
              ? `ES ${formatCurrency(live95.es)}`
              : riskSummary?.latest_observation
                ? formatTimestamp(riskSummary.latest_observation)
                : undefined
          }
          tone="accent"
        />
        <MetricBlock
          label={
            live99
              ? `VaR/ES ${Math.round(live99.alpha * 100)}% / ${live99.horizon_days}d`
              : `ES / ${selectedModel.toUpperCase()}`
          }
          value={formatCurrency(live99?.var ?? esValue)}
          hint={live99 ? `ES ${formatCurrency(live99.es)}` : riskSummary?.sample_size != null ? `${riskSummary.sample_size} obs` : undefined}
          tone="warning"
        />
        <MetricBlock
          label="Nowcast 1D 99%"
          value={formatCurrency(nowcast99?.nowcast_var)}
          hint={nowcast99?.nowcast_es != null ? `ES ${formatCurrency(nowcast99.nowcast_es)}` : nowcastRegime}
          tone="warning"
        />
        <MetricBlock
          label={stressed ? `Stress ${Math.round(stressed.alpha * 100)}% / ${stressed.horizon_days}d` : "Stress"}
          value={formatCurrency(stressed?.es ?? esValue)}
          hint={stressed?.scenario_name ?? marketRegime ?? qualityStatus}
          tone={stressed?.is_stressed ? "warning" : "neutral"}
        />
        <MetricBlock
          label="Capital headroom"
          value={capital ? formatPercent(capital.headroom_ratio ?? 0, 0) : "n/a"}
          hint={capital ? `${formatCurrency(capital.total_capital_remaining_eur)} remaining` : undefined}
          tone="success"
        />
        <MetricBlock
          label="MT5 Bridge"
          value={(liveState?.status ?? "pending").toUpperCase()}
          hint={liveState?.generated_at ? `${transport} / ${formatTimestamp(liveState.generated_at)}` : "Connecting"}
          tone={liveState?.status === "ok" ? "success" : liveState?.degraded ? "warning" : "neutral"}
        />
      </section>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(260px,0.8fr)]">
        <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} title="Watchlist" />

        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Live posture
            </span>
            <StatusBadge
              label={transport}
              tone={transport === "stream" ? "success" : transport === "polling" ? "warning" : "neutral"}
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Reference model</span>
              <span className="mono font-semibold text-[var(--color-text)]">{selectedModel.toUpperCase()}</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Risk quality</span>
              <StatusBadge
                label={String(dataQuality?.status ?? "unknown")}
                tone={dataQuality?.status === "healthy" ? "success" : dataQuality?.status === "thin_history" ? "warning" : "neutral"}
              />
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Tick quality</span>
              <StatusBadge
                label={String(tickQuality?.status ?? "unknown")}
                tone={tickQuality?.status === "healthy" ? "success" : tickQuality?.status === "stale" ? "warning" : "neutral"}
              />
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Market regime</span>
              <StatusBadge
                label={String(microstructure?.regime ?? "unknown")}
                tone={microstructure?.regime === "stressed" ? "warning" : microstructure?.regime === "volatile" ? "accent" : "success"}
              />
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Avg spread</span>
              <span className="mono text-[var(--color-text)]">
                {microstructure?.avg_spread_bps != null ? `${Number(microstructure.avg_spread_bps).toFixed(1)} bps` : "n/a"}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Manual MT5 events</span>
              <span className="mono text-[var(--color-text)]">{reconciliation?.manual_event_count ?? 0}</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Unmatched executions</span>
              <span className="mono text-[var(--color-text)]">{reconciliation?.unmatched_execution_count ?? 0}</span>
            </div>
            {capital ? (
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Capital status</span>
                <StatusBadge label={capital.status} tone="accent" />
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

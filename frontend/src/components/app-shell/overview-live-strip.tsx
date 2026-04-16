"use client";

import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { LivePostureBanner } from "@/components/app-shell/live-posture-banner";
import { LiveRuntimeBadgeGroup } from "@/components/app-shell/live-runtime-badge-group";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import type { CapitalUsageSnapshotResponse, MT5LiveStateResponse } from "@/lib/api/types";
import { preferredHeadlineRisk } from "@/lib/risk-surface";
import {
  formatCurrency,
  formatOperationalTruth,
  formatPercent,
  formatSourceLabel,
  formatTimestamp,
  formatTimestampWithSource,
  joinLabelParts,
} from "@/lib/utils";
import { useRelativeTime } from "@/lib/use-relative-time";

function firstNumericValue(values: Record<string, number> | undefined) {
  if (!values) return null;
  const first = Object.values(values)[0];
  return typeof first === "number" ? first : null;
}

export function OverviewLiveStrip({
  initialCapital,
  fallbackSelectedModel,
  fallbackVarValue,
  fallbackEsValue,
  fallbackSnapshotCreatedAt,
  fallbackSnapshotSource,
}: {
  initialCapital: CapitalUsageSnapshotResponse | null;
  fallbackSelectedModel: string;
  fallbackVarValue: number;
  fallbackEsValue: number;
  fallbackSnapshotCreatedAt: string | null;
  fallbackSnapshotSource: string | null;
}) {
  const { liveState, heartbeatAt, transport } = useDeskLive();
  return (
    <OverviewLiveStripPanel
      liveState={liveState}
      heartbeatAt={heartbeatAt}
      transport={transport}
      initialCapital={initialCapital}
      fallbackSelectedModel={fallbackSelectedModel}
      fallbackVarValue={fallbackVarValue}
      fallbackEsValue={fallbackEsValue}
      fallbackSnapshotCreatedAt={fallbackSnapshotCreatedAt}
      fallbackSnapshotSource={fallbackSnapshotSource}
    />
  );
}

export function OverviewLiveStripPanel({
  liveState,
  heartbeatAt,
  transport,
  initialCapital,
  fallbackSelectedModel,
  fallbackVarValue,
  fallbackEsValue,
  fallbackSnapshotCreatedAt,
  fallbackSnapshotSource,
}: {
  liveState: MT5LiveStateResponse | null;
  heartbeatAt: string | null;
  transport: "stream" | "polling" | "connecting";
  initialCapital: CapitalUsageSnapshotResponse | null;
  fallbackSelectedModel: string;
  fallbackVarValue: number;
  fallbackEsValue: number;
  fallbackSnapshotCreatedAt: string | null;
  fallbackSnapshotSource: string | null;
}) {
  const riskSummary = liveState?.risk_summary ?? null;
  const liveBridgeConnected = Boolean(liveState?.connected);
  const capital = liveBridgeConnected
    ? (liveState?.capital_usage ?? null)
    : (liveState?.capital_usage ?? initialCapital);
  const reconciliation = liveState?.reconciliation ?? null;
  const selectedModel = riskSummary?.reference_model ?? fallbackSelectedModel;
  const varValue = riskSummary
    ? (riskSummary.var?.[selectedModel] ?? firstNumericValue(riskSummary.var))
    : (liveBridgeConnected ? null : fallbackVarValue);
  const esValue = riskSummary
    ? (riskSummary.es?.[selectedModel] ?? firstNumericValue(riskSummary.es))
    : (liveBridgeConnected ? null : fallbackEsValue);
  const headline = riskSummary?.headline_risk ?? [];
  const live95 = preferredHeadlineRisk(headline, ["live_1d_95"]);
  const live99 = preferredHeadlineRisk(headline, ["live_1d_99"]);
  const stressed = preferredHeadlineRisk(headline, [
    "stressed_10d_975",
    "stressed_10d_99",
    "governance_10d_975",
    "governance_10d_99",
  ]);
  const stressUpliftVsLive99 = (
    stressed != null && live99 != null && typeof live99.var === "number" && live99.var > 0
  )
    ? Number(stressed.var) / Number(live99.var)
    : null;
  const dataQuality = riskSummary?.data_quality ?? null;
  const microstructure = liveState?.microstructure ?? riskSummary?.microstructure ?? null;
  const tickQuality = liveState?.tick_quality ?? riskSummary?.tick_quality ?? null;
  const nowcast = liveState?.risk_nowcast ?? riskSummary?.risk_nowcast ?? null;
  const nowcast99 = (nowcast?.live_1d_99 ?? null) as { nowcast_var?: number; nowcast_es?: number } | null;
  const nowcastRegime = typeof nowcast?.regime === "string" ? nowcast.regime : undefined;
  const marketRegime = typeof microstructure?.regime === "string" ? microstructure.regime : undefined;
  const qualityStatus = typeof dataQuality?.status === "string" ? dataQuality.status : undefined;
  const riskTimestamp =
    riskSummary?.latest_observation ??
    riskSummary?.generated_at ??
    initialCapital?.snapshot_timestamp ??
    initialCapital?.created_at ??
    fallbackSnapshotCreatedAt ??
    null;
  const riskSource =
    riskSummary?.source ??
    initialCapital?.snapshot_source ??
    initialCapital?.source ??
    fallbackSnapshotSource ??
    null;
  const capitalTimestamp = capital?.snapshot_timestamp ?? capital?.created_at ?? null;
  const capitalSource = capital?.snapshot_source ?? capital?.source ?? null;
  const riskFreshness = formatTimestampWithSource({ source: riskSource, timestamp: riskTimestamp });
  const capitalFreshness = formatTimestampWithSource({ source: capitalSource, timestamp: capitalTimestamp });
  const operationalTruth = reconciliation?.operational_truth
    ? formatOperationalTruth(reconciliation.operational_truth)
    : (initialCapital ?? fallbackSnapshotSource)
      ? "Persisted snapshot"
      : "n/a";
  const marketReference = formatTimestampWithSource({
    source: reconciliation?.market_reference_source,
    timestamp: reconciliation?.market_reference_timestamp,
  });
  const account = liveState?.account ?? null;
  const liveProfit = account?.profit ?? null;
  const liveEquity = account?.equity ?? null;
  const liveBalance = account?.balance ?? null;
  const liveMarginLevel = account?.margin_level ?? null;
  const liveFreshness = useRelativeTime(heartbeatAt ?? liveState?.generated_at);
  const concentration = computeBudgetConcentration(
    liveState?.risk_budget?.models ?? null,
    selectedModel,
  );

  return (
    <div className="space-y-4">
      <LivePostureBanner liveState={liveState} transport={transport} />
      {/* Live Account Ticker */}
      {account ? (
        <section className="grid gap-2 sm:grid-cols-5">
          <LiveTickerCell
            label="Balance"
            value={formatCurrency(liveBalance, 2)}
            tone="neutral"
          />
          <LiveTickerCell
            label="Equity"
            value={formatCurrency(liveEquity, 2)}
            tone={liveEquity != null && liveBalance != null && liveEquity >= liveBalance ? "success" : "warning"}
          />
          <LiveTickerCell
            label="Profit"
            value={liveProfit != null ? `${liveProfit >= 0 ? "+" : ""}${liveProfit.toFixed(2)}` : "n/a"}
            tone={liveProfit != null ? (liveProfit >= 0 ? "success" : "danger") : "neutral"}
            highlight
          />
          <LiveTickerCell
            label="Margin Level"
            value={liveMarginLevel != null ? `${liveMarginLevel.toFixed(0)}%` : "n/a"}
            tone={liveMarginLevel != null ? (liveMarginLevel > 500 ? "success" : liveMarginLevel > 150 ? "warning" : "danger") : "neutral"}
          />
          <LiveTickerCell
            label="Updated"
            value={liveFreshness}
            tone="neutral"
            mono
          />
        </section>
      ) : null}

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <MetricBlock
          label={
            live95
              ? `VaR/ES ${Math.round(live95.alpha * 100)}% / ${live95.horizon_days}d`
              : `VaR / ${selectedModel.toUpperCase()}`
          }
          value={formatCurrency(live95?.var ?? varValue)}
          hint={
            joinLabelParts(
              live95 ? `ES ${formatCurrency(live95.es)}` : null,
              riskFreshness,
            ) || undefined
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
          hint={
            joinLabelParts(
              live99 ? `ES ${formatCurrency(live99.es)}` : riskSummary?.sample_size != null ? `${riskSummary.sample_size} obs` : null,
              riskFreshness,
            ) || undefined
          }
          tone="warning"
        />
        <MetricBlock
          label="Nowcast 1D 99%"
          value={formatCurrency(nowcast99?.nowcast_var)}
          hint={
            joinLabelParts(
              nowcast99?.nowcast_es != null ? `ES ${formatCurrency(nowcast99.nowcast_es)}` : nowcastRegime,
              riskFreshness,
            ) || undefined
          }
          tone="warning"
        />
        <MetricBlock
          label={stressed ? `Stress ${Math.round(stressed.alpha * 100)}% / ${stressed.horizon_days}d` : "Stress"}
          value={formatCurrency(stressed?.es ?? esValue)}
          hint={
            joinLabelParts(
              stressUpliftVsLive99 == null ? null : `x${stressUpliftVsLive99.toFixed(2)} vs live 99%`,
              stressed?.scenario_name ?? marketRegime ?? qualityStatus,
              reconciliation?.market_closed ? marketReference : riskFreshness,
            ) || undefined
          }
          tone={stressed?.is_stressed ? "warning" : "neutral"}
        />
        <MetricBlock
          label="Capital headroom"
          value={capital ? formatPercent(capital.headroom_ratio ?? 0, 0) : "n/a"}
          hint={
            capital
              ? joinLabelParts(
                  `${formatCurrency(capital.total_capital_remaining_eur)} remaining`,
                  capitalFreshness,
                )
              : undefined
          }
          tone="success"
        />
        <MetricBlock
          label="MT5 Bridge"
          value={(liveState?.status ?? "pending").toUpperCase()}
          hint={
            reconciliation?.market_closed && reconciliation.market_reference_timestamp
              ? joinLabelParts("market closed", marketReference)
              : liveState?.generated_at
                ? joinLabelParts(transport, formatTimestampWithSource({
                    source: liveState?.source,
                    timestamp: liveState.generated_at,
                  }))
                : "Connecting"
          }
          tone={liveState?.status === "ok" ? "success" : liveState?.degraded ? "warning" : "neutral"}
        />
      </section>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(260px,0.8fr)]">
        <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} title="Watchlist" />

        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
          <div className="mb-3 flex items-center justify-between gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Live posture
            </span>
            <span className="flex items-center gap-1.5">
              <LiveRuntimeBadgeGroup
                liveState={liveState}
                heartbeatAt={heartbeatAt}
                transport={transport}
                showFreshness
              />
            </span>
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Reference model</span>
              <span className="mono font-semibold text-[var(--color-text)]">{selectedModel.toUpperCase()}</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Top-3 concentration</span>
              <span className="mono text-[var(--color-text)]">
                {concentration?.top3Share == null ? "n/a" : formatPercent(concentration.top3Share, 0)}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Effective positions</span>
              <span className="mono text-[var(--color-text)]">
                {concentration?.effectiveCount == null ? "n/a" : concentration.effectiveCount.toFixed(1)}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="text-[var(--color-text-muted)]">Dominant risk</span>
              <span className="mono text-right text-[var(--color-text)]">
                {concentration?.dominantLabel
                  ? `${concentration.dominantLabel} (${formatPercent(concentration.dominantShare ?? 0, 0)})`
                  : "n/a"}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="text-[var(--color-text-muted)]">Truth source</span>
              <span className="text-right text-[var(--color-text)]">{operationalTruth}</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Risk quality</span>
              <StatusBadge
                label={String(dataQuality?.status ?? "unknown")}
                tone={dataQuality?.status === "healthy" ? "success" : dataQuality?.status === "thin_history" ? "warning" : "neutral"}
              />
            </div>
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="text-[var(--color-text-muted)]">Risk basis</span>
              <span className="text-right text-[var(--color-text)]">{formatSourceLabel(riskSource)}</span>
            </div>
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="text-[var(--color-text-muted)]">Risk as of</span>
              <span className="mono text-right text-[var(--color-text)]">
                {riskTimestamp ? formatTimestamp(riskTimestamp) : "n/a"}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Tick quality</span>
              <StatusBadge
                label={String(tickQuality?.status ?? "unknown").replaceAll("_", " ")}
                tone={
                  tickQuality?.status === "healthy"
                    ? "success"
                    : tickQuality?.status === "stale"
                      ? "warning"
                      : tickQuality?.status === "market_closed"
                        ? "accent"
                        : "neutral"
                }
              />
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Market regime</span>
              <StatusBadge
                label={String(microstructure?.regime ?? "unknown").replaceAll("_", " ")}
                tone={
                  microstructure?.regime === "stressed"
                    ? "warning"
                    : microstructure?.regime === "volatile"
                      ? "accent"
                      : microstructure?.regime === "closed"
                        ? "neutral"
                        : "success"
                }
              />
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-[var(--color-text-muted)]">Avg spread</span>
              <span className="mono text-[var(--color-text)]">
                {microstructure?.avg_spread_bps != null ? `${Number(microstructure.avg_spread_bps).toFixed(1)} bps` : "n/a"}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="text-[var(--color-text-muted)]">Capital basis</span>
              <span className="text-right text-[var(--color-text)]">{formatSourceLabel(capitalSource)}</span>
            </div>
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="text-[var(--color-text-muted)]">Capital as of</span>
              <span className="mono text-right text-[var(--color-text)]">
                {capitalTimestamp ? formatTimestamp(capitalTimestamp) : "n/a"}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="text-[var(--color-text-muted)]">Market ref</span>
              <span className="mono text-right text-[var(--color-text)]">{marketReference}</span>
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

/* ─── Live Ticker Cell ─── */

const tickerToneColors: Record<string, string> = {
  success: "text-[var(--color-green)]",
  warning: "text-[var(--color-amber)]",
  danger: "text-[var(--color-red)]",
  accent: "text-[var(--color-accent)]",
  neutral: "text-[var(--color-text)]",
};

const tickerToneBg: Record<string, string> = {
  success: "bg-[var(--color-green-soft)]",
  danger: "bg-[var(--color-red-soft)]",
};

function LiveTickerCell({
  label,
  value,
  tone = "neutral",
  highlight,
  mono: useMono,
}: {
  label: string;
  value: string;
  tone?: "neutral" | "success" | "warning" | "danger" | "accent";
  highlight?: boolean;
  mono?: boolean;
}) {
  const valueColor = tickerToneColors[tone] ?? tickerToneColors.neutral;
  const bg = highlight && (tone === "success" || tone === "danger")
    ? tickerToneBg[tone]
    : "bg-[var(--color-surface)]";

  return (
    <div className={`rounded-[var(--radius-md)] border border-[var(--color-border)] px-3 py-2 ${bg} transition-colors duration-300`}>
      <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
        {label}
      </div>
      <div className={`mt-0.5 text-lg font-semibold tabular-nums tracking-tight ${valueColor} ${useMono ? "mono" : ""} ${highlight ? "animate-[tick-fade_300ms_ease]" : ""}`}>
        {value}
      </div>
    </div>
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return null;
}

function computeBudgetConcentration(
  models: unknown,
  selectedModel: string,
): {
  top3Share: number | null;
  effectiveCount: number | null;
  dominantLabel: string | null;
  dominantShare: number | null;
} | null {
  const modelMap = asRecord(models);
  if (modelMap == null) {
    return null;
  }
  const selectedPayload = asRecord(modelMap[selectedModel]);
  const fallbackKey = Object.keys(modelMap)[0];
  const modelPayload = selectedPayload ?? (fallbackKey ? asRecord(modelMap[fallbackKey]) : null);
  if (modelPayload == null) {
    return null;
  }
  const positions = asRecord(modelPayload.positions);
  if (positions == null) {
    return null;
  }
  const contributors: Array<{ label: string; absComponent: number }> = [];
  for (const [symbol, rawPosition] of Object.entries(positions)) {
    const position = asRecord(rawPosition);
    if (position == null) {
      continue;
    }
    const label = typeof position.symbol === "string" && position.symbol ? position.symbol : symbol;
    const componentVar = asNumber(position.component_var);
    if (componentVar == null) {
      continue;
    }
    contributors.push({ label, absComponent: Math.abs(componentVar) });
  }
  if (contributors.length === 0) {
    return null;
  }
  contributors.sort((left, right) => right.absComponent - left.absComponent);
  const total = contributors.reduce((acc, item) => acc + item.absComponent, 0);
  if (total <= 1e-12) {
    return {
      top3Share: null,
      effectiveCount: null,
      dominantLabel: null,
      dominantShare: null,
    };
  }
  const shares = contributors.map((item) => item.absComponent / total);
  const hhi = shares.reduce((acc, item) => acc + item * item, 0);
  return {
    top3Share: shares.slice(0, 3).reduce((acc, item) => acc + item, 0),
    effectiveCount: hhi > 1e-12 ? 1 / hhi : null,
    dominantLabel: contributors[0]?.label ?? null,
    dominantShare: shares[0] ?? null,
  };
}

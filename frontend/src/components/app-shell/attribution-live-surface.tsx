"use client";

import { useQuery } from "@tanstack/react-query";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { LivePostureBanner } from "@/components/app-shell/live-posture-banner";
import { LiveRuntimeBadgeGroup } from "@/components/app-shell/live-runtime-badge-group";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { AttributionTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { ModelComparisonResponse, RiskAttributionResponse } from "@/lib/api/types";
import { CHART_PALETTE, makeBarOption } from "@/lib/chart-options";
import {
  deskArtifactQueryKey,
  deskArtifactQueryOptions,
} from "@/components/app-shell/desk-artifact-query";
import { formatCurrency, formatTimestamp } from "@/lib/utils";
import { flattenAttribution } from "@/lib/view-models";

function preferredSource(liveState: ReturnType<typeof useDeskLive>["liveState"]) {
  return liveState?.risk_budget ? "mt5_live_bridge" : "auto";
}

export function AttributionLiveSurface({
  portfolioSlug, preferredModel, initialAttribution, initialComparison,
}: {
  portfolioSlug: string; preferredModel?: string;
  initialAttribution: RiskAttributionResponse | null;
  initialComparison: ModelComparisonResponse | null;
}) {
  const { liveState, transport } = useDeskLive();
  const attributionSource = preferredSource(liveState);
  const attributionQuery = useQuery({
    queryKey: deskArtifactQueryKey("attribution", portfolioSlug, attributionSource),
    queryFn: () => api.latestAttribution(portfolioSlug, attributionSource),
    initialData: initialAttribution,
    ...deskArtifactQueryOptions,
  });
  const comparisonQuery = useQuery({
    queryKey: deskArtifactQueryKey("models", "comparison", portfolioSlug),
    queryFn: () => api.latestModelComparison(portfolioSlug),
    initialData: initialComparison,
    ...deskArtifactQueryOptions,
  });
  const attribution = attributionQuery.data ?? initialAttribution;
  const comparison = comparisonQuery.data ?? initialComparison;

  const selectedModel = preferredModel ?? liveState?.risk_budget?.preferred_model ?? liveState?.risk_summary?.reference_model ?? comparison?.champion_model ?? (attribution ? Object.keys(attribution.models)[0] : undefined) ?? "hist";
  const rows = attribution ? flattenAttribution(attribution, selectedModel) : [];
  const primaryContributor = rows[0];
  const diversifier = rows.slice().sort((a, b) => a.componentVar - b.componentVar).find((r) => r.componentVar < 0) ?? null;

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Attribution" title="Per-symbol contribution and portfolio pressure"
        aside={<>
          <StatusBadge label={selectedModel.toUpperCase()} tone="accent" />
          <StatusBadge label={(attribution?.snapshot_source ?? "auto").replaceAll("_", " ")} tone={attribution?.snapshot_source === "mt5_live_bridge" ? "success" : "neutral"} />
          <LiveRuntimeBadgeGroup liveState={liveState} transport={transport} showBridge={false} />
        </>}
      />
      <LivePostureBanner liveState={liveState} transport={transport} />

      {/* Model selector */}
      {attribution ? (
        <div className="flex flex-wrap gap-1.5">
          {Object.keys(attribution.models).map((model) => (
            <a key={model} href={`/desk/attribution?portfolio=${encodeURIComponent(portfolioSlug)}&model=${model}`}
              className={`rounded-[var(--radius-sm)] border px-2.5 py-1 text-[11px] uppercase tracking-wider transition-colors ${
                model === selectedModel
                  ? "border-[var(--color-accent)]/30 bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                  : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text-soft)]"
              }`}>
              {model}
            </a>
          ))}
        </div>
      ) : null}

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />

      <ChartSurface
        option={makeBarOption(rows.map((r) => ({ label: r.symbol, value: r.componentVar })), { color: CHART_PALETTE.gold, negativeColor: CHART_PALETTE.green, mode: rows.length <= 5 ? "sparse" : "standard" })}
        mode={rows.length <= 5 ? "sparse" : "standard"}
        dataCount={rows.length}
        title="Component VaR by symbol"
        meta={attribution?.snapshot_timestamp ? formatTimestamp(attribution.snapshot_timestamp) : undefined}
        insight={rows.length ? (
          <div className="grid gap-3">
            <MetricBlock label="Primary contributor" value={primaryContributor?.symbol ?? "n/a"} hint={primaryContributor ? `${formatCurrency(primaryContributor.componentVar)} cVaR` : undefined} tone="warning" className="bg-transparent" />
            <MetricBlock label="Diversifier" value={diversifier?.symbol ?? "none"} hint={diversifier ? `${formatCurrency(diversifier.componentVar)} cVaR` : undefined} tone="success" className="bg-transparent" />
          </div>
        ) : undefined}
        emptyState={<p className="text-xs text-[var(--color-text-muted)]">No attribution data available.</p>}
      />

      <div className="space-y-2">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Attribution table</h4>
        <AttributionTable rows={rows} />
      </div>
    </div>
  );
}

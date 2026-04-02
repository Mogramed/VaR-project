"use client";

import { startTransition, useEffect, useState } from "react";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { AttributionTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { ModelComparisonResponse, MT5LiveStateResponse, RiskAttributionResponse } from "@/lib/api/types";
import { CHART_PALETTE, makeBarOption } from "@/lib/chart-options";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatTimestamp } from "@/lib/utils";
import { flattenAttribution } from "@/lib/view-models";

function preferredSource(liveState: MT5LiveStateResponse | null) {
  return liveState?.risk_budget ? "mt5_live_bridge" : "historical";
}

export function AttributionLiveSurface({
  portfolioSlug, preferredModel, initialLiveState, initialAttribution, initialComparison,
}: {
  portfolioSlug: string; preferredModel?: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialAttribution: RiskAttributionResponse | null;
  initialComparison: ModelComparisonResponse | null;
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [attribution, setAttribution] = useState(initialAttribution);
  const [comparison, setComparison] = useState(initialComparison);
  const attributionSource = preferredSource(liveState);

  useEffect(() => {
    let c = false;
    (async () => {
      try {
        const next = await api.latestAttribution(portfolioSlug, attributionSource);
        if (!c) startTransition(() => setAttribution(next));
      } catch {
        if (attributionSource !== "historical") {
          try {
            const fb = await api.latestAttribution(portfolioSlug, "historical");
            if (!c) startTransition(() => setAttribution(fb));
          } catch { /* keep current */ }
        }
      }
    })();
    return () => { c = true; };
  }, [attributionSource, portfolioSlug, liveState?.sequence]);

  useEffect(() => {
    let c = false;
    const t = window.setInterval(() => {
      api.latestModelComparison(portfolioSlug).then((n) => { if (!c) startTransition(() => setComparison(n)); }).catch(() => {});
    }, 30000);
    return () => { c = true; window.clearInterval(t); };
  }, [portfolioSlug]);

  const selectedModel = preferredModel ?? liveState?.risk_budget?.preferred_model ?? liveState?.risk_summary?.reference_model ?? comparison?.champion_model ?? (attribution ? Object.keys(attribution.models)[0] : undefined) ?? "hist";
  const rows = attribution ? flattenAttribution(attribution, selectedModel) : [];
  const primaryContributor = rows[0];
  const diversifier = rows.slice().sort((a, b) => a.componentVar - b.componentVar).find((r) => r.componentVar < 0) ?? null;

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Attribution" title="Per-symbol contribution and portfolio pressure"
        aside={<>
          <StatusBadge label={selectedModel.toUpperCase()} tone="accent" />
          <StatusBadge label={(attribution?.snapshot_source ?? "historical").replaceAll("_", " ")} tone={attribution?.snapshot_source === "mt5_live_bridge" ? "success" : "neutral"} />
          <StatusBadge label={transport} tone={transport === "stream" ? "success" : "warning"} />
        </>}
      />

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

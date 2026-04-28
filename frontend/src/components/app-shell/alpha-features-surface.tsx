"use client";

import { useQuery } from "@tanstack/react-query";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { DashboardActiveFilters } from "@/components/app-shell/dashboard-active-filters";
import { DecisionAlphaIntelligencePanel } from "@/components/app-shell/decision-alpha-panels";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { RiskDecisionResponse } from "@/lib/api/types";
import { CHART_PALETTE, makeBarOption } from "@/lib/chart-options";
import {
  deskArtifactQueryKey,
  deskArtifactQueryOptions,
} from "@/components/app-shell/desk-artifact-query";
import { useDashboardPrefs } from "@/lib/dashboard-preferences-context";
import { formatPercent, formatTimestamp } from "@/lib/utils";

function flagFromUnknown(value: unknown): boolean | null {
  if (typeof value === "boolean") {
    return value;
  }
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    return numeric >= 0.5;
  }
  return null;
}

export function AlphaFeaturesSurface({
  portfolioSlug,
  initialDecisions,
}: {
  portfolioSlug: string;
  initialDecisions: RiskDecisionResponse[];
}) {
  const { accountId } = useDeskLive();
  const { matchesSymbol } = useDashboardPrefs();
  const decisionsQuery = useQuery({
    queryKey: deskArtifactQueryKey("alpha-features-decisions", portfolioSlug, accountId ?? "default", 40),
    queryFn: () => api.recentDecisions(portfolioSlug, 40, accountId),
    initialData: initialDecisions,
    ...deskArtifactQueryOptions,
  });
  const decisions = (decisionsQuery.data ?? initialDecisions).filter((decision) => matchesSymbol(decision.symbol));
  const latestWithIntelligence = decisions.find((decision) => decision.decision_intelligence != null) ?? decisions[0] ?? null;
  const intelligence = latestWithIntelligence?.decision_intelligence ?? null;
  const modelRuntime = intelligence?.model_runtime ?? null;
  const runtimeRows = (modelRuntime?.source_rows ?? {}) as Record<string, number | string | undefined>;
  const calculationAvailability = (
    intelligence as unknown as { calculation_availability?: Record<string, unknown> } | null
  )?.calculation_availability;
  const calculations = Object.entries(intelligence?.calculations ?? {}).filter(
    ([key]) =>
      !key.startsWith("feature_available_") &&
      !key.startsWith("calc_available_") &&
      !key.startsWith("calc_derived_"),
  );

  const isCalculationAvailable = (key: string) => {
    const explicit = flagFromUnknown(calculationAvailability?.[key]);
    if (explicit != null) {
      return explicit;
    }
    const fallback = flagFromUnknown(intelligence?.calculations?.[`calc_available_${key}`]);
    if (fallback != null) {
      return fallback;
    }
    return true;
  };

  const formatCalcValue = (value: unknown) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(4) : "n/a";
  };
  const scoreSeries = decisions
    .filter((decision) => decision.decision_intelligence != null)
    .slice(0, 12)
    .reverse()
    .map((decision, index) => ({
      label: decision.symbol || String(index + 1),
      value: Number(decision.decision_intelligence?.score ?? 0),
    }));

  return (
    <div className="desk-page space-y-4">
      <PageHeader
        eyebrow="Alpha"
        title="Features & calculations"
        aside={<StatusBadge label={portfolioSlug} tone="accent" />}
      />
      <DashboardActiveFilters showHorizon={false} showModel={false} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-8">
        <MetricBlock label="Latest symbol" value={latestWithIntelligence?.symbol ?? "n/a"} />
        <MetricBlock label="Signal" value={intelligence?.signal ?? "n/a"} />
        <MetricBlock label="Confidence" value={formatPercent(intelligence?.confidence ?? null, 0)} />
        <MetricBlock
          label="Runtime"
          value={modelRuntime?.trained_model ? "Trained" : "Fallback"}
          tone={modelRuntime?.trained_model ? "success" : "warning"}
        />
        <MetricBlock label="Train samples" value={String(Number(modelRuntime?.sample_count ?? 0))} />
        <MetricBlock label="Backtest rows" value={String(Number(runtimeRows.backtest ?? 0))} />
        <MetricBlock label="Execution rows" value={String(Number(runtimeRows.execution ?? 0))} />
        <MetricBlock label="Updated" value={formatTimestamp(latestWithIntelligence?.created_at ?? latestWithIntelligence?.time_utc ?? null)} />
      </section>

      <DecisionAlphaIntelligencePanel intelligence={intelligence} title="Decision Alpha live explainability" />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(280px,0.95fr)]">
        <ChartSurface
          option={makeBarOption(
            scoreSeries,
            {
              color: CHART_PALETTE.gold,
              negativeColor: CHART_PALETTE.red,
              mode: scoreSeries.length <= 5 ? "sparse" : "standard",
            },
          )}
          mode={scoreSeries.length <= 5 ? "sparse" : "standard"}
          dataCount={scoreSeries.length}
          title="Recent Decision Alpha scores"
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No Decision Alpha score history yet.</p>}
        />
        <section className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Calculations snapshot
          </div>
          {calculations.length === 0 ? (
            <p className="mt-2 text-xs text-[var(--color-text-muted)]">No calculations available.</p>
          ) : (
            <div className="mt-2 space-y-1.5">
              {calculations.map(([key, value]) => (
                <div key={key} className="grid grid-cols-[minmax(0,1fr)_110px] gap-2 text-[11px]">
                  <span className="truncate text-[var(--color-text-soft)]">{key.replace(/[_-]+/g, " ")}</span>
                  <span className="mono text-right text-[var(--color-text)]">
                    {isCalculationAvailable(key) ? formatCalcValue(value) : "n/a"}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

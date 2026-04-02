"use client";

import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/app-shell/page-header";
import { FormMetaTile, FieldInput, FieldLabel } from "@/components/forms/shared";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type {
  RiskAttributionAssetClassResponse,
  RiskAttributionModelResponse,
  RiskAttributionPositionResponse,
  StressReportResponse,
  StressScenarioRequest,
} from "@/lib/api/types";
import { preferredHeadlineRisk } from "@/lib/risk-surface";
import { formatCurrency, formatPercent } from "@/lib/utils";

const defaults: StressScenarioRequest[] = [
  { name: "Volatility regime shift", vol_multiplier: 1.5, shock_pnl: 0.0 },
  { name: "FX directional down shock", vol_multiplier: 1.1, shock_pnl: -15000 },
  { name: "Correlated multi-asset drawdown", vol_multiplier: 2.0, shock_pnl: -25000 },
];

function asNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function selectAttributionModel(
  attribution: StressReportResponse["attribution"] | StressReportResponse["scenarios"][number]["attribution"] | null | undefined,
) {
  if (!attribution?.models) return null;
  return attribution.models.hist ?? Object.values(attribution.models)[0] ?? null;
}

function positionContributors(model: RiskAttributionModelResponse | null | undefined) {
  return Object.values(model?.positions ?? {}).sort(
    (left, right) => (right.component_es ?? right.component_var ?? 0) - (left.component_es ?? left.component_var ?? 0),
  ) as RiskAttributionPositionResponse[];
}

function assetClassContributors(model: RiskAttributionModelResponse | null | undefined) {
  return Object.values(model?.asset_classes ?? {}).sort(
    (left, right) => (right.component_es ?? right.component_var ?? 0) - (left.component_es ?? left.component_var ?? 0),
  ) as RiskAttributionAssetClassResponse[];
}

export function StressLiveSurface({ portfolioSlug }: { portfolioSlug: string }) {
  const [scenarios, setScenarios] = useState(defaults);
  const [name, setName] = useState("Custom");
  const [vol, setVol] = useState("2.0");
  const [shock, setShock] = useState("0");

  const mutation = useMutation({
    mutationFn: () =>
      api.runStressTest({
        portfolio_slug: portfolioSlug,
        scenarios: scenarios.length > 0 ? scenarios : undefined,
      }),
  });

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Stress" title="VaR, ES and scenario governance" />

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Scenario library</h3>
          <div className="mt-3 space-y-1.5">
            {scenarios.map((sc, i) => (
              <div
                key={`${sc.name}-${i}`}
                className="flex items-center justify-between rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-xs"
              >
                <span className="font-semibold text-[var(--color-text)]">{sc.name}</span>
                <div className="flex items-center gap-3">
                  <span className="text-[var(--color-text-muted)]">x{sc.vol_multiplier.toFixed(1)}</span>
                  {sc.shock_pnl !== 0 ? (
                    <span className="text-[var(--color-text-muted)]">{formatCurrency(sc.shock_pnl)}</span>
                  ) : null}
                  <button
                    type="button"
                    className="text-[var(--color-text-muted)] hover:text-[var(--color-red)]"
                    onClick={() => setScenarios((prev) => prev.filter((_, j) => j !== i))}
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 border-t border-[var(--color-border)] pt-4">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Add scenario</div>
            <div className="mt-2 grid gap-2 sm:grid-cols-3">
              <div>
                <FieldLabel htmlFor="stress-name">Name</FieldLabel>
                <FieldInput id="stress-name" value={name} onChange={(e) => setName(e.target.value)} />
              </div>
              <div>
                <FieldLabel htmlFor="stress-vol">Vol mult</FieldLabel>
                <FieldInput id="stress-vol" type="number" min="0.1" step="0.1" value={vol} onChange={(e) => setVol(e.target.value)} />
              </div>
              <div>
                <FieldLabel htmlFor="stress-shock">Shock EUR</FieldLabel>
                <FieldInput id="stress-shock" type="number" step="100" value={shock} onChange={(e) => setShock(e.target.value)} />
              </div>
            </div>
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                className="h-7 rounded-[var(--radius-sm)] border border-[var(--color-border)] px-3 text-[11px] font-medium text-[var(--color-text)] transition hover:bg-[var(--color-surface-hover)]"
                onClick={() => {
                  if (!name.trim()) return;
                  setScenarios((prev) => [
                    ...prev,
                    { name: name.trim(), vol_multiplier: Number(vol) || 1, shock_pnl: Number(shock) || 0 },
                  ]);
                  setName("Custom");
                  setVol("2.0");
                  setShock("0");
                }}
              >
                Add
              </button>
              <button
                type="button"
                className="h-7 rounded-[var(--radius-sm)] border border-[var(--color-border)] px-3 text-[11px] text-[var(--color-text-muted)] transition hover:bg-[var(--color-surface-hover)]"
                onClick={() => setScenarios(defaults)}
              >
                Reset
              </button>
            </div>
          </div>

          <div className="mt-4">
            <button
              type="button"
              disabled={mutation.isPending}
              onClick={() => mutation.mutate()}
              className="h-8 rounded-[var(--radius-md)] bg-[var(--color-accent)] px-4 text-[12px] font-semibold text-[#1a1206] transition hover:brightness-110 disabled:opacity-50"
            >
              {mutation.isPending ? "Running..." : "Run stress test"}
            </button>
            {mutation.error ? (
              <div className="mt-2 text-[11px] text-[var(--color-red)]">
                {mutation.error instanceof Error ? mutation.error.message : "Failed"}
              </div>
            ) : null}
          </div>
        </div>

        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] p-4">
          <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Stress report</h3>
          {!mutation.data ? (
            <p className="mt-3 text-xs text-[var(--color-text-muted)]">
              Run the stress suite to compare baseline risk, stressed ES and historical worst episodes.
            </p>
          ) : (
            <StressResults report={mutation.data} />
          )}
        </div>
      </div>
    </div>
  );
}

function StressResults({ report }: { report: StressReportResponse }) {
  const live99 = preferredHeadlineRisk(report.headline_risk, ["live_1d_99"]);
  const stressed99 = preferredHeadlineRisk(report.headline_risk, ["stressed_10d_975", "stressed_10d_99", "governance_10d_975", "governance_10d_99"]);
  const live99Es = asNumber(live99?.es);
  const stressed99Es = asNumber(stressed99?.es);
  const historicalExtremes = report.historical_extremes ?? [];
  const dataQuality = (report.risk_surface?.data_quality ?? null) as { status?: string; observation_count?: number } | null;
  const worstScenario = useMemo(() => {
    if (report.scenarios.length === 0) return null;
    return report.scenarios.reduce((left, right) => {
      const leftPrimary = (left.primary_metric ?? {}) as { es?: number };
      const rightPrimary = (right.primary_metric ?? {}) as { es?: number };
      const leftEs = leftPrimary.es ?? left.es ?? 0;
      const rightEs = rightPrimary.es ?? right.es ?? 0;
      return rightEs > leftEs ? right : left;
    });
  }, [report.scenarios]);
  const uplift =
    stressed99Es != null && live99Es != null && live99Es > 0 ? (stressed99Es - live99Es) / live99Es : null;
  const worstScenarioEs = asNumber(
    (worstScenario?.primary_metric as { es?: unknown } | null | undefined)?.es ?? worstScenario?.es,
  );
  const baselineAttributionModel = selectAttributionModel(report.attribution);
  const worstScenarioAttributionModel = selectAttributionModel(worstScenario?.attribution);
  const baselinePositions = positionContributors(baselineAttributionModel);
  const baselineAssetClasses = assetClassContributors(baselineAttributionModel);
  const stressedPositions = positionContributors(worstScenarioAttributionModel);
  const stressedAssetClasses = assetClassContributors(worstScenarioAttributionModel);
  const baselineLargestIncremental = baselinePositions
    .slice()
    .sort((left, right) => (right.incremental_es ?? 0) - (left.incremental_es ?? 0))[0];
  const stressedLargestIncremental = stressedPositions
    .slice()
    .sort((left, right) => (right.incremental_es ?? 0) - (left.incremental_es ?? 0))[0];

  return (
    <div className="space-y-4">
      <div className="grid gap-2 sm:grid-cols-4">
        <FormMetaTile
          label="Baseline VaR"
          value={formatCurrency(live99?.var ?? report.baseline_var)}
          hint={live99Es != null ? `ES ${formatCurrency(live99Es)}` : report.baseline_es != null ? `ES ${formatCurrency(report.baseline_es)}` : undefined}
          tone="accent"
        />
        <FormMetaTile
          label="Stressed ES 10D"
          value={formatCurrency(stressed99Es)}
          hint={stressed99?.scenario_name ?? stressed99?.label}
          tone="danger"
        />
        <FormMetaTile
          label="Stress uplift"
          value={uplift != null ? formatPercent(uplift, 0) : "n/a"}
          tone={uplift != null && uplift > 0.5 ? "danger" : "warning"}
        />
        <FormMetaTile
          label="Worst scenario"
          value={worstScenario?.name ?? "n/a"}
          hint={worstScenarioEs != null ? `ES ${formatCurrency(worstScenarioEs)}` : undefined}
          tone="warning"
        />
        <FormMetaTile
          label="Data quality"
          value={dataQuality?.status ?? "n/a"}
          hint={dataQuality?.observation_count != null ? `${dataQuality.observation_count} obs` : undefined}
          tone={dataQuality?.status === "healthy" ? "success" : dataQuality?.status === "thin_history" ? "warning" : "neutral"}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Baseline contributors
            </div>
            <StatusBadge label={baselineAttributionModel?.model?.toUpperCase() ?? "n/a"} tone="accent" />
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            <FormMetaTile
              label="Top symbol cES"
              value={baselinePositions[0]?.symbol ?? "n/a"}
              hint={baselinePositions[0] ? formatCurrency(baselinePositions[0].component_es) : undefined}
              tone="warning"
            />
            <FormMetaTile
              label="Top asset class cES"
              value={baselineAssetClasses[0]?.asset_class ?? "n/a"}
              hint={baselineAssetClasses[0] ? formatCurrency(baselineAssetClasses[0].component_es) : undefined}
              tone="accent"
            />
            <FormMetaTile
              label="Largest iES"
              value={baselineLargestIncremental?.symbol ?? "n/a"}
              hint={baselineLargestIncremental ? formatCurrency(baselineLargestIncremental.incremental_es) : undefined}
            />
          </div>
          {baselinePositions.length > 0 ? (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="border-b border-[var(--color-border)] text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                    <th className="pb-2 pr-3">Symbol</th>
                    <th className="pb-2 pr-3 text-right">cVaR</th>
                    <th className="pb-2 pr-3 text-right">cES</th>
                    <th className="pb-2 text-right">iES</th>
                  </tr>
                </thead>
                <tbody>
                  {baselinePositions.slice(0, 5).map((item) => (
                    <tr key={item.symbol} className="border-b border-[var(--color-border)]">
                      <td className="py-2 pr-3 font-semibold text-[var(--color-text)]">{item.symbol}</td>
                      <td className="py-2 pr-3 text-right">{formatCurrency(item.component_var)}</td>
                      <td className="py-2 pr-3 text-right">{formatCurrency(item.component_es)}</td>
                      <td className="py-2 text-right">{formatCurrency(item.incremental_es)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>

        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Worst-scenario contributors
            </div>
            <StatusBadge label={worstScenario?.name ?? "n/a"} tone="warning" />
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            <FormMetaTile
              label="Top symbol cES"
              value={stressedPositions[0]?.symbol ?? "n/a"}
              hint={stressedPositions[0] ? formatCurrency(stressedPositions[0].component_es) : undefined}
              tone="warning"
            />
            <FormMetaTile
              label="Top asset class cES"
              value={stressedAssetClasses[0]?.asset_class ?? "n/a"}
              hint={stressedAssetClasses[0] ? formatCurrency(stressedAssetClasses[0].component_es) : undefined}
              tone="danger"
            />
            <FormMetaTile
              label="Largest iES"
              value={stressedLargestIncremental?.symbol ?? "n/a"}
              hint={stressedLargestIncremental ? formatCurrency(stressedLargestIncremental.incremental_es) : undefined}
              tone="warning"
            />
          </div>
          {stressedAssetClasses.length > 0 ? (
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="border-b border-[var(--color-border)] text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                    <th className="pb-2 pr-3">Asset class</th>
                    <th className="pb-2 pr-3 text-right">cVaR</th>
                    <th className="pb-2 pr-3 text-right">cES</th>
                    <th className="pb-2 text-right">Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {stressedAssetClasses.slice(0, 5).map((item) => (
                    <tr key={item.asset_class} className="border-b border-[var(--color-border)]">
                      <td className="py-2 pr-3 font-semibold text-[var(--color-text)]">{item.asset_class}</td>
                      <td className="py-2 pr-3 text-right">{formatCurrency(item.component_var)}</td>
                      <td className="py-2 pr-3 text-right">{formatCurrency(item.component_es)}</td>
                      <td className="py-2 text-right">
                        {item.contribution_pct_es != null ? formatPercent(item.contribution_pct_es, 0) : "n/a"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-[var(--color-border)] text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              <th className="pb-2 pr-3">Scenario</th>
              <th className="pb-2 pr-3 text-right">Vol</th>
              <th className="pb-2 pr-3 text-right">Primary VaR</th>
              <th className="pb-2 pr-3 text-right">Primary ES</th>
              <th className="pb-2 text-right">Status</th>
            </tr>
          </thead>
          <tbody>
            {report.scenarios.map((scenario) => {
              const primary = (scenario.primary_metric ?? {}) as { var?: number; es?: number };
              const ratio =
                report.baseline_es && (primary.es ?? scenario.es) != null && report.baseline_es > 0
                  ? (Number(primary.es ?? scenario.es) - report.baseline_es) / report.baseline_es
                  : null;
              return (
                <tr key={scenario.name} className="border-b border-[var(--color-border)]">
                  <td className="py-2 pr-3 font-semibold text-[var(--color-text)]">{scenario.name}</td>
                  <td className="py-2 pr-3 text-right text-[var(--color-text-soft)]">x{scenario.vol_multiplier.toFixed(1)}</td>
                  <td className="py-2 pr-3 text-right font-semibold text-[var(--color-text)]">{formatCurrency(primary.var ?? scenario.var)}</td>
                  <td className="py-2 pr-3 text-right text-[var(--color-text-soft)]">{formatCurrency(primary.es ?? scenario.es)}</td>
                  <td className="py-2 text-right">
                    <StatusBadge
                      label={ratio == null ? "n/a" : formatPercent(ratio, 0)}
                      tone={ratio != null && ratio > 0.5 ? "warning" : "success"}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {historicalExtremes.length > 0 ? (
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Historical worst episodes
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            {historicalExtremes.map((item) => (
              <FormMetaTile
                key={item.horizon_days}
                label={`Worst ${item.horizon_days}d`}
                value={formatCurrency(item.worst_loss)}
                hint={item.tail_mean_loss != null ? `Tail mean ${formatCurrency(item.tail_mean_loss)}` : (item.worst_end_date ?? undefined)}
                tone="warning"
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

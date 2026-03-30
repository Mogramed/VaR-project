"use client";

import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { PageHeader } from "@/components/app-shell/page-header";
import { FormMetaTile, FieldInput, FieldLabel } from "@/components/forms/shared";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { StressReportResponse, StressScenarioRequest } from "@/lib/api/types";
import { formatCurrency, formatPercent } from "@/lib/utils";

const defaultScenarios: StressScenarioRequest[] = [
  { name: "2008 Crisis", vol_multiplier: 3.0, shock_pnl: 0.0 },
  { name: "COVID March 2020", vol_multiplier: 4.0, shock_pnl: 0.0 },
  { name: "ECB Rate Shock", vol_multiplier: 2.0, shock_pnl: 0.0 },
  { name: "Mild Stress", vol_multiplier: 1.5, shock_pnl: 0.0 },
];

export function StressLiveSurface({
  portfolioSlug,
}: {
  portfolioSlug: string;
}) {
  const [scenarios, setScenarios] = useState<StressScenarioRequest[]>(defaultScenarios);
  const [customName, setCustomName] = useState("Custom");
  const [customVol, setCustomVol] = useState("2.0");
  const [customShock, setCustomShock] = useState("0");

  const mutation = useMutation({
    mutationFn: async () =>
      api.runStressTest({
        portfolio_slug: portfolioSlug,
        scenarios: scenarios.length > 0 ? scenarios : undefined,
      }),
  });

  const report = mutation.data;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Stress Testing"
        title="Multi-scenario VaR stress analysis"
        description="Apply volatility multipliers and additive PnL shocks to the portfolio, then compare stressed VaR/ES against the baseline."
      />

      <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        {/* Scenario configuration */}
        <div className="surface rounded-[1.8rem] p-6">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Scenarios
          </div>

          <div className="mt-4 space-y-3">
            {scenarios.map((sc, i) => (
              <div
                key={sc.name}
                className="flex items-center gap-3 rounded-[1.2rem] border border-white/8 bg-white/[0.03] px-4 py-3"
              >
                <div className="flex-1 text-sm font-semibold text-white">{sc.name}</div>
                <div className="text-xs text-[var(--color-text-muted)]">
                  vol &times;{sc.vol_multiplier.toFixed(1)}
                </div>
                {sc.shock_pnl !== 0 && (
                  <div className="text-xs text-[var(--color-text-muted)]">
                    shock {sc.shock_pnl > 0 ? "+" : ""}{sc.shock_pnl.toFixed(2)}
                  </div>
                )}
                <button
                  type="button"
                  className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-red)] transition-colors"
                  onClick={() => setScenarios((prev) => prev.filter((_, j) => j !== i))}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>

          <div className="mt-5 border-t border-white/8 pt-5">
            <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
              Add custom scenario
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <div>
                <FieldLabel htmlFor="stress-custom-name">Name</FieldLabel>
                <FieldInput
                  id="stress-custom-name"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                />
              </div>
              <div>
                <FieldLabel htmlFor="stress-custom-vol">Vol multiplier</FieldLabel>
                <FieldInput
                  id="stress-custom-vol"
                  type="number"
                  min="0.1"
                  step="0.1"
                  value={customVol}
                  onChange={(e) => setCustomVol(e.target.value)}
                />
              </div>
              <div>
                <FieldLabel htmlFor="stress-custom-shock">Shock PnL (EUR)</FieldLabel>
                <FieldInput
                  id="stress-custom-shock"
                  type="number"
                  step="100"
                  value={customShock}
                  onChange={(e) => setCustomShock(e.target.value)}
                />
              </div>
            </div>
            <div className="mt-3 flex gap-3">
              <button
                type="button"
                className="inline-flex h-10 items-center justify-center rounded-full border border-white/12 bg-white/5 px-4 text-sm font-semibold text-[var(--color-text)] transition hover:bg-white/8"
                onClick={() => {
                  if (!customName.trim()) return;
                  setScenarios((prev) => [
                    ...prev,
                    {
                      name: customName.trim(),
                      vol_multiplier: Number(customVol) || 1.0,
                      shock_pnl: Number(customShock) || 0.0,
                    },
                  ]);
                  setCustomName("Custom");
                  setCustomVol("2.0");
                  setCustomShock("0");
                }}
              >
                Add scenario
              </button>
              <button
                type="button"
                className="inline-flex h-10 items-center justify-center rounded-full border border-white/12 bg-white/5 px-4 text-xs text-[var(--color-text-muted)] transition hover:bg-white/8"
                onClick={() => setScenarios(defaultScenarios)}
              >
                Reset to defaults
              </button>
            </div>
          </div>

          <div className="mt-6">
            <button
              type="button"
              className="inline-flex h-12 items-center justify-center rounded-full bg-[var(--color-accent)] px-5 text-sm font-semibold text-[#1a1206] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:shadow-[0_18px_44px_rgba(216,155,73,0.22)] disabled:opacity-50"
              disabled={mutation.isPending || scenarios.length === 0}
              onClick={() => mutation.mutate()}
            >
              {mutation.isPending ? "Running stress test..." : "Run stress test"}
            </button>
            {mutation.error && (
              <div className="mt-3 text-sm text-[var(--color-red)]">
                {mutation.error instanceof Error ? mutation.error.message : "Stress test failed."}
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        <div className="surface-strong rounded-[1.8rem] p-6">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Stress Report
          </div>

          {!report ? (
            <div className="mt-8 max-w-xl">
              <h3 className="text-2xl font-semibold text-white">No stress results yet.</h3>
              <p className="mt-3 text-sm leading-7 text-[var(--color-text-soft)]">
                Configure scenarios and run the stress test to see VaR/ES under each scenario compared to the baseline.
              </p>
              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                <FormMetaTile label="Baseline VaR" value="---" hint="Current portfolio" />
                <FormMetaTile label="Worst VaR" value="---" hint="Max stressed VaR" />
                <FormMetaTile label="VaR increase" value="---" hint="Worst vs baseline" />
              </div>
            </div>
          ) : (
            <StressResults report={report} />
          )}
        </div>
      </div>
    </div>
  );
}

function StressResults({ report }: { report: StressReportResponse }) {
  const worstScenario = useMemo(() => {
    if (report.scenarios.length === 0) return null;
    return report.scenarios.reduce((a, b) => (a.var > b.var ? a : b));
  }, [report.scenarios]);

  const varIncrease = worstScenario
    ? ((worstScenario.var - report.baseline_var) / report.baseline_var)
    : null;

  return (
    <div>
      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <FormMetaTile
          label="Baseline VaR"
          value={formatCurrency(report.baseline_var)}
          hint={`ES: ${formatCurrency(report.baseline_es)}`}
          tone="accent"
        />
        <FormMetaTile
          label="Worst VaR"
          value={worstScenario ? formatCurrency(worstScenario.var) : "n/a"}
          hint={worstScenario?.name ?? "n/a"}
          tone="danger"
        />
        <FormMetaTile
          label="VaR increase"
          value={varIncrease != null ? formatPercent(varIncrease, 0) : "n/a"}
          hint="Worst vs baseline"
          tone={varIncrease != null && varIncrease > 1.0 ? "danger" : "warning"}
        />
      </div>

      <div className="mt-6">
        <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
          Scenario breakdown
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/8 text-left text-xs text-[var(--color-text-muted)]">
                <th className="pb-3 pr-4 font-medium">Scenario</th>
                <th className="pb-3 pr-4 font-medium text-right">Vol mult.</th>
                <th className="pb-3 pr-4 font-medium text-right">Shock</th>
                <th className="pb-3 pr-4 font-medium text-right">VaR</th>
                <th className="pb-3 pr-4 font-medium text-right">ES</th>
                <th className="pb-3 font-medium text-right">vs Baseline</th>
              </tr>
            </thead>
            <tbody>
              {report.scenarios.map((sc) => {
                const ratio = report.baseline_var > 0 ? sc.var / report.baseline_var : 0;
                return (
                  <tr key={sc.name} className="border-b border-white/6">
                    <td className="py-3 pr-4 font-semibold text-white">{sc.name}</td>
                    <td className="py-3 pr-4 text-right text-[var(--color-text-soft)]">
                      &times;{sc.vol_multiplier.toFixed(1)}
                    </td>
                    <td className="py-3 pr-4 text-right text-[var(--color-text-soft)]">
                      {sc.shock_pnl !== 0 ? formatCurrency(sc.shock_pnl) : "—"}
                    </td>
                    <td className="py-3 pr-4 text-right font-semibold text-white">
                      {formatCurrency(sc.var)}
                    </td>
                    <td className="py-3 pr-4 text-right text-[var(--color-text-soft)]">
                      {formatCurrency(sc.es)}
                    </td>
                    <td className="py-3 text-right">
                      <StatusBadge
                        label={formatPercent(ratio, 0)}
                        tone={ratio > 3 ? "danger" : ratio > 2 ? "warning" : "success"}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-4 text-xs text-[var(--color-text-muted)]">
        Alpha: {formatPercent(report.alpha, 0)} — Portfolio: {report.portfolio_slug}
      </div>
    </div>
  );
}

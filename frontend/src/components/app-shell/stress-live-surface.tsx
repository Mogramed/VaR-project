"use client";

import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/app-shell/page-header";
import { FormMetaTile, FieldInput, FieldLabel } from "@/components/forms/shared";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { StressReportResponse, StressScenarioRequest } from "@/lib/api/types";
import { formatCurrency, formatPercent } from "@/lib/utils";

const defaults: StressScenarioRequest[] = [
  { name: "2008 Crisis", vol_multiplier: 3.0, shock_pnl: 0.0 },
  { name: "COVID Mar 2020", vol_multiplier: 4.0, shock_pnl: 0.0 },
  { name: "ECB Rate Shock", vol_multiplier: 2.0, shock_pnl: 0.0 },
  { name: "Mild Stress", vol_multiplier: 1.5, shock_pnl: 0.0 },
];

export function StressLiveSurface({ portfolioSlug }: { portfolioSlug: string }) {
  const [scenarios, setScenarios] = useState(defaults);
  const [name, setName] = useState("Custom");
  const [vol, setVol] = useState("2.0");
  const [shock, setShock] = useState("0");

  const mutation = useMutation({
    mutationFn: () => api.runStressTest({ portfolio_slug: portfolioSlug, scenarios: scenarios.length > 0 ? scenarios : undefined }),
  });

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Stress" title="Multi-scenario VaR stress analysis" />

      <div className="grid gap-4 xl:grid-cols-2">
        {/* Config */}
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Scenarios</h3>
          <div className="mt-3 space-y-1.5">
            {scenarios.map((sc, i) => (
              <div key={sc.name} className="flex items-center justify-between rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-xs">
                <span className="font-semibold text-[var(--color-text)]">{sc.name}</span>
                <div className="flex items-center gap-3">
                  <span className="text-[var(--color-text-muted)]">×{sc.vol_multiplier.toFixed(1)}</span>
                  {sc.shock_pnl !== 0 && <span className="text-[var(--color-text-muted)]">{sc.shock_pnl > 0 ? "+" : ""}{sc.shock_pnl.toFixed(0)}</span>}
                  <button type="button" className="text-[var(--color-text-muted)] hover:text-[var(--color-red)]" onClick={() => setScenarios((p) => p.filter((_, j) => j !== i))}>×</button>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 border-t border-[var(--color-border)] pt-4">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Add scenario</div>
            <div className="mt-2 grid gap-2 sm:grid-cols-3">
              <div><FieldLabel htmlFor="s-name">Name</FieldLabel><FieldInput id="s-name" value={name} onChange={(e) => setName(e.target.value)} /></div>
              <div><FieldLabel htmlFor="s-vol">Vol mult</FieldLabel><FieldInput id="s-vol" type="number" min="0.1" step="0.1" value={vol} onChange={(e) => setVol(e.target.value)} /></div>
              <div><FieldLabel htmlFor="s-shock">Shock EUR</FieldLabel><FieldInput id="s-shock" type="number" step="100" value={shock} onChange={(e) => setShock(e.target.value)} /></div>
            </div>
            <div className="mt-2 flex gap-2">
              <button type="button" className="h-7 rounded-[var(--radius-sm)] border border-[var(--color-border)] px-3 text-[11px] font-medium text-[var(--color-text)] transition hover:bg-[var(--color-surface-hover)]"
                onClick={() => { if (!name.trim()) return; setScenarios((p) => [...p, { name: name.trim(), vol_multiplier: Number(vol) || 1, shock_pnl: Number(shock) || 0 }]); setName("Custom"); setVol("2.0"); setShock("0"); }}>
                Add
              </button>
              <button type="button" className="h-7 rounded-[var(--radius-sm)] border border-[var(--color-border)] px-3 text-[11px] text-[var(--color-text-muted)] transition hover:bg-[var(--color-surface-hover)]"
                onClick={() => setScenarios(defaults)}>Reset</button>
            </div>
          </div>

          <div className="mt-4">
            <button type="button" disabled={mutation.isPending || scenarios.length === 0}
              onClick={() => mutation.mutate()}
              className="h-8 rounded-[var(--radius-md)] bg-[var(--color-accent)] px-4 text-[12px] font-semibold text-[#1a1206] transition hover:brightness-110 disabled:opacity-50">
              {mutation.isPending ? "Running..." : "Run stress test"}
            </button>
            {mutation.error && <div className="mt-2 text-[11px] text-[var(--color-red)]">{mutation.error instanceof Error ? mutation.error.message : "Failed"}</div>}
          </div>
        </div>

        {/* Results */}
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] p-4">
          <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Stress Report</h3>
          {!mutation.data ? (
            <p className="mt-3 text-xs text-[var(--color-text-muted)]">Configure scenarios and run the test to compare stressed VaR/ES vs baseline.</p>
          ) : (
            <StressResults report={mutation.data} />
          )}
        </div>
      </div>
    </div>
  );
}

function StressResults({ report }: { report: StressReportResponse }) {
  const worst = useMemo(() => report.scenarios.length === 0 ? null : report.scenarios.reduce((a, b) => a.var > b.var ? a : b), [report.scenarios]);
  const increase = worst ? (worst.var - report.baseline_var) / report.baseline_var : null;

  return (
    <div>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <FormMetaTile label="Baseline VaR" value={formatCurrency(report.baseline_var)} hint={`ES: ${formatCurrency(report.baseline_es)}`} tone="accent" />
        <FormMetaTile label="Worst VaR" value={worst ? formatCurrency(worst.var) : "n/a"} hint={worst?.name ?? "n/a"} tone="danger" />
        <FormMetaTile label="VaR increase" value={increase != null ? formatPercent(increase, 0) : "n/a"} tone={increase != null && increase > 1.0 ? "danger" : "warning"} />
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-[var(--color-border)] text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              <th className="pb-2 pr-3">Scenario</th>
              <th className="pb-2 pr-3 text-right">Vol</th>
              <th className="pb-2 pr-3 text-right">VaR</th>
              <th className="pb-2 pr-3 text-right">ES</th>
              <th className="pb-2 text-right">vs Base</th>
            </tr>
          </thead>
          <tbody>
            {report.scenarios.map((sc) => {
              const ratio = report.baseline_var > 0 ? sc.var / report.baseline_var : 0;
              return (
                <tr key={sc.name} className="border-b border-[var(--color-border)]">
                  <td className="py-2 pr-3 font-semibold text-[var(--color-text)]">{sc.name}</td>
                  <td className="py-2 pr-3 text-right text-[var(--color-text-soft)]">×{sc.vol_multiplier.toFixed(1)}</td>
                  <td className="py-2 pr-3 text-right font-semibold text-[var(--color-text)]">{formatCurrency(sc.var)}</td>
                  <td className="py-2 pr-3 text-right text-[var(--color-text-soft)]">{formatCurrency(sc.es)}</td>
                  <td className="py-2 text-right"><StatusBadge label={formatPercent(ratio, 0)} tone={ratio > 3 ? "danger" : ratio > 2 ? "warning" : "success"} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-3 text-[10px] text-[var(--color-text-muted)]">Alpha: {formatPercent(report.alpha, 0)} — {report.portfolio_slug}</div>
    </div>
  );
}

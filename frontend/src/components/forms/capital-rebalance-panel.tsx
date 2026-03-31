"use client";

import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { FieldInput, FieldLabel, FormMetaTile } from "@/components/forms/shared";
import { api } from "@/lib/api/client";
import type { CapitalUsageSnapshotResponse } from "@/lib/api/types";
import { formatCurrency, formatPercent } from "@/lib/utils";

export function CapitalRebalancePanel({
  portfolioSlug,
  referenceModel,
  onRebalanced,
}: {
  portfolioSlug: string;
  referenceModel: string;
  onRebalanced?: (result: CapitalUsageSnapshotResponse) => void;
}) {
  const router = useRouter();
  const [budget, setBudget] = useState("12000000");
  const [reserveRatio, setReserveRatio] = useState("0.18");

  const mutation = useMutation({
    mutationFn: async () =>
      api.rebalanceCapital({
        portfolio_slug: portfolioSlug,
        total_budget_eur: Number(budget),
        reserve_ratio: Number(reserveRatio),
        reference_model: referenceModel,
      }),
    onSuccess: (result) => {
      onRebalanced?.(result);
      if (!onRebalanced) router.refresh();
    },
  });

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <form className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
        onSubmit={(e) => { e.preventDefault(); mutation.mutate(); }}>
        <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Capital Rebalance</h3>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <div>
            <FieldLabel htmlFor="cap-budget">Budget EUR</FieldLabel>
            <FieldInput id="cap-budget" type="number" min="1" step="100000" value={budget} onChange={(e) => setBudget(e.target.value)} />
          </div>
          <div>
            <FieldLabel htmlFor="cap-reserve">Reserve ratio</FieldLabel>
            <FieldInput id="cap-reserve" type="number" min="0" max="1" step="0.01" value={reserveRatio} onChange={(e) => setReserveRatio(e.target.value)} />
          </div>
        </div>
        <div className="mt-3 flex items-center justify-between rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)]">Model</span>
          <span className="mono text-xs font-semibold text-[var(--color-text)]">{referenceModel.toUpperCase()}</span>
        </div>
        <div className="mt-4 flex items-center gap-2">
          <button type="submit" disabled={mutation.isPending}
            className="h-8 rounded-[var(--radius-md)] bg-[var(--color-accent)] px-4 text-[12px] font-semibold text-[#1a1206] transition hover:brightness-110 disabled:opacity-50">
            {mutation.isPending ? "Rebalancing..." : "Rebalance"}
          </button>
          {mutation.error ? (
            <span className="text-[11px] text-[var(--color-red)]">{mutation.error instanceof Error ? mutation.error.message : "Failed"}</span>
          ) : null}
        </div>
      </form>

      {/* Result */}
      {mutation.data ? (
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] p-4">
          <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Rebalance Result</h3>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            <FormMetaTile label="Budget" value={formatCurrency(mutation.data.total_capital_budget_eur)} />
            <FormMetaTile label="Consumed" value={formatCurrency(mutation.data.total_capital_consumed_eur)} tone="warning" />
            <FormMetaTile label="Reserved" value={formatCurrency(mutation.data.total_capital_reserved_eur)} />
            <FormMetaTile label="Headroom" value={formatPercent(mutation.data.headroom_ratio ?? 0, 0)} tone="success" />
          </div>
          {(mutation.data.recommendations ?? []).length > 0 ? (
            <div className="mt-3 space-y-2">
              {(mutation.data.recommendations ?? []).slice(0, 4).map((rec) => (
                <div key={`${rec.symbol_from}-${rec.symbol_to}`} className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-2.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-[var(--color-text)]">{rec.symbol_from} → {rec.symbol_to}</span>
                    <span className="mono text-[var(--color-accent)]">{formatCurrency(rec.amount_eur)}</span>
                  </div>
                  <p className="mt-1 text-[11px] text-[var(--color-text-muted)]">{rec.reason}</p>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : (
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
          <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Output</h3>
          <p className="mt-2 text-xs text-[var(--color-text-muted)]">Submit a rebalance to see the new headroom posture.</p>
        </div>
      )}
    </div>
  );
}

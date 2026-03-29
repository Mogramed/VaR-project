"use client";

import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { FieldInput, FieldLabel } from "@/components/forms/shared";
import { api } from "@/lib/api/client";
import type { CapitalUsageSnapshotResponse } from "@/lib/api/types";
import { formatCurrency, formatPercent } from "@/lib/utils";

export function CapitalRebalancePanel({
  portfolioSlug,
  referenceModel,
}: {
  portfolioSlug: string;
  referenceModel: string;
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
    onSuccess: () => router.refresh(),
  });

  return (
    <div className="grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
      <form
        className="surface rounded-[1.8rem] p-6"
        onSubmit={(event) => {
          event.preventDefault();
          mutation.mutate();
        }}
      >
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Capital Rebalance
        </div>
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <div>
            <FieldLabel htmlFor="capital-budget">Total budget EUR</FieldLabel>
            <FieldInput
              id="capital-budget"
              type="number"
              min="1"
              step="100000"
              value={budget}
              onChange={(event) => setBudget(event.target.value)}
            />
          </div>
          <div>
            <FieldLabel htmlFor="capital-reserve">Reserve ratio</FieldLabel>
            <FieldInput
              id="capital-reserve"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={reserveRatio}
              onChange={(event) => setReserveRatio(event.target.value)}
            />
          </div>
        </div>
        <div className="mt-4 rounded-[1.3rem] border border-white/8 bg-black/18 p-4">
          <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
            Reference model
          </div>
          <div className="mt-2 text-base font-semibold text-white">
            {referenceModel.toUpperCase()}
          </div>
        </div>
        <div className="mt-6 flex items-center gap-3">
          <button
            type="submit"
            className="inline-flex h-12 items-center justify-center rounded-full bg-[var(--color-accent)] px-5 text-sm font-semibold text-[#1a1206] transition hover:translate-y-[-1px]"
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "Rebalancing..." : "Rebalance capital"}
          </button>
          {mutation.error ? (
            <div className="text-sm text-[var(--color-red)]">
              {mutation.error instanceof Error
                ? mutation.error.message
                : "Request failed."}
            </div>
          ) : null}
        </div>
      </form>
      <CapitalRebalanceResult result={mutation.data} />
    </div>
  );
}

function CapitalRebalanceResult({
  result,
}: {
  result: CapitalUsageSnapshotResponse | undefined;
}) {
  if (!result) {
    return (
      <div className="surface rounded-[1.8rem] p-6">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Output
        </div>
        <div className="mt-10 max-w-md">
          <h3 className="text-2xl font-semibold text-white">
            No rebalance submitted yet.
          </h3>
          <p className="mt-3 text-sm leading-7 text-[var(--color-text-soft)]">
            Use this panel to stress a new capital budget and review the resulting
            headroom posture.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="surface-strong rounded-[1.8rem] p-6">
      <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
        Rebalance Snapshot
      </div>
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {[
          ["Budget", formatCurrency(result.total_capital_budget_eur)],
          ["Consumed", formatCurrency(result.total_capital_consumed_eur)],
          ["Reserved", formatCurrency(result.total_capital_reserved_eur)],
          ["Headroom", formatPercent(result.headroom_ratio ?? 0, 0)],
        ].map(([label, value]) => (
          <div
            key={label}
            className="rounded-[1.3rem] border border-white/8 bg-black/18 p-4"
          >
            <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
              {label}
            </div>
            <div className="mt-3 text-xl font-semibold text-white">{value}</div>
          </div>
        ))}
      </div>
      <div className="mt-6 space-y-3">
        {(result.recommendations ?? []).slice(0, 4).map((recommendation) => (
          <div
            key={`${recommendation.symbol_from}-${recommendation.symbol_to}`}
            className="rounded-[1.2rem] border border-white/8 bg-black/14 p-4"
          >
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-semibold text-white">
                {recommendation.symbol_from}{" -> "}{recommendation.symbol_to}
              </div>
              <div className="mono text-sm text-[var(--color-accent)]">
                {formatCurrency(recommendation.amount_eur)}
              </div>
            </div>
            <div className="mt-2 text-sm text-[var(--color-text-soft)]">
              {recommendation.reason}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

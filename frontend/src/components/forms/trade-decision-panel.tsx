"use client";

import { useMutation } from "@tanstack/react-query";
import { TrendingUp, DollarSign } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { api } from "@/lib/api/client";
import type { RiskDecisionResponse } from "@/lib/api/types";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { ButtonLink, StatusBadge } from "@/components/ui/primitives";
import {
  FieldInputWithIcon,
  FieldLabel,
  FieldSelect,
  FieldTextarea,
  FormError,
  FormMetaTile,
  FormSection,
  PresetPill,
  SubmitButton,
} from "@/components/forms/shared";

const decisionPresets = ["500000", "1000000", "2500000", "5000000"];

export function TradeDecisionPanel({
  portfolioSlug,
  onEvaluated,
}: {
  portfolioSlug: string;
  onEvaluated?: (result: RiskDecisionResponse) => void;
}) {
  const router = useRouter();
  const [symbol, setSymbol] = useState("EURUSD");
  const [side, setSide] = useState("buy");
  const [exposureChange, setExposureChange] = useState("2500000");
  const [note, setNote] = useState("");

  const mutation = useMutation({
    mutationFn: async () =>
      api.evaluateDecision({
        portfolio_slug: portfolioSlug,
        symbol,
        exposure_change: (side === "buy" ? 1 : -1) * Number(exposureChange),
        note,
      }),
    onSuccess: (result) => {
      onEvaluated?.(result);
      if (!onEvaluated) router.refresh();
    },
  });

  const result = mutation.data;
  const fillRatio = useMemo(() => {
    if (!result || result.requested_exposure_change === 0) return null;
    return Math.abs(result.approved_exposure_change / result.requested_exposure_change);
  }, [result]);

  return (
    <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
      {/* Form */}
      <form
        className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
        onSubmit={(e) => { e.preventDefault(); mutation.mutate(); }}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Advisory Decision</h3>
          <span className="text-[10px] text-[var(--color-text-muted)]">Advisory only</span>
        </div>

        <FormSection title="Trade parameters">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <FieldLabel htmlFor="dec-symbol">Symbol</FieldLabel>
              <FieldInputWithIcon icon={TrendingUp} id="dec-symbol" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
            </div>
            <div>
              <FieldLabel htmlFor="dec-side">Side</FieldLabel>
              <FieldSelect id="dec-side" value={side} onChange={(e) => setSide(e.target.value)}>
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </FieldSelect>
            </div>
          </div>
        </FormSection>

        <FormSection title="Sizing">
          <FieldLabel htmlFor="dec-exposure">Target exposure change</FieldLabel>
          <FieldInputWithIcon icon={DollarSign} id="dec-exposure" type="number" min="1" step="1000" value={exposureChange} onChange={(e) => setExposureChange(e.target.value)} />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {decisionPresets.map((p) => (
              <PresetPill key={p} active={p === exposureChange} onClick={() => setExposureChange(p)}>{formatCurrency(Number(p))}</PresetPill>
            ))}
          </div>
          <p className="mt-2 text-[11px] text-[var(--color-text-muted)]">
            Advisory mode works from the same exposure language as execution. The desk translates this target into
            post-trade risk, budget usage and the likely broker order footprint.
          </p>
        </FormSection>

        <FormSection>
          <FieldLabel htmlFor="dec-note">Note</FieldLabel>
          <FieldTextarea id="dec-note" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Optional context" />
        </FormSection>

        <div className="mt-4 flex items-center gap-2">
          <SubmitButton isPending={mutation.isPending} label="Evaluate" pendingLabel="Evaluating..." />
          <FormError message={mutation.error instanceof Error ? mutation.error.message : mutation.error ? "Failed" : null} />
        </div>
      </form>

      {/* Result */}
      <DecisionResult result={result} fillRatio={fillRatio} />
    </div>
  );
}

function DecisionResult({ result, fillRatio }: { result: RiskDecisionResponse | undefined; fillRatio: number | null }) {
  if (!result) {
    return (
      <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
        <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Output</h3>
        <p className="mt-2 text-xs text-[var(--color-text-muted)]">Submit a trade proposal to see the verdict.</p>
      </div>
    );
  }

  return (
    <div className="rounded-[var(--radius-lg)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] p-4">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Verdict</span>
          <div className="mt-1 text-2xl font-semibold text-[var(--color-text)]">{result.decision}</div>
        </div>
        <StatusBadge label={result.model_used.toUpperCase()} tone="accent" />
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-4">
        <FormMetaTile label="Requested exposure" value={formatCurrency(result.requested_exposure_change)} />
        <FormMetaTile label="Approved exposure" value={formatCurrency(result.approved_exposure_change)} tone="success" />
        <FormMetaTile label="Resulting exposure" value={formatCurrency(result.resulting_exposure)} />
        <FormMetaTile label="Fill" value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)} />
      </div>

      {result.suggested_exposure_change != null && result.suggested_exposure_change !== result.approved_exposure_change ? (
        <div className="mt-3 rounded-[var(--radius-md)] border border-[var(--color-amber)]/20 bg-[var(--color-amber-soft)] p-2.5 text-[11px] text-[var(--color-text-soft)]">
          Suggested: {formatCurrency(result.suggested_exposure_change)}
        </div>
      ) : null}

      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <StatePanel title="Pre-trade" state={result.pre_trade} />
        <StatePanel title="Post-trade" state={result.post_trade} />
      </div>

      {result.reasons.length > 0 ? (
        <div className="mt-3 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Rationale</div>
          <div className="mt-1.5 space-y-1">
            {result.reasons.map((r) => (
              <div key={r} className="flex gap-2 text-[11px] text-[var(--color-text-soft)]">
                <span className="mt-1.5 size-1 shrink-0 rounded-full bg-[var(--color-accent)]" />
                {r}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="mt-4">
        <ButtonLink href="/desk/blotter" variant="secondary">Continue to blotter</ButtonLink>
      </div>
    </div>
  );
}

function StatePanel({ title, state }: { title: string; state: RiskDecisionResponse["pre_trade"] }) {
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
      <div className="text-[11px] font-semibold text-[var(--color-text)]">{title}</div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
        {([
          ["VaR", formatCurrency(state.var)],
          ["ES", formatCurrency(state.es)],
          ["Budget", state.budget_utilization_var == null ? "n/a" : formatPercent(state.budget_utilization_var, 0)],
          ["Headroom", formatCurrency(state.headroom_var)],
        ] as const).map(([label, value]) => (
          <div key={label}>
            <span className="text-[var(--color-text-muted)]">{label}</span>
            <div className="mono font-semibold text-[var(--color-text)]">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

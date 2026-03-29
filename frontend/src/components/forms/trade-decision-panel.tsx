"use client";

import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { api } from "@/lib/api/client";
import type { RiskDecisionResponse } from "@/lib/api/types";
import { formatCurrency, formatPercent, formatSignedCurrency } from "@/lib/utils";
import { ButtonLink } from "@/components/ui/primitives";
import {
  FieldInput,
  FieldHint,
  FieldLabel,
  FieldSelect,
  FieldTextarea,
  FormMetaTile,
  PresetPill,
} from "@/components/forms/shared";

const decisionPresets = ["500000", "1000000", "2500000", "5000000"];

export function TradeDecisionPanel({
  portfolioSlug,
}: {
  portfolioSlug: string;
}) {
  const router = useRouter();
  const [symbol, setSymbol] = useState("EURUSD");
  const [side, setSide] = useState("buy");
  const [notional, setNotional] = useState("2500000");
  const [note, setNote] = useState("");

  const mutation = useMutation({
    mutationFn: async () =>
      api.evaluateDecision({
        portfolio_slug: portfolioSlug,
        symbol,
        delta_position_eur: (side === "buy" ? 1 : -1) * Number(notional),
        note,
      }),
    onSuccess: () => router.refresh(),
  });

  const result = mutation.data;
  const fillRatio = useMemo(() => {
    if (!result || result.requested_delta_position_eur === 0) {
      return null;
    }
    return Math.abs(
      result.approved_delta_position_eur / result.requested_delta_position_eur,
    );
  }, [result]);
  const signedDelta =
    (side === "buy" ? 1 : -1) * Number(notional || 0);

  return (
    <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
      <form
        className="surface rounded-[1.8rem] p-6"
        onSubmit={(event) => {
          event.preventDefault();
          mutation.mutate();
        }}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Advisory Decision
            </div>
            <h3 className="mt-3 text-2xl font-semibold tracking-[-0.04em] text-white">
              Evaluate the trade before it touches the desk.
            </h3>
            <p className="mt-2 max-w-xl text-sm leading-7 text-[var(--color-text-soft)]">
              Submit one proposed trade and keep the sizing, sign and advisory-only
              posture visible while the decision engine computes the verdict.
            </p>
          </div>
          <div className="hidden xl:block">
            <div className="rounded-full border border-white/10 px-3 py-1 text-xs uppercase tracking-[0.24em] text-[var(--color-text-soft)]">
              Advisory only
            </div>
          </div>
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <FormMetaTile
            label="Portfolio"
            value={portfolioSlug}
            hint="Current decision scope"
            tone="accent"
          />
          <FormMetaTile
            label="Signed delta"
            value={formatSignedCurrency(signedDelta)}
            hint={side === "buy" ? "Risk-increasing direction" : "Risk-reducing direction"}
            tone={side === "buy" ? "warning" : "success"}
          />
          <FormMetaTile
            label="Execution gate"
            value="ACCEPT / REDUCE / REJECT"
            hint="No live execution is triggered here"
          />
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <div>
            <FieldLabel htmlFor="decision-symbol">Symbol</FieldLabel>
            <FieldInput
              id="decision-symbol"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value.toUpperCase())}
              placeholder="EURUSD"
            />
            <FieldHint>Use a desk symbol like `EURUSD`, `USDJPY` or `GBPUSD`.</FieldHint>
          </div>
          <div>
            <FieldLabel htmlFor="decision-side">Side</FieldLabel>
            <FieldSelect
              id="decision-side"
              value={side}
              onChange={(event) => setSide(event.target.value)}
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </FieldSelect>
            <FieldHint>The sign of the delta updates automatically from the side.</FieldHint>
          </div>
          <div>
            <FieldLabel htmlFor="decision-notional">Notional EUR</FieldLabel>
            <FieldInput
              id="decision-notional"
              type="number"
              min="1"
              step="1000"
              value={notional}
              onChange={(event) => setNotional(event.target.value)}
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {decisionPresets.map((preset) => (
                <PresetPill
                  key={preset}
                  active={preset === notional}
                  onClick={() => setNotional(preset)}
                >
                  {formatCurrency(Number(preset))}
                </PresetPill>
              ))}
            </div>
          </div>
          <div className="flex items-end">
            <div className="w-full rounded-[1.4rem] border border-white/8 bg-white/[0.03] px-4 py-4">
              <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                Routing note
              </div>
              <div className="mt-2 text-base font-semibold text-white">
                {side === "buy" ? "Pressure added to the desk" : "Pressure released from the desk"}
              </div>
              <div className="mt-2 text-xs leading-6 text-[var(--color-text-muted)]">
                The decision layer will still accept a risk-reducing trade even when the
                portfolio is already tense.
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4">
          <FieldLabel htmlFor="decision-note">Operator note</FieldLabel>
          <FieldTextarea
            id="decision-note"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="Optional context for the proposal."
          />
          <FieldHint>
            Helpful for audit trail and later report narrative, especially on clipped trades.
          </FieldHint>
        </div>
        <div className="mt-6 flex items-center gap-3">
          <button
            type="submit"
            className="inline-flex h-12 items-center justify-center rounded-full bg-[var(--color-accent)] px-5 text-sm font-semibold text-[#1a1206] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:shadow-[0_18px_44px_rgba(216,155,73,0.22)]"
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "Evaluating..." : "Evaluate trade"}
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

      <DecisionResult result={result} fillRatio={fillRatio} />
    </div>
  );
}

function DecisionResult({
  result,
  fillRatio,
}: {
  result: RiskDecisionResponse | undefined;
  fillRatio: number | null;
}) {
  if (!result) {
    return (
      <div className="surface rounded-[1.8rem] p-6">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Output
        </div>
        <div className="mt-8 max-w-xl">
          <h3 className="text-2xl font-semibold text-white">
            No decision evaluated yet.
          </h3>
          <p className="mt-3 text-sm leading-7 text-[var(--color-text-soft)]">
            Submit a trade proposal to render the accept, reduce or reject verdict
            with its before/after risk posture.
          </p>
          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            <FormMetaTile label="Verdict" value="Pending" hint="No evaluation yet" />
            <FormMetaTile label="Sizing" value="Requested vs approved" hint="Fill ratio will appear here" />
            <FormMetaTile label="Risk state" value="Pre / post" hint="VaR, ES and headroom" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="surface-strong rounded-[1.8rem] p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Verdict
          </div>
          <div className="mt-3 text-4xl font-semibold tracking-[-0.05em] text-white">
            {result.decision}
          </div>
        </div>
        <div className="rounded-full border border-white/10 px-3 py-1 text-xs uppercase tracking-[0.24em] text-[var(--color-text-soft)]">
          {result.model_used}
        </div>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-4">
        {[
          ["Requested delta", formatCurrency(result.requested_delta_position_eur)],
          ["Approved delta", formatCurrency(result.approved_delta_position_eur)],
          ["Resulting position", formatCurrency(result.resulting_position_eur)],
          ["Fill ratio", fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)],
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

      {result.suggested_delta_position_eur != null &&
      result.suggested_delta_position_eur !== result.approved_delta_position_eur ? (
        <div className="mt-4 rounded-[1.35rem] border border-amber-300/16 bg-amber-300/8 px-4 py-4 text-sm text-[var(--color-text-soft)]">
          Suggested signed delta {formatCurrency(result.suggested_delta_position_eur)} if you
          want the desk to stay within the same advisory envelope.
        </div>
      ) : null}

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <StatePanel title="Pre-trade" state={result.pre_trade} />
        <StatePanel title="Post-trade" state={result.post_trade} />
      </div>
      <div className="mt-6 border-t border-white/8 pt-5">
        <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
          Rationale
        </div>
        <ul className="mt-3 space-y-3 text-sm leading-7 text-[var(--color-text-soft)]">
          {result.reasons.map((reason) => (
            <li key={reason} className="flex gap-3">
              <span className="mt-[0.72rem] size-1.5 rounded-full bg-[var(--color-accent)]" />
              <span>{reason}</span>
            </li>
          ))}
        </ul>
      </div>
      <div className="mt-6">
        <ButtonLink href="/desk/simulation" variant="secondary">
          Continue to simulation
        </ButtonLink>
      </div>
    </div>
  );
}

function StatePanel({
  title,
  state,
}: {
  title: string;
  state: RiskDecisionResponse["pre_trade"];
}) {
  return (
    <div className="rounded-[1.4rem] border border-white/8 bg-black/18 p-4">
      <div className="text-sm font-semibold text-white">{title}</div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {[
          ["VaR", formatCurrency(state.var)],
          ["ES", formatCurrency(state.es)],
          [
            "Budget util.",
            state.budget_utilization_var == null
              ? "n/a"
              : formatPercent(state.budget_utilization_var, 0),
          ],
          ["Headroom", formatCurrency(state.headroom_var)],
          ["Gross", formatCurrency(state.gross_notional)],
          ["Status", state.status],
        ].map(([label, value]) => (
          <div key={label} className="border-t border-white/8 pt-3">
            <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
              {label}
            </div>
            <div className="mt-2 text-base font-semibold text-white">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

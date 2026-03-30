"use client";

import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { api } from "@/lib/api/client";
import type {
  ExecutionPreviewResponse,
  ExecutionResultResponse,
  MT5TerminalStatusResponse,
} from "@/lib/api/types";
import { formatCurrency, formatPercent, formatSignedCurrency } from "@/lib/utils";
import { StatusBadge } from "@/components/ui/primitives";
import {
  FieldHint,
  FieldInput,
  FieldLabel,
  FieldSelect,
  FieldTextarea,
  FormMetaTile,
  PresetPill,
} from "@/components/forms/shared";

const executionPresets = ["250000", "500000", "1000000", "2500000"];

function safeNum(obj: Record<string, unknown> | undefined, key: string): number | null {
  if (!obj) return null;
  const v = obj[key];
  return typeof v === "number" ? v : null;
}

export function ExecutionPanel({
  portfolioSlug,
  terminalStatus,
  onSubmitted,
}: {
  portfolioSlug: string;
  terminalStatus: MT5TerminalStatusResponse;
  onSubmitted?: (result: ExecutionResultResponse) => void;
}) {
  const router = useRouter();
  const [symbol, setSymbol] = useState("EURUSD");
  const [side, setSide] = useState("buy");
  const [notional, setNotional] = useState("500000");
  const [note, setNote] = useState("");

  const previewMutation = useMutation({
    mutationFn: async () =>
      api.previewExecution({
        portfolio_slug: portfolioSlug,
        symbol,
        exposure_change: (side === "buy" ? 1 : -1) * Number(notional),
        note,
      }),
  });

  const submitMutation = useMutation({
    mutationFn: async () =>
      api.submitExecution({
        portfolio_slug: portfolioSlug,
        symbol,
        exposure_change: (side === "buy" ? 1 : -1) * Number(notional),
        note,
      }),
    onSuccess: (payload) => {
      onSubmitted?.(payload);
      if (!onSubmitted) {
        router.refresh();
      }
    },
  });

  const preview = previewMutation.data;
  const result = submitMutation.data;
  const signedDelta = (side === "buy" ? 1 : -1) * Number(notional || 0);
  const activeError = submitMutation.error ?? previewMutation.error;
  const fillRatio = useMemo(() => {
    const source = result?.guard ?? preview?.guard;
    if (!source || source.requested_exposure_change === 0) {
      return null;
    }
    return Math.abs(
      source.executable_exposure_change / source.requested_exposure_change,
    );
  }, [preview, result]);

  return (
    <div className="grid gap-6 xl:grid-cols-[0.96fr_1.04fr]">
      <form
        className="surface rounded-[1.8rem] p-6"
        onSubmit={(event) => {
          event.preventDefault();
          previewMutation.mutate();
        }}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              MT5 Demo Execution
            </div>
            <h3 className="mt-3 text-2xl font-semibold tracking-[-0.04em] text-white">
              Preview first, then route to MetaTrader 5.
            </h3>
            <p className="mt-2 max-w-xl text-sm leading-7 text-[var(--color-text-soft)]">
              The order is never sent blindly: the desk computes the risk verdict, broker lot
              sizing and margin check before the operator can confirm the demo execution.
            </p>
          </div>
          <StatusBadge label={terminalStatus.ready ? "Ready" : "Guarded"} tone={terminalStatus.ready ? "success" : "warning"} />
        </div>

        {!terminalStatus.ready ? (
          <div className="mt-5 rounded-[1.4rem] border border-amber-300/16 bg-amber-300/8 px-4 py-4 text-sm leading-7 text-[var(--color-text-soft)]">
            {terminalStatus.message}
          </div>
        ) : null}

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <FormMetaTile label="Portfolio" value={portfolioSlug} hint="Execution scope" tone="accent" />
          <FormMetaTile
            label="Signed exposure"
            value={formatSignedCurrency(signedDelta)}
            hint="Request sent to the guard"
            tone={side === "buy" ? "warning" : "success"}
          />
          <FormMetaTile
            label="Workflow"
            value="Preview -> Submit"
            hint="No direct terminal access from the browser"
          />
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <div>
            <FieldLabel htmlFor="execution-symbol">Symbol</FieldLabel>
            <FieldInput
              id="execution-symbol"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value.toUpperCase())}
            />
            <FieldHint>Keep the ticket inside the current FX portfolio universe.</FieldHint>
          </div>
          <div>
            <FieldLabel htmlFor="execution-side">Side</FieldLabel>
            <FieldSelect
              id="execution-side"
              value={side}
              onChange={(event) => setSide(event.target.value)}
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </FieldSelect>
            <FieldHint>Direction controls the sign of the requested exposure change.</FieldHint>
          </div>
          <div>
            <FieldLabel htmlFor="execution-notional">Target exposure change</FieldLabel>
            <FieldInput
              id="execution-notional"
              type="number"
              min="1"
              step="1000"
              value={notional}
              onChange={(event) => setNotional(event.target.value)}
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {executionPresets.map((preset) => (
                <PresetPill
                  key={preset}
                  active={preset === notional}
                  onClick={() => setNotional(preset)}
                >
                  {formatCurrency(Number(preset))}
                </PresetPill>
              ))}
            </div>
            <FieldHint>
              Enter the target base-currency exposure change. Guard computes broker lots, margin and VaR impact before confirmation.
            </FieldHint>
          </div>
          <div className="flex items-end">
            <div className="w-full rounded-[1.4rem] border border-white/8 bg-white/[0.03] px-4 py-4">
              <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                Guard output
              </div>
              <div className="mt-2 text-base font-semibold text-white">
                Risk decision + broker lot sizing + margin check
              </div>
              <div className="mt-2 text-xs leading-6 text-[var(--color-text-muted)]">
                The platform can reduce or block the order before it reaches MT5.
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <FieldLabel htmlFor="execution-note">Operator note</FieldLabel>
          <FieldTextarea
            id="execution-note"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="Optional context before routing to the demo account."
          />
          <FieldHint>Stored in the audit trail and attached to the MT5 comment when possible.</FieldHint>
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-3">
          <button
            type="submit"
            className="inline-flex h-12 items-center justify-center rounded-full bg-[var(--color-accent)] px-5 text-sm font-semibold text-[#1a1206] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:shadow-[0_18px_44px_rgba(216,155,73,0.22)]"
            disabled={previewMutation.isPending}
          >
            {previewMutation.isPending ? "Previewing..." : "Preview with guard"}
          </button>
          <button
            type="button"
            className="inline-flex h-12 items-center justify-center rounded-full border border-white/12 bg-white/5 px-5 text-sm font-semibold text-[var(--color-text)] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:border-[var(--color-border-strong)] hover:bg-white/8 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!preview?.guard.submit_allowed || submitMutation.isPending}
            onClick={() => submitMutation.mutate()}
          >
            {submitMutation.isPending ? "Sending..." : "Send to MT5"}
          </button>
          {activeError ? (
            <div className="text-sm text-[var(--color-red)]">
              {activeError instanceof Error
                ? activeError.message
                : "Request failed."}
            </div>
          ) : null}
        </div>
      </form>

      <ExecutionOutput preview={preview} result={result} fillRatio={fillRatio} />
    </div>
  );
}

function ExecutionOutput({
  preview,
  result,
  fillRatio,
}: {
  preview: ExecutionPreviewResponse | undefined;
  result: ExecutionResultResponse | undefined;
  fillRatio: number | null;
}) {
  const guard = result?.guard ?? preview?.guard;

  if (!guard) {
    return (
      <div className="surface rounded-[1.8rem] p-6">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Execution Output
        </div>
        <div className="mt-8 max-w-xl">
          <h3 className="text-2xl font-semibold text-white">No MT5 preview yet.</h3>
          <p className="mt-3 text-sm leading-7 text-[var(--color-text-soft)]">
            Launch a guarded preview to see the risk decision, executable size, lot conversion and margin posture before sending anything to the demo account.
          </p>
        </div>
        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          <FormMetaTile label="Lots" value="---" hint="Broker volume" />
          <FormMetaTile label="Margin" value="---" hint="Required EUR" />
          <FormMetaTile label="VaR delta" value="---" hint="Pre vs post-trade" />
        </div>
      </div>
    );
  }

  const orderRequest = preview?.order_request ?? result?.order_request;
  const account = preview?.account ?? result?.account_before;
  const riskDecision = preview?.risk_decision ?? result?.risk_decision;

  const brokerPrice = safeNum(orderRequest, "price");
  const eurPerLot = safeNum(orderRequest, "eur_per_lot");
  const minNotional = safeNum(orderRequest, "minimum_notional_eur");

  const preTadeVar = riskDecision?.pre_trade?.var ?? null;
  const postTradeVar = riskDecision?.post_trade?.var ?? null;
  const varDelta = preTadeVar != null && postTradeVar != null ? postTradeVar - preTadeVar : null;
  const budgetUtil = riskDecision?.post_trade?.budget_utilization_var ?? null;

  return (
    <div className="surface-strong rounded-[1.8rem] p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Guard Verdict
          </div>
          <div className="mt-3 text-4xl font-semibold tracking-[-0.05em] text-white">
            {guard.decision}
          </div>
        </div>
        <StatusBadge label={result?.status ?? "Preview only"} tone={result ? "accent" : "neutral"} />
      </div>

      {/* Summary row */}
      <div className="mt-6 grid gap-4 sm:grid-cols-4">
        <FormMetaTile label="Model" value={guard.model_used.toUpperCase()} hint="Decision reference" tone="accent" />
        <FormMetaTile label="Executable exposure" value={formatCurrency(guard.executable_exposure_change)} hint={`Approved ${formatCurrency(guard.approved_exposure_change)}`} tone="success" />
        <FormMetaTile label="Lots" value={guard.volume_lots.toFixed(2)} hint={guard.side ?? "n/a"} />
        <FormMetaTile label="Fill ratio" value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)} hint="Executable vs requested" />
      </div>

      {/* Broker Ticket */}
      <div className="mt-6">
        <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
          Broker Ticket
        </div>
        <div className="mt-3 grid gap-4 sm:grid-cols-4">
          <FormMetaTile
            label="Volume"
            value={`${guard.volume_lots.toFixed(2)} lots`}
            hint={guard.side ?? "n/a"}
          />
          <FormMetaTile
            label="Price"
            value={brokerPrice != null ? brokerPrice.toFixed(5) : "n/a"}
            hint="Indicative at preview"
          />
          <FormMetaTile
            label="Exposure / lot"
            value={eurPerLot != null ? formatCurrency(eurPerLot) : "n/a"}
            hint="Contract notional"
          />
          <FormMetaTile
            label="Min exposure"
            value={minNotional != null ? formatCurrency(minNotional) : "n/a"}
            hint="Broker minimum"
          />
        </div>
      </div>

      {/* VaR Impact */}
      <div className="mt-6">
        <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
          VaR Impact
        </div>
        <div className="mt-3 grid gap-4 sm:grid-cols-4">
          <FormMetaTile
            label="Pre-trade VaR"
            value={preTadeVar != null ? formatCurrency(preTadeVar) : "n/a"}
            hint="Current portfolio"
          />
          <FormMetaTile
            label="Post-trade VaR"
            value={postTradeVar != null ? formatCurrency(postTradeVar) : "n/a"}
            hint="After this trade"
          />
          <FormMetaTile
            label="VaR delta"
            value={varDelta != null ? formatSignedCurrency(varDelta) : "n/a"}
            hint="Risk added / removed"
            tone={varDelta == null ? "neutral" : varDelta > 0 ? "warning" : "success"}
          />
          <FormMetaTile
            label="Budget util."
            value={budgetUtil != null ? formatPercent(budgetUtil, 1) : "n/a"}
            hint="Post-trade utilisation"
            tone={
              budgetUtil == null
                ? "neutral"
                : budgetUtil > 0.9
                ? "danger"
                : budgetUtil > 0.7
                ? "warning"
                : "success"
            }
          />
        </div>
      </div>

      {/* Margin Check + Terminal */}
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
          <div className="flex items-center gap-3">
            <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
              Margin Check
            </div>
            <StatusBadge label={guard.margin_ok ? "Pass" : "Fail"} tone={guard.margin_ok ? "success" : "danger"} />
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <div>
              <div className="text-xs text-[var(--color-text-muted)]">Required</div>
              <div className="mt-1 text-sm font-semibold text-white">
                {formatCurrency(guard.margin_required, 2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-[var(--color-text-muted)]">Free after</div>
              <div className="mt-1 text-sm font-semibold text-white">
                {formatCurrency(guard.free_margin_after, 2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-[var(--color-text-muted)]">Equity</div>
              <div className="mt-1 text-sm font-semibold text-white">
                {account?.equity != null ? formatCurrency(account.equity, 2) : "n/a"}
              </div>
            </div>
            <div>
              <div className="text-xs text-[var(--color-text-muted)]">Margin level</div>
              <div className="mt-1 text-sm font-semibold text-white">
                {account?.margin_level != null ? formatPercent(account.margin_level / 100, 0) : "n/a"}
              </div>
            </div>
          </div>
        </div>
        <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
          <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
            Terminal
          </div>
          <div className="mt-3 flex items-center gap-3">
            <StatusBadge label={guard.submit_allowed ? "Submit enabled" : "Blocked"} tone={guard.submit_allowed ? "success" : "warning"} />
            <span className="text-sm text-[var(--color-text-soft)]">
              {guard.execution_enabled ? "Kill switch on" : "Kill switch off"}
            </span>
          </div>
        </div>
      </div>

      {/* Reasons */}
      <div className="mt-6 rounded-[1.35rem] border border-white/8 bg-black/18 px-4 py-4 text-sm leading-7 text-[var(--color-text-soft)]">
        <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
          Reasons
        </div>
        <ul className="mt-3 space-y-2">
          {(guard.reasons ?? []).map((reason) => (
            <li key={reason} className="flex gap-3">
              <span className="mt-[0.72rem] size-1.5 rounded-full bg-[var(--color-accent)]" />
              <span>{reason}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* MT5 result banner */}
      {result ? (
        <div className="mt-6 rounded-[1.35rem] border border-emerald-300/16 bg-emerald-300/8 px-4 py-4 text-sm leading-7 text-[var(--color-text-soft)]">
          MT5 returned {String(result.mt5_result?.retcode ?? "n/a")} and the execution status is <span className="font-semibold text-white">{result.status}</span>.
        </div>
      ) : null}

      {/* Post-fill tiles */}
      {result ? (
        <div className="mt-6 grid gap-4 sm:grid-cols-4">
          <FormMetaTile
            label="Broker"
            value={result.broker_status ?? "n/a"}
            hint={result.reconciliation_status ?? "No reconciliation yet"}
            tone={result.status === "EXECUTED" ? "success" : result.status === "FAILED" ? "warning" : "accent"}
          />
          <FormMetaTile
            label="Filled lots"
            value={(result.filled_volume_lots ?? 0).toFixed(2)}
            hint={`Remaining ${(result.remaining_volume_lots ?? 0).toFixed(2)}`}
            tone="accent"
          />
          <FormMetaTile
            label="Fill ratio"
            value={result.fill_ratio == null ? "n/a" : formatPercent(result.fill_ratio, 0)}
            hint={`Submitted ${(result.submitted_volume_lots ?? 0).toFixed(2)} lots`}
            tone="success"
          />
          <FormMetaTile
            label="Slippage"
            value={result.slippage_points == null ? "n/a" : result.slippage_points.toFixed(1)}
            hint={`Position ${result.position_id ?? "n/a"}`}
          />
        </div>
      ) : null}
    </div>
  );
}

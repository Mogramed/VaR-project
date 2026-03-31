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
      if (!onSubmitted) router.refresh();
    },
  });

  const preview = previewMutation.data;
  const result = submitMutation.data;
  const activeError = submitMutation.error ?? previewMutation.error;
  const fillRatio = useMemo(() => {
    const source = result?.guard ?? preview?.guard;
    if (!source || source.requested_exposure_change === 0) return null;
    return Math.abs(source.executable_exposure_change / source.requested_exposure_change);
  }, [preview, result]);

  return (
    <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
      {/* Form */}
      <form
        className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4"
        onSubmit={(e) => { e.preventDefault(); previewMutation.mutate(); }}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-[13px] font-semibold text-[var(--color-text)]">MT5 Execution</h3>
          <StatusBadge label={terminalStatus.ready ? "Ready" : "Guarded"} tone={terminalStatus.ready ? "success" : "warning"} />
        </div>

        {!terminalStatus.ready ? (
          <div className="mt-3 rounded-[var(--radius-md)] border border-[var(--color-amber)]/20 bg-[var(--color-amber-soft)] px-3 py-2 text-[11px] text-[var(--color-text-soft)]">
            {terminalStatus.message}
          </div>
        ) : null}

        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <div>
            <FieldLabel htmlFor="exec-symbol">Symbol</FieldLabel>
            <FieldInput id="exec-symbol" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
          </div>
          <div>
            <FieldLabel htmlFor="exec-side">Side</FieldLabel>
            <FieldSelect id="exec-side" value={side} onChange={(e) => setSide(e.target.value)}>
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </FieldSelect>
          </div>
        </div>

        <div className="mt-3">
          <FieldLabel htmlFor="exec-notional">Exposure change</FieldLabel>
          <FieldInput id="exec-notional" type="number" min="1" step="1000" value={notional} onChange={(e) => setNotional(e.target.value)} />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {executionPresets.map((p) => (
              <PresetPill key={p} active={p === notional} onClick={() => setNotional(p)}>
                {formatCurrency(Number(p))}
              </PresetPill>
            ))}
          </div>
        </div>

        <div className="mt-3">
          <FieldLabel htmlFor="exec-note">Note</FieldLabel>
          <FieldTextarea id="exec-note" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Optional context" />
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button type="submit" disabled={previewMutation.isPending}
            className="h-8 rounded-[var(--radius-md)] bg-[var(--color-accent)] px-4 text-[12px] font-semibold text-[#1a1206] transition hover:brightness-110 disabled:opacity-50">
            {previewMutation.isPending ? "Previewing..." : "Preview"}
          </button>
          <button type="button" disabled={!preview?.guard.submit_allowed || submitMutation.isPending}
            onClick={() => submitMutation.mutate()}
            className="h-8 rounded-[var(--radius-md)] border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-4 text-[12px] font-semibold text-[var(--color-text)] transition hover:bg-[var(--color-surface-hover)] disabled:opacity-40">
            {submitMutation.isPending ? "Sending..." : "Send to MT5"}
          </button>
          {activeError ? (
            <span className="text-[11px] text-[var(--color-red)]">
              {activeError instanceof Error ? activeError.message : "Failed"}
            </span>
          ) : null}
        </div>
      </form>

      {/* Output */}
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
      <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
        <h3 className="text-[13px] font-semibold text-[var(--color-text)]">Execution Output</h3>
        <p className="mt-2 text-xs text-[var(--color-text-muted)]">
          Preview to see the risk verdict, lot sizing and margin check.
        </p>
      </div>
    );
  }

  const orderRequest = preview?.order_request ?? result?.order_request;
  const account = preview?.account ?? result?.account_before;
  const riskDecision = preview?.risk_decision ?? result?.risk_decision;
  const brokerPrice = safeNum(orderRequest, "price");
  const preTadeVar = riskDecision?.pre_trade?.var ?? null;
  const postTradeVar = riskDecision?.post_trade?.var ?? null;
  const varDelta = preTadeVar != null && postTradeVar != null ? postTradeVar - preTadeVar : null;
  const budgetUtil = riskDecision?.post_trade?.budget_utilization_var ?? null;

  return (
    <div className="rounded-[var(--radius-lg)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] p-4">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Guard Verdict</span>
          <div className="mt-1 text-2xl font-semibold text-[var(--color-text)]">{guard.decision}</div>
        </div>
        <StatusBadge label={result?.status ?? "Preview"} tone={result ? "accent" : "neutral"} />
      </div>

      {/* Key metrics */}
      <div className="mt-4 grid gap-2 sm:grid-cols-4">
        <FormMetaTile label="Model" value={guard.model_used.toUpperCase()} tone="accent" />
        <FormMetaTile label="Executable" value={formatCurrency(guard.executable_exposure_change)} tone="success" />
        <FormMetaTile label="Lots" value={guard.volume_lots.toFixed(2)} />
        <FormMetaTile label="Fill" value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)} />
      </div>

      {/* VaR Impact */}
      <div className="mt-3 grid gap-2 sm:grid-cols-4">
        <FormMetaTile label="Pre VaR" value={preTadeVar != null ? formatCurrency(preTadeVar) : "n/a"} />
        <FormMetaTile label="Post VaR" value={postTradeVar != null ? formatCurrency(postTradeVar) : "n/a"} />
        <FormMetaTile label="VaR delta" value={varDelta != null ? formatSignedCurrency(varDelta) : "n/a"} tone={varDelta == null ? "neutral" : varDelta > 0 ? "warning" : "success"} />
        <FormMetaTile label="Budget util" value={budgetUtil != null ? formatPercent(budgetUtil, 1) : "n/a"} tone={budgetUtil == null ? "neutral" : budgetUtil > 0.9 ? "danger" : budgetUtil > 0.7 ? "warning" : "success"} />
      </div>

      {/* Margin + Terminal */}
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Margin <StatusBadge label={guard.margin_ok ? "Pass" : "Fail"} tone={guard.margin_ok ? "success" : "danger"} />
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
            <div><span className="text-[var(--color-text-muted)]">Required</span><div className="mono font-semibold text-[var(--color-text)]">{formatCurrency(guard.margin_required, 2)}</div></div>
            <div><span className="text-[var(--color-text-muted)]">Free after</span><div className="mono font-semibold text-[var(--color-text)]">{formatCurrency(guard.free_margin_after, 2)}</div></div>
          </div>
        </div>
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Terminal</div>
          <div className="mt-2 flex items-center gap-2">
            <StatusBadge label={guard.submit_allowed ? "Submit OK" : "Blocked"} tone={guard.submit_allowed ? "success" : "warning"} />
            <span className="text-[11px] text-[var(--color-text-muted)]">{guard.execution_enabled ? "Kill switch on" : "Off"}</span>
          </div>
        </div>
      </div>

      {/* Reasons */}
      {(guard.reasons ?? []).length > 0 ? (
        <div className="mt-3 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Reasons</div>
          <div className="mt-1.5 space-y-1">
            {(guard.reasons ?? []).map((r) => (
              <div key={r} className="flex gap-2 text-[11px] text-[var(--color-text-soft)]">
                <span className="mt-1.5 size-1 shrink-0 rounded-full bg-[var(--color-accent)]" />
                {r}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* MT5 result */}
      {result ? (
        <div className="mt-3 rounded-[var(--radius-md)] border border-[var(--color-green)]/20 bg-[var(--color-green-soft)] p-3 text-[11px] text-[var(--color-text-soft)]">
          MT5 retcode {String(result.mt5_result?.retcode ?? "n/a")} — status <span className="font-semibold text-[var(--color-text)]">{result.status}</span>
        </div>
      ) : null}

      {/* Post-fill */}
      {result ? (
        <div className="mt-2 grid gap-2 sm:grid-cols-4">
          <FormMetaTile label="Broker" value={result.broker_status ?? "n/a"} tone={result.status === "EXECUTED" ? "success" : "warning"} />
          <FormMetaTile label="Filled" value={(result.filled_volume_lots ?? 0).toFixed(2)} hint={`Remaining ${(result.remaining_volume_lots ?? 0).toFixed(2)}`} />
          <FormMetaTile label="Fill ratio" value={result.fill_ratio == null ? "n/a" : formatPercent(result.fill_ratio, 0)} tone="success" />
          <FormMetaTile label="Slippage" value={result.slippage_points == null ? "n/a" : result.slippage_points.toFixed(1)} />
        </div>
      ) : null}
    </div>
  );
}

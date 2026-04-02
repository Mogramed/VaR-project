"use client";

import { useMutation } from "@tanstack/react-query";
import { TrendingUp, DollarSign } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { api } from "@/lib/api/client";
import type {
  ExecutionPreviewResponse,
  ExecutionResultResponse,
  MT5TerminalStatusResponse,
} from "@/lib/api/types";
import { preferredHeadlineRisk } from "@/lib/risk-surface";
import { formatCurrency, formatPercent, formatSignedCurrency } from "@/lib/utils";
import { StatusBadge } from "@/components/ui/primitives";
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

const executionPresets = ["250000", "500000", "1000000", "2500000"];

export function ExecutionPanel({
  portfolioSlug,
  terminalStatus,
  onSubmitted,
  initialSymbol,
  initialExposureChange,
  initialSide,
}: {
  portfolioSlug: string;
  terminalStatus: MT5TerminalStatusResponse;
  onSubmitted?: (result: ExecutionResultResponse) => void;
  initialSymbol?: string;
  initialExposureChange?: number | null;
  initialSide?: "buy" | "sell";
}) {
  const router = useRouter();
  const [symbol, setSymbol] = useState(initialSymbol ?? "EURUSD");
  const [side, setSide] = useState<"buy" | "sell">(
    initialSide ?? ((initialExposureChange ?? 0) < 0 ? "sell" : "buy"),
  );
  const [exposureChange, setExposureChange] = useState(
    String(Math.max(Math.abs(initialExposureChange ?? 500000), 1)),
  );
  const [note, setNote] = useState("");

  const previewMutation = useMutation({
    mutationFn: async () =>
      api.previewExecution({
        portfolio_slug: portfolioSlug,
        symbol,
        exposure_change: (side === "buy" ? 1 : -1) * Number(exposureChange),
        note,
      }),
  });

  const submitMutation = useMutation({
    mutationFn: async () =>
      api.submitExecution({
        portfolio_slug: portfolioSlug,
        symbol,
        exposure_change: (side === "buy" ? 1 : -1) * Number(exposureChange),
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

        <FormSection title="Order parameters">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <FieldLabel htmlFor="exec-symbol">Symbol</FieldLabel>
              <FieldInputWithIcon icon={TrendingUp} id="exec-symbol" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
            </div>
            <div>
              <FieldLabel htmlFor="exec-side">Side</FieldLabel>
              <FieldSelect id="exec-side" value={side} onChange={(e) => setSide(e.target.value as "buy" | "sell")}>
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </FieldSelect>
            </div>
          </div>
        </FormSection>

        <FormSection title="Sizing">
          <FieldLabel htmlFor="exec-exposure">Target exposure change</FieldLabel>
          <FieldInputWithIcon icon={DollarSign} id="exec-exposure" type="number" min="1" step="1000" value={exposureChange} onChange={(e) => setExposureChange(e.target.value)} />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {executionPresets.map((p) => (
              <PresetPill key={p} active={p === exposureChange} onClick={() => setExposureChange(p)}>
                {formatCurrency(Number(p))}
              </PresetPill>
            ))}
          </div>
          <p className="mt-2 text-[11px] text-[var(--color-text-muted)]">
            Enter the portfolio exposure change you want. The platform converts it into broker lots,
            runs guardrails, and shows the resulting risk and margin posture before submit.
          </p>
        </FormSection>

        <FormSection>
          <FieldLabel htmlFor="exec-note">Note</FieldLabel>
          <FieldTextarea id="exec-note" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Optional context" />
        </FormSection>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <SubmitButton isPending={previewMutation.isPending} label="Preview" pendingLabel="Previewing..." />
          <SubmitButton
            type="button"
            isPending={submitMutation.isPending}
            label="Send to MT5"
            pendingLabel="Sending..."
            variant="secondary"
            disabled={!preview?.guard.submit_allowed || submitMutation.isPending}
            onClick={() => submitMutation.mutate()}
          />
          <FormError message={activeError instanceof Error ? activeError.message : activeError ? "Failed" : null} />
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
  const riskDecision = preview?.risk_decision ?? result?.risk_decision;
  const preTradeVar = riskDecision?.pre_trade?.var ?? null;
  const postTradeVar = riskDecision?.post_trade?.var ?? null;
  const varDelta = preTradeVar != null && postTradeVar != null ? postTradeVar - preTradeVar : null;
  const budgetUtil = riskDecision?.post_trade?.budget_utilization_var ?? null;
  const currentExposure = riskDecision?.pre_trade?.symbol_exposure ?? null;
  const resultingExposure = riskDecision?.resulting_exposure ?? null;
  const preHeadline95 = preferredHeadlineRisk(riskDecision?.pre_trade?.headline_risk, ["live_1d_95"]);
  const postHeadline95 = preferredHeadlineRisk(riskDecision?.post_trade?.headline_risk, ["live_1d_95"]);
  const preStressed = preferredHeadlineRisk(riskDecision?.pre_trade?.headline_risk, ["stressed_10d_975", "stressed_10d_99", "governance_10d_975", "governance_10d_99"]);
  const postStressed = preferredHeadlineRisk(riskDecision?.post_trade?.headline_risk, ["stressed_10d_975", "stressed_10d_99", "governance_10d_975", "governance_10d_99"]);
  const stressedDelta =
    preStressed != null && postStressed != null ? postStressed.es - preStressed.es : null;
  const previewMicro = preview?.microstructure ?? null;
  const previewItems = Array.isArray(previewMicro?.items) ? previewMicro.items : [];
  const symbolMicro = (previewItems.find((item) => item?.symbol === preview?.symbol) ?? previewItems[0] ?? null) as
    | {
        spread_bps?: number;
        spread?: number;
        regime?: string;
      }
    | null;
  const previewRiskNowcast = (preview?.risk_nowcast ?? null) as
    | {
        pre_trade?: { live_1d_99?: { nowcast_var?: number; nowcast_es?: number } };
        post_trade?: { live_1d_99?: { nowcast_var?: number; nowcast_es?: number } };
      }
    | null;
  const preNowcast99 = (previewRiskNowcast?.pre_trade?.live_1d_99 ?? null) as
    | { nowcast_var?: number; nowcast_es?: number }
    | null;
  const postNowcast99 = (previewRiskNowcast?.post_trade?.live_1d_99 ?? null) as
    | { nowcast_var?: number; nowcast_es?: number }
    | null;
  const pnlExplain = (preview?.pnl_explain ?? null) as
    | { unrealized?: number; realized?: number }
    | null;

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
        <FormMetaTile label="Requested exposure" value={formatCurrency(guard.requested_exposure_change)} />
        <FormMetaTile label="Approved exposure" value={formatCurrency(guard.approved_exposure_change)} tone="success" />
        <FormMetaTile label="Broker lots" value={guard.volume_lots.toFixed(2)} tone="accent" />
        <FormMetaTile label="Fill" value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)} />
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-4">
        <FormMetaTile label="Current exposure" value={currentExposure != null ? formatCurrency(currentExposure) : "n/a"} />
        <FormMetaTile label="After trade" value={resultingExposure != null ? formatCurrency(resultingExposure) : "n/a"} />
        <FormMetaTile label="Model" value={guard.model_used.toUpperCase()} tone="accent" />
        <FormMetaTile label="Executable" value={formatCurrency(guard.executable_exposure_change)} tone="success" />
      </div>

      {/* VaR Impact */}
      <div className="mt-3 grid gap-2 sm:grid-cols-4">
        <FormMetaTile label="Pre VaR" value={preTradeVar != null ? formatCurrency(preTradeVar) : "n/a"} />
        <FormMetaTile label="Post VaR" value={postTradeVar != null ? formatCurrency(postTradeVar) : "n/a"} />
        <FormMetaTile label="VaR delta" value={varDelta != null ? formatSignedCurrency(varDelta) : "n/a"} tone={varDelta == null ? "neutral" : varDelta > 0 ? "warning" : "success"} />
        <FormMetaTile label="Budget util" value={budgetUtil != null ? formatPercent(budgetUtil, 1) : "n/a"} tone={budgetUtil == null ? "neutral" : budgetUtil > 0.9 ? "danger" : budgetUtil > 0.7 ? "warning" : "success"} />
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-4">
        <FormMetaTile
          label="Pre 1D 95%"
          value={preHeadline95 != null ? formatCurrency(preHeadline95.var) : "n/a"}
          hint={preHeadline95 != null ? `ES ${formatCurrency(preHeadline95.es)}` : undefined}
        />
        <FormMetaTile
          label="Post 1D 95%"
          value={postHeadline95 != null ? formatCurrency(postHeadline95.var) : "n/a"}
          hint={postHeadline95 != null ? `ES ${formatCurrency(postHeadline95.es)}` : undefined}
        />
        <FormMetaTile
          label="Pre stressed 10D 99%"
          value={preStressed != null ? formatCurrency(preStressed.es) : "n/a"}
          hint={preStressed?.scenario_name ?? undefined}
          tone="warning"
        />
        <FormMetaTile
          label="Stress delta"
          value={stressedDelta != null ? formatSignedCurrency(stressedDelta) : "n/a"}
          tone={stressedDelta == null ? "neutral" : stressedDelta > 0 ? "warning" : "success"}
        />
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-4">
        <FormMetaTile
          label="Nowcast pre 1D 99%"
          value={preNowcast99?.nowcast_var != null ? formatCurrency(preNowcast99.nowcast_var) : "n/a"}
          hint={preNowcast99?.nowcast_es != null ? `ES ${formatCurrency(preNowcast99.nowcast_es)}` : undefined}
          tone="warning"
        />
        <FormMetaTile
          label="Nowcast post 1D 99%"
          value={postNowcast99?.nowcast_var != null ? formatCurrency(postNowcast99.nowcast_var) : "n/a"}
          hint={postNowcast99?.nowcast_es != null ? `ES ${formatCurrency(postNowcast99.nowcast_es)}` : undefined}
          tone="warning"
        />
        <FormMetaTile
          label="Estimated spread cost"
          value={preview?.estimated_spread_cost != null ? formatCurrency(preview.estimated_spread_cost) : "n/a"}
          tone="accent"
        />
        <FormMetaTile
          label="Expected slippage"
          value={preview?.expected_slippage_points != null ? preview.expected_slippage_points.toFixed(1) : "n/a"}
          hint={symbolMicro?.regime ?? undefined}
          tone={symbolMicro?.regime === "stressed" ? "danger" : symbolMicro?.regime === "volatile" ? "warning" : "neutral"}
        />
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-4">
        <FormMetaTile
          label="Live spread"
          value={symbolMicro?.spread_bps != null ? `${symbolMicro.spread_bps.toFixed(1)} bps` : "n/a"}
          hint={symbolMicro?.spread != null ? symbolMicro.spread.toFixed(5) : undefined}
        />
        <FormMetaTile
          label="Regime"
          value={symbolMicro?.regime ?? "n/a"}
          tone={symbolMicro?.regime === "stressed" ? "danger" : symbolMicro?.regime === "volatile" ? "warning" : "success"}
        />
        <FormMetaTile
          label="Unrealized PnL"
          value={pnlExplain?.unrealized != null ? formatCurrency(pnlExplain.unrealized) : "n/a"}
          tone={pnlExplain?.unrealized != null && pnlExplain.unrealized >= 0 ? "success" : "warning"}
        />
        <FormMetaTile
          label="Realized PnL"
          value={pnlExplain?.realized != null ? formatCurrency(pnlExplain.realized) : "n/a"}
          tone={pnlExplain?.realized != null && pnlExplain.realized >= 0 ? "success" : "warning"}
        />
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
          MT5 retcode {String(result.mt5_result?.retcode ?? "n/a")} - status <span className="font-semibold text-[var(--color-text)]">{result.status}</span>
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

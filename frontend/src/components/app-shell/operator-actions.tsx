"use client";

import { Activity, Clock3, FileText, Radar, RefreshCw, RotateCcw, Square } from "lucide-react";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { ApiError, api } from "@/lib/api/client";
import { useOperatorRunAction } from "@/lib/use-operator-run";
import { cn, formatTimestamp } from "@/lib/utils";

function ActionButton({
  icon: Icon,
  label,
  pending,
  disabled,
  accent,
  state = "idle",
  onClick,
}: {
  icon: React.ElementType;
  label: string;
  pending?: boolean;
  disabled?: boolean;
  accent?: boolean;
  state?: "idle" | "queued" | "running" | "success" | "error";
  onClick: () => void;
}) {
  const statusClass =
    state === "error"
      ? "border-[var(--color-red)]/35 bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(118,36,36,0.18))] text-[var(--color-red)] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
      : state === "success"
        ? "border-[var(--color-green)]/35 bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(20,74,52,0.22))] text-[var(--color-green)] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
        : state === "queued" || state === "running"
          ? "border-[var(--color-amber)]/35 bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(105,72,16,0.2))] text-[var(--color-amber)] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
          : "";
  return (
    <button
      type="button"
      title={label}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex h-9 items-center gap-2 rounded-[14px] border px-3.5 text-[11px] font-semibold tracking-[0.02em] transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-40",
        accent
          ? "border-[var(--color-accent)]/30 bg-[var(--color-accent)] text-[#1a1206] shadow-[0_0_8px_rgba(240,185,11,0.15)] hover:brightness-110"
          : "border-[var(--color-border)] bg-[linear-gradient(180deg,rgba(26,30,39,0.94),rgba(16,19,26,0.98))] text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-soft)] hover:shadow-[0_10px_30px_rgba(0,0,0,0.2)]",
        statusClass,
      )}
    >
      <Icon className={cn("size-3.5", pending && "animate-spin")} />
      <span>{label}</span>
    </button>
  );
}

function actionLabel(action: string): string {
  if (action === "snapshot") return "snap";
  return action;
}

function summarizeActionError(error: Error | null): { message: string; hint: string | null; retryable: boolean } {
  if (!error) {
    return {
      message: "",
      hint: null,
      retryable: false,
    };
  }
  if (error instanceof ApiError) {
    const normalizedCode = String(error.errorCode ?? "").toLowerCase();
    const timeoutCode = normalizedCode.includes("timeout") || error.status === 504;
    const networkCode = normalizedCode.includes("unreachable") || normalizedCode.includes("request_failed");
    return {
      message: timeoutCode
        ? "Action timeout from frontend gateway."
        : networkCode
          ? "Unable to reach backend for this action."
          : error.message,
      hint: timeoutCode
        ? "The run may still be queued or running. Check recent runs below, then retry if needed."
        : networkCode
          ? "Check API/worker health and retry once transport is stable."
          : error.hint ?? null,
      retryable: timeoutCode || networkCode || [429, 502, 503, 504].includes(error.status),
    };
  }
  const message = error.message || "Operator action failed.";
  const lowered = message.toLowerCase();
  const looksTransient =
    lowered.includes("timeout")
    || lowered.includes("network")
    || lowered.includes("connect")
    || lowered.includes("socket")
    || lowered.includes("fetch failed");
  return {
    message,
    hint: looksTransient ? "Transient failure detected. Retry is available." : null,
    retryable: looksTransient,
  };
}

export function OperatorActions({
  portfolioSlug,
  accountId,
}: {
  portfolioSlug: string;
  accountId?: string;
}) {
  const { notifyOperatorRunCompleted } = useDeskLive();
  const sync = useOperatorRunAction({
    action: "sync",
    portfolioSlug,
    accountId,
    enqueue: async (payload: { portfolio_slug: string; account_id?: string }) => api.enqueueOperatorSync(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
    },
  });
  const snapshot = useOperatorRunAction({
    action: "snapshot",
    portfolioSlug,
    accountId,
    enqueue: async (payload: { portfolio_slug: string; account_id?: string }) => api.enqueueOperatorSnapshot(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
    },
  });
  const backtest = useOperatorRunAction({
    action: "backtest",
    portfolioSlug,
    accountId,
    enqueue: async (payload: { portfolio_slug: string; account_id?: string }) => api.enqueueOperatorBacktest(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
    },
  });
  const report = useOperatorRunAction({
    action: "report",
    portfolioSlug,
    accountId,
    enqueue: async (payload: { portfolio_slug: string; account_id?: string }) => api.enqueueOperatorReport(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
    },
  });

  const actions = [
    { key: "sync", label: "Sync", icon: RefreshCw, hook: sync, accent: false },
    { key: "snapshot", label: "Snap", icon: Activity, hook: snapshot, accent: false },
    { key: "backtest", label: "Backtest", icon: Radar, hook: backtest, accent: false },
    { key: "report", label: "Report", icon: FileText, hook: report, accent: true },
  ] as const;

  const activeContext = actions.find((item) => item.hook.canInterrupt) ?? null;
  const activeRun = activeContext?.hook.run ?? null;
  const busy = Boolean(activeContext) || actions.some((item) => item.hook.pending);
  const failingContext = actions.find((item) => item.hook.uiState === "error") ?? null;
  const errorDetails = summarizeActionError(failingContext?.hook.error ?? null);
  const stage = activeRun?.stage?.replaceAll("_", " ") ?? null;
  const elapsed = activeContext?.hook.elapsedSeconds ?? activeRun?.elapsed_seconds ?? null;
  const progressPercent = activeContext?.hook.progressPercent ?? null;
  const retryDisabled =
    !failingContext
    || busy
    || failingContext.hook.pending
    || failingContext.hook.canInterrupt;

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-2 rounded-[18px] border border-[var(--color-border)] bg-[linear-gradient(180deg,rgba(17,20,27,0.96),rgba(10,12,17,0.98))] p-1.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
        {actions.map((item) => (
          <ActionButton
            key={item.key}
            icon={item.icon}
            label={item.label}
            pending={item.hook.pending}
            state={item.hook.uiState}
            disabled={busy && !item.hook.canInterrupt}
            accent={item.accent}
            onClick={() =>
              item.hook.execute({
                portfolio_slug: portfolioSlug,
                account_id: accountId,
              })
            }
          />
        ))}
        {activeContext ? (
          <ActionButton
            icon={Square}
            label="Stop"
            pending={activeContext.hook.interrupting}
            disabled={activeContext.hook.interrupting}
            onClick={() => activeContext.hook.interrupt("Interrupted from operator panel.")}
          />
        ) : null}
        {failingContext ? (
          <ActionButton
            icon={RotateCcw}
            label="Retry"
            disabled={retryDisabled}
            onClick={() => failingContext.hook.retry()}
          />
        ) : null}
      </div>

      {activeContext && activeRun ? (
        <div className="rounded-[18px] border border-[var(--color-accent)]/20 bg-[linear-gradient(180deg,rgba(31,26,15,0.38),rgba(14,16,23,0.94))] px-3.5 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
          <div className="flex items-center justify-between gap-2 text-[10px] uppercase tracking-wider text-[var(--color-text-muted)]">
            <span className="flex items-center gap-1.5">
              <Clock3 className="size-3 text-[var(--color-accent)]" />
              <span className="font-semibold text-[var(--color-text-soft)]">{activeContext.label}</span>
              {activeContext.hook.statusLabel}
            </span>
            <span className="mono font-semibold tabular-nums text-[var(--color-accent)]">
              {elapsed != null ? `${Math.max(0, Math.round(elapsed))}s` : "n/a"}
            </span>
          </div>
          {stage ? (
            <div className="mt-1 text-[11px] text-[var(--color-text-soft)]">
              Stage: {stage}
            </div>
          ) : null}
          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[var(--color-border)]/40">
            <div
              className="h-full rounded-full bg-[var(--color-accent)] transition-[width] duration-300"
              style={{ width: `${Math.max(5, Math.min(progressPercent ?? 12, 100))}%` }}
            />
          </div>
          <div className="mt-1 flex items-center justify-between gap-2 text-[10px] text-[var(--color-text-muted)]">
            <span>Started: {formatTimestamp(activeRun.started_at ?? activeRun.created_at)}</span>
            <span>Updated: {formatTimestamp(activeContext.hook.lastUpdatedAt)}</span>
          </div>
        </div>
      ) : null}

      {failingContext && failingContext.hook.error ? (
        <div className="rounded-[18px] border border-[var(--color-red)]/25 bg-[linear-gradient(180deg,rgba(69,21,21,0.22),rgba(18,14,18,0.96))] px-3.5 py-3 text-[10px] shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
          <div className="font-semibold uppercase tracking-wider text-[var(--color-red)]">
            {actionLabel(failingContext.key)} failed
          </div>
          <div className="mt-0.5 text-[var(--color-text-soft)]">
            {errorDetails.message}
          </div>
          {errorDetails.hint ? (
            <div className="mt-0.5 text-[var(--color-text-muted)]">
              Next: {errorDetails.hint}
            </div>
          ) : null}
          {errorDetails.retryable ? (
            <div className="mt-0.5 text-[var(--color-text-muted)]">
              Retry is available.
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

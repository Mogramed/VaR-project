"use client";

import { Activity, FileText, Radar, RefreshCw } from "lucide-react";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { api } from "@/lib/api/client";
import { useOperatorRunAction } from "@/lib/use-operator-run";
import { cn } from "@/lib/utils";

function ActionButton({
  icon: Icon,
  label,
  pending,
  disabled,
  accent,
  onClick,
}: {
  icon: React.ElementType;
  label: string;
  pending?: boolean;
  disabled?: boolean;
  accent?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      title={label}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] border px-2 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        accent
          ? "border-[var(--color-accent)]/30 bg-[var(--color-accent)] text-[#1a1206] hover:brightness-110"
          : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-soft)]",
      )}
    >
      <Icon className={cn("size-3", pending && "animate-spin")} />
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

export function OperatorActions({ portfolioSlug }: { portfolioSlug: string }) {
  const { notifyOperatorRunCompleted } = useDeskLive();

  const sync = useOperatorRunAction({
    action: "sync",
    portfolioSlug,
    enqueue: async (payload: { portfolio_slug: string }) => api.enqueueOperatorSync(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
    },
  });
  const snapshot = useOperatorRunAction({
    action: "snapshot",
    portfolioSlug,
    enqueue: async (payload: { portfolio_slug: string }) => api.enqueueOperatorSnapshot(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
    },
  });
  const backtest = useOperatorRunAction({
    action: "backtest",
    portfolioSlug,
    enqueue: async (payload: { portfolio_slug: string }) => api.enqueueOperatorBacktest(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
    },
  });
  const report = useOperatorRunAction({
    action: "report",
    portfolioSlug,
    enqueue: async (payload: { portfolio_slug: string }) => api.enqueueOperatorReport(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
    },
  });

  const activeRun = sync.run?.status === "running" || sync.run?.status === "queued"
    ? sync.run
    : snapshot.run?.status === "running" || snapshot.run?.status === "queued"
      ? snapshot.run
      : backtest.run?.status === "running" || backtest.run?.status === "queued"
        ? backtest.run
        : report.run?.status === "running" || report.run?.status === "queued"
          ? report.run
          : null;
  const busy = sync.pending || snapshot.pending || backtest.pending || report.pending;
  const error = sync.error ?? snapshot.error ?? backtest.error ?? report.error ?? null;
  const stage = activeRun?.stage?.replaceAll("_", " ") ?? null;
  const elapsed = activeRun?.id === sync.run?.id
    ? sync.elapsedSeconds
    : activeRun?.id === snapshot.run?.id
      ? snapshot.elapsedSeconds
      : activeRun?.id === backtest.run?.id
        ? backtest.elapsedSeconds
        : activeRun?.id === report.run?.id
          ? report.elapsedSeconds
          : activeRun?.elapsed_seconds;

  return (
    <div className="flex items-center gap-1">
      <ActionButton icon={RefreshCw} label="Sync" pending={sync.pending} disabled={busy} onClick={() => sync.execute({ portfolio_slug: portfolioSlug })} />
      <ActionButton icon={Activity} label="Snap" pending={snapshot.pending} disabled={busy} onClick={() => snapshot.execute({ portfolio_slug: portfolioSlug })} />
      <ActionButton icon={Radar} label="Backtest" pending={backtest.pending} disabled={busy} onClick={() => backtest.execute({ portfolio_slug: portfolioSlug })} />
      <ActionButton icon={FileText} label="Report" pending={report.pending} disabled={busy} accent onClick={() => report.execute({ portfolio_slug: portfolioSlug })} />
      {stage ? (
        <span className="hidden text-[10px] uppercase tracking-[0.18em] text-[var(--color-text-muted)] lg:inline">
          {stage}
          {elapsed != null ? ` - ${Math.max(0, Math.round(elapsed))}s` : ""}
        </span>
      ) : null}
      {error ? (
        <span className="text-[10px] text-[var(--color-red)]">
          {error instanceof Error ? error.message : "Failed"}
        </span>
      ) : null}
    </div>
  );
}

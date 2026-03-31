"use client";

import { useMutation } from "@tanstack/react-query";
import { Activity, FileText, Radar, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api/client";
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
  const router = useRouter();

  const sync = useMutation({
    mutationFn: async () => api.syncMarketData({ portfolio_slug: portfolioSlug }),
    onSuccess: () => router.refresh(),
  });
  const snapshot = useMutation({
    mutationFn: async () => api.runSnapshot({ portfolio_slug: portfolioSlug }),
    onSuccess: () => router.refresh(),
  });
  const backtest = useMutation({
    mutationFn: async () => api.runBacktest({ portfolio_slug: portfolioSlug }),
    onSuccess: () => router.refresh(),
  });
  const report = useMutation({
    mutationFn: async () => api.runReport(undefined, portfolioSlug),
    onSuccess: () => router.refresh(),
  });

  const busy = sync.isPending || snapshot.isPending || backtest.isPending || report.isPending;
  const error = sync.error ?? snapshot.error ?? backtest.error ?? report.error ?? null;

  return (
    <div className="flex items-center gap-1">
      <ActionButton icon={RefreshCw} label="Sync" pending={sync.isPending} disabled={busy} onClick={() => sync.mutate()} />
      <ActionButton icon={Activity} label="Snap" pending={snapshot.isPending} disabled={busy} onClick={() => snapshot.mutate()} />
      <ActionButton icon={Radar} label="Backtest" pending={backtest.isPending} disabled={busy} onClick={() => backtest.mutate()} />
      <ActionButton icon={FileText} label="Report" pending={report.isPending} disabled={busy} accent onClick={() => report.mutate()} />
      {error ? (
        <span className="text-[10px] text-[var(--color-red)]">
          {error instanceof Error ? error.message : "Failed"}
        </span>
      ) : null}
    </div>
  );
}

"use client";

import { useMutation } from "@tanstack/react-query";
import { Activity, FileText, Radar, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api/client";


export function OperatorActions({ portfolioSlug }: { portfolioSlug: string }) {
  const router = useRouter();

  const snapshot = useMutation({
    mutationFn: async () => api.runSnapshot({ portfolio_slug: portfolioSlug }),
    onSuccess: () => router.refresh(),
  });
  const sync = useMutation({
    mutationFn: async () => api.syncMarketData({ portfolio_slug: portfolioSlug }),
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
  const error =
    sync.error ?? snapshot.error ?? backtest.error ?? report.error ?? null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => sync.mutate()}
        disabled={busy}
        className="inline-flex h-10 items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 text-xs uppercase tracking-[0.22em] text-[var(--color-text-soft)] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:border-[var(--color-border-strong)] hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
      >
        <RefreshCw className={`size-4 ${sync.isPending ? "animate-spin" : ""}`} />
        Sync MT5
      </button>
      <button
        type="button"
        onClick={() => snapshot.mutate()}
        disabled={busy}
        className="inline-flex h-10 items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 text-xs uppercase tracking-[0.22em] text-[var(--color-text-soft)] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:border-[var(--color-border-strong)] hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
      >
        <Activity className={`size-4 ${snapshot.isPending ? "animate-spin" : ""}`} />
        Snapshot
      </button>
      <button
        type="button"
        onClick={() => backtest.mutate()}
        disabled={busy}
        className="inline-flex h-10 items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 text-xs uppercase tracking-[0.22em] text-[var(--color-text-soft)] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:border-[var(--color-border-strong)] hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
      >
        <Radar className={`size-4 ${backtest.isPending ? "animate-spin" : ""}`} />
        Backtest
      </button>
      <button
        type="button"
        onClick={() => report.mutate()}
        disabled={busy}
        className="inline-flex h-10 items-center justify-center gap-2 rounded-full bg-[var(--color-accent)] px-4 text-xs font-semibold uppercase tracking-[0.22em] text-[#1a1206] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:shadow-[0_16px_34px_rgba(216,155,73,0.24)] disabled:cursor-not-allowed disabled:opacity-60"
      >
        <FileText className={`size-4 ${report.isPending ? "animate-spin" : ""}`} />
        Report
      </button>
      {error ? (
        <div className="w-full text-xs text-[var(--color-red)]">
          {error instanceof Error ? error.message : "Operator action failed."}
        </div>
      ) : null}
    </div>
  );
}

"use client";

import { FileDown, RefreshCw } from "lucide-react";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import type { ReportRunResponse } from "@/lib/api/types";
import { api } from "@/lib/api/client";
import { useOperatorRunAction } from "@/lib/use-operator-run";

export function ReportActions({
  portfolioSlug,
  onGenerated,
}: {
  portfolioSlug?: string;
  onGenerated?: (result: ReportRunResponse) => void | Promise<void>;
}) {
  const { notifyOperatorRunCompleted, accountId } = useDeskLive();
  const operatorRun = useOperatorRunAction({
    action: "report",
    portfolioSlug,
    accountId,
    enqueue: async (payload: { portfolio_slug?: string; account_id?: string }) => api.enqueueOperatorReport(payload),
    onSucceeded: async (run) => {
      notifyOperatorRunCompleted(run);
      const report = (run.result?.report ?? {}) as ReportRunResponse;
      await onGenerated?.(report);
    },
  });

  const pdfUrl = portfolioSlug
    ? accountId
      ? `/api/reports/pdf?portfolio=${encodeURIComponent(portfolioSlug)}&account=${encodeURIComponent(accountId)}`
      : `/api/reports/pdf?portfolio=${encodeURIComponent(portfolioSlug)}`
    : "/api/reports/pdf";

  return (
    <div className="flex items-center gap-2 print:hidden">
      <button
        type="button"
        onClick={() => operatorRun.execute({ portfolio_slug: portfolioSlug, account_id: accountId })}
        disabled={operatorRun.pending}
        className="flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--color-border)] px-2.5 text-[11px] font-medium text-[var(--color-text-soft)] transition-colors hover:border-[var(--color-border-strong)] hover:text-[var(--color-text)] disabled:opacity-50"
      >
        <RefreshCw className={`size-3 ${operatorRun.pending ? "animate-spin" : ""}`} />
        Generate
      </button>
      <a
        href={pdfUrl}
        className="flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--color-accent)] px-2.5 text-[11px] font-semibold text-[#1a1206] transition hover:brightness-110"
      >
        <FileDown className="size-3" />
        PDF
      </a>
      {operatorRun.error ? (
        <span className="text-[10px] text-[var(--color-red)]">
          {operatorRun.error instanceof Error ? operatorRun.error.message : "Failed"}
        </span>
      ) : null}
    </div>
  );
}

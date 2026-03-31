"use client";

import { useMutation } from "@tanstack/react-query";
import { FileDown, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api/client";
import type { ReportRunResponse } from "@/lib/api/types";

export function ReportActions({
  portfolioSlug,
  onGenerated,
}: {
  portfolioSlug?: string;
  onGenerated?: (result: ReportRunResponse) => void | Promise<void>;
}) {
  const router = useRouter();
  const mutation = useMutation({
    mutationFn: async () => api.runReport(undefined, portfolioSlug),
    onSuccess: (result) => {
      onGenerated?.(result);
      if (!onGenerated) router.refresh();
    },
  });

  const pdfUrl = portfolioSlug
    ? `/api/reports/pdf?portfolio=${encodeURIComponent(portfolioSlug)}`
    : "/api/reports/pdf";

  return (
    <div className="flex items-center gap-2 print:hidden">
      <button
        type="button"
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending}
        className="flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--color-border)] px-2.5 text-[11px] font-medium text-[var(--color-text-soft)] transition-colors hover:border-[var(--color-border-strong)] hover:text-[var(--color-text)] disabled:opacity-50"
      >
        <RefreshCw className={`size-3 ${mutation.isPending ? "animate-spin" : ""}`} />
        Generate
      </button>
      <a
        href={pdfUrl}
        className="flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] bg-[var(--color-accent)] px-2.5 text-[11px] font-semibold text-[#1a1206] transition hover:brightness-110"
      >
        <FileDown className="size-3" />
        PDF
      </a>
      {mutation.error ? (
        <span className="text-[10px] text-[var(--color-red)]">
          {mutation.error instanceof Error ? mutation.error.message : "Failed"}
        </span>
      ) : null}
    </div>
  );
}

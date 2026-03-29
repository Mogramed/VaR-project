"use client";

import { useMutation } from "@tanstack/react-query";
import { FileDown, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api/client";

export function ReportActions({ portfolioSlug }: { portfolioSlug?: string }) {
  const router = useRouter();
  const mutation = useMutation({
    mutationFn: async () => api.runReport(undefined, portfolioSlug),
    onSuccess: () => router.refresh(),
  });
  const pdfUrl = portfolioSlug
    ? `/api/reports/pdf?portfolio=${encodeURIComponent(portfolioSlug)}`
    : "/api/reports/pdf";

  return (
    <div className="flex flex-wrap items-center gap-3 print:hidden">
      <button
        type="button"
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending}
        className="inline-flex h-11 items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 text-sm font-semibold text-white transition hover:border-[var(--color-border-strong)] hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:opacity-60"
      >
        <RefreshCw className={`size-4 ${mutation.isPending ? "animate-spin" : ""}`} />
        {mutation.isPending ? "Generating..." : "Generate fresh report"}
      </button>
      <a
        href={pdfUrl}
        className="inline-flex h-11 items-center justify-center gap-2 rounded-full bg-[var(--color-accent)] px-4 text-sm font-semibold text-[#1a1206] transition hover:translate-y-[-1px] hover:shadow-[0_16px_34px_rgba(216,155,73,0.24)]"
      >
        <FileDown className="size-4" />
        Download PDF
      </a>
      {mutation.error ? (
        <div className="w-full text-sm text-[var(--color-red)]">
          {mutation.error instanceof Error ? mutation.error.message : "Could not generate report."}
        </div>
      ) : null}
    </div>
  );
}

"use client";

export default function ReportExportError({
  error,
}: {
  error: Error & { digest?: string };
}) {
  return (
    <main
      data-report-export-root="true"
      className="flex min-h-screen items-center justify-center bg-[#090a0d] px-8 py-8 text-[var(--color-text)]"
    >
      <div className="mx-auto max-w-md text-center">
        <div className="text-sm font-semibold uppercase tracking-wider text-[var(--color-red)]">
          Report generation failed
        </div>
        <p className="mt-3 text-sm leading-relaxed text-[var(--color-text-muted)]">
          {error.message || "Unable to load report data. The backend API may be unreachable."}
        </p>
      </div>
    </main>
  );
}

"use client";

export default function DeskError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="rounded-full border border-[var(--color-red)]/20 bg-[var(--color-red-soft)] px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-red)]">
        Error
      </div>
      <h2 className="text-xl font-semibold text-[var(--color-text)]">
        Something went wrong
      </h2>
      <p className="max-w-md text-sm leading-relaxed text-[var(--color-text-muted)]">
        {error.message || "The page failed to load. The backend API may be unreachable."}
      </p>
      <button
        onClick={reset}
        className="mt-2 h-8 rounded-[var(--radius-md)] bg-[var(--color-accent)] px-4 text-[12px] font-semibold text-[#1a1206] transition hover:brightness-110"
      >
        Try again
      </button>
    </div>
  );
}

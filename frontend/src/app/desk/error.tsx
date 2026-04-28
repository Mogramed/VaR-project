"use client";

import { useEffect, useMemo } from "react";

const CHUNK_RELOAD_SESSION_KEY = "var:desk:chunk-reload-at";
const CHUNK_RELOAD_THROTTLE_MS = 45_000;

function isChunkLoadErrorMessage(message: string) {
  const normalized = message.toLowerCase();
  return normalized.includes("chunkloaderror")
    || normalized.includes("failed to load chunk")
    || normalized.includes("loading chunk")
    || normalized.includes("dynamically imported module")
    || normalized.includes("/_next/static/chunks");
}

export default function DeskError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const rawMessage = String(error.message ?? "");
  const chunkLoadError = useMemo(
    () => isChunkLoadErrorMessage(rawMessage),
    [rawMessage],
  );

  useEffect(() => {
    if (!chunkLoadError || typeof window === "undefined") {
      return;
    }
    const lastReloadAtRaw = window.sessionStorage.getItem(CHUNK_RELOAD_SESSION_KEY);
    const lastReloadAt = lastReloadAtRaw == null ? 0 : Number.parseInt(lastReloadAtRaw, 10);
    const now = Date.now();
    if (Number.isFinite(lastReloadAt) && now - lastReloadAt < CHUNK_RELOAD_THROTTLE_MS) {
      return;
    }
    window.sessionStorage.setItem(CHUNK_RELOAD_SESSION_KEY, String(now));
    window.location.reload();
  }, [chunkLoadError]);

  const handleRetry = () => {
    if (chunkLoadError && typeof window !== "undefined") {
      window.sessionStorage.setItem(CHUNK_RELOAD_SESSION_KEY, String(Date.now()));
      window.location.reload();
      return;
    }
    reset();
  };

  const renderedMessage = chunkLoadError
    ? "Frontend assets were refreshed. Reloading the page fetches the latest report bundle."
    : rawMessage || "The page failed to load. The backend API may be unreachable.";

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="rounded-full border border-[var(--color-red)]/20 bg-[var(--color-red-soft)] px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-red)]">
        Error
      </div>
      <h2 className="text-xl font-semibold text-[var(--color-text)]">
        Something went wrong
      </h2>
      <p className="max-w-md text-sm leading-relaxed text-[var(--color-text-muted)]">
        {renderedMessage}
      </p>
      <button
        onClick={handleRetry}
        className="mt-2 h-8 rounded-[var(--radius-md)] bg-[var(--color-accent)] px-4 text-[12px] font-semibold text-[#1a1206] transition hover:brightness-110"
      >
        {chunkLoadError ? "Reload app" : "Try again"}
      </button>
    </div>
  );
}

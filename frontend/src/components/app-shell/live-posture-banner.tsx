"use client";

import {
  deriveLiveRuntimeDiagnostics,
  type DeskLiveTransport,
} from "@/components/app-shell/live-runtime-phase";
import { StatusBadge } from "@/components/ui/primitives";
import type { MT5LiveStateResponse } from "@/lib/api/types";
import { formatTimestamp } from "@/lib/utils";

function formatRetryDelay(seconds: number | null) {
  if (seconds == null) {
    return null;
  }
  if (seconds >= 10) {
    return `${Math.round(seconds)}s`;
  }
  return `${seconds.toFixed(1)}s`;
}

function bannerToneClass(phase: "pending" | "live" | "recovering" | "degraded" | "offline", marketClosed: boolean) {
  if (marketClosed) {
    return "border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-soft)]";
  }
  if (phase === "recovering") {
    return "border-[var(--color-accent)]/30 bg-[var(--color-accent-soft)]/25 text-[var(--color-text-soft)]";
  }
  if (phase === "degraded") {
    return "border-[var(--color-amber)]/30 bg-[var(--color-amber-soft)] text-[var(--color-text-soft)]";
  }
  if (phase === "offline") {
    return "border-[var(--color-red)]/30 bg-[var(--color-red-soft)] text-[var(--color-text-soft)]";
  }
  return "border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-soft)]";
}

function bannerMessage(
  phase: "pending" | "live" | "recovering" | "degraded" | "offline",
  marketClosed: boolean,
  retryDelay: string | null,
  retrying: boolean,
) {
  if (marketClosed) {
    return "Market closed. Le dernier etat broker valide est conserve.";
  }
  if (phase === "pending") {
    return "Connexion au flux live MT5 en cours.";
  }
  if (phase === "recovering") {
    if (retrying && retryDelay) {
      return `Flux en reprise. Nouvelle tentative automatique dans ${retryDelay}.`;
    }
    return "Flux en reprise. Synchronisation automatique en cours.";
  }
  if (phase === "degraded") {
    if (retrying && retryDelay) {
      return `Flux temporairement retarde. Retry automatique dans ${retryDelay} avec conservation du dernier snapshot.`;
    }
    return "Flux temporairement retarde. Dernier snapshot operateur conserve pour la continuite.";
  }
  if (phase === "offline") {
    return "Connexion MT5 indisponible. Le poste reste utilisable sur l etat persiste.";
  }
  return "Flux live nominal.";
}

export function LivePostureBanner({
  liveState,
  transport,
  className = "",
}: {
  liveState: MT5LiveStateResponse | null;
  transport: DeskLiveTransport;
  className?: string;
}) {
  const diagnostics = deriveLiveRuntimeDiagnostics(liveState, transport);
  const phase = diagnostics.phase;
  const marketClosed = diagnostics.marketClosed;
  const retryDelay = formatRetryDelay(diagnostics.retryInSeconds);

  if (phase === "live" && !marketClosed) {
    return null;
  }

  const generatedAt = liveState?.generated_at ? formatTimestamp(liveState.generated_at) : "n/a";
  const sequence = liveState?.sequence ?? null;
  const badgeTone = phase === "offline"
    ? "danger"
    : phase === "degraded"
      ? "warning"
      : phase === "recovering"
        ? "accent"
        : "neutral";
  const badgeLabel = phase === "pending" ? "pending" : phase;
  const hasError = typeof liveState?.last_error === "string" && liveState.last_error.trim().length > 0;
  const errorLabel = hasError ? liveState?.last_error?.trim().slice(0, 220) : null;

  return (
    <div className={`rounded-[var(--radius-md)] border px-3 py-2 text-[11px] ${bannerToneClass(phase, marketClosed)} ${className}`.trim()}>
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge label={badgeLabel} tone={badgeTone} />
        <span>{bannerMessage(phase, marketClosed, retryDelay, diagnostics.isRetrying)}</span>
        {diagnostics.failureCount > 0 ? (
          <StatusBadge label={`fails ${diagnostics.failureCount}`} tone="warning" />
        ) : null}
        {diagnostics.isRetrying && retryDelay ? (
          <StatusBadge label={`retry ${retryDelay}`} tone="accent" />
        ) : null}
        <span className="mono text-[10px] text-[var(--color-text-muted)]">
          seq {sequence ?? "n/a"} | {generatedAt}
        </span>
      </div>
      {errorLabel ? (
        <p className="mt-2 text-[11px] text-[var(--color-text-muted)]">
          Last error: <span className="mono">{errorLabel}</span>
        </p>
      ) : null}
    </div>
  );
}

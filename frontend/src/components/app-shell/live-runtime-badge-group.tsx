"use client";

import { StatusBadge } from "@/components/ui/primitives";
import {
  deriveLiveRuntimeDiagnostics,
  type DeskLiveTransport,
} from "@/components/app-shell/live-runtime-phase";
import type { MT5LiveStateResponse } from "@/lib/api/types";
import { formatTimestamp } from "@/lib/utils";
import { useRelativeTime } from "@/lib/use-relative-time";

function formatRetryDelay(seconds: number | null) {
  if (seconds == null) {
    return null;
  }
  if (seconds >= 10) {
    return `${Math.round(seconds)}s`;
  }
  return `${seconds.toFixed(1)}s`;
}

function bridgeTone(liveState: MT5LiveStateResponse | null, transport: DeskLiveTransport) {
  const diagnostics = deriveLiveRuntimeDiagnostics(liveState, transport);
  const phase = diagnostics.phase;
  if (phase === "live") {
    return "success" as const;
  }
  if (phase === "recovering") {
    return "accent" as const;
  }
  if (phase === "degraded") {
    return "warning" as const;
  }
  if (phase === "offline") {
    return "danger" as const;
  }
  return "neutral" as const;
}

function bridgeLabel(liveState: MT5LiveStateResponse | null, transport: DeskLiveTransport) {
  const diagnostics = deriveLiveRuntimeDiagnostics(liveState, transport);
  const phase = diagnostics.phase;
  if (phase === "pending") {
    return "bridge pending";
  }
  if (phase === "live") {
    return "mt5 live";
  }
  if (phase === "recovering") {
    if (diagnostics.isRetrying) {
      return "mt5 retrying";
    }
    return "mt5 recovering";
  }
  if (phase === "degraded") {
    return "mt5 degraded";
  }
  return "mt5 offline";
}

function transportTone(transport: DeskLiveTransport) {
  if (transport === "stream") {
    return "success" as const;
  }
  if (transport === "polling") {
    return "warning" as const;
  }
  return "neutral" as const;
}

function transportLabel(transport: DeskLiveTransport) {
  if (transport === "stream") {
    return "sse";
  }
  if (transport === "polling") {
    return "polling";
  }
  return "connecting";
}

export function LiveRuntimeBadgeGroup({
  liveState,
  heartbeatAt = null,
  transport,
  showBridge = true,
  showFreshness = false,
}: {
  liveState: MT5LiveStateResponse | null;
  heartbeatAt?: string | null;
  transport: DeskLiveTransport;
  showBridge?: boolean;
  showFreshness?: boolean;
}) {
  const freshnessTimestamp = heartbeatAt ?? liveState?.generated_at ?? null;
  const relativeUpdatedAt = useRelativeTime(freshnessTimestamp);
  const diagnostics = deriveLiveRuntimeDiagnostics(liveState, transport);
  const phase = diagnostics.phase;
  const retryDelayLabel = formatRetryDelay(diagnostics.retryInSeconds);
  const delayed = phase === "degraded" || diagnostics.isRetrying;

  return (
    <>
      {showBridge ? (
        <StatusBadge label={bridgeLabel(liveState, transport)} tone={bridgeTone(liveState, transport)} />
      ) : null}
      <StatusBadge label={transportLabel(transport)} tone={transportTone(transport)} />
      {diagnostics.isRetrying && retryDelayLabel ? (
        <StatusBadge label={`retry ${retryDelayLabel}`} tone="accent" />
      ) : null}
      {diagnostics.failureCount > 0 ? (
        <StatusBadge label={`fails ${diagnostics.failureCount}`} tone="warning" />
      ) : null}
      {delayed ? <StatusBadge label="delayed" tone="warning" /> : null}
      {diagnostics.marketClosed ? <StatusBadge label="market closed" tone="neutral" /> : null}
      {showFreshness && freshnessTimestamp ? (
        <span
          className="mono text-[10px] tabular-nums text-[var(--color-text-muted)]"
          title={formatTimestamp(freshnessTimestamp)}
        >
          {relativeUpdatedAt}
        </span>
      ) : null}
    </>
  );
}

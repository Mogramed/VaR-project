import type { MT5LiveStateResponse } from "@/lib/api/types";

export type DeskLiveTransport = "stream" | "polling" | "connecting";

export type LiveRuntimePhase = "pending" | "live" | "recovering" | "degraded" | "offline";

export interface LiveRuntimeDiagnostics {
  phase: LiveRuntimePhase;
  marketClosed: boolean;
  isRetrying: boolean;
  retryInSeconds: number | null;
  failureCount: number;
}

function asFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function asPositiveNumber(value: unknown): number | null {
  const parsed = asFiniteNumber(value);
  if (parsed == null || parsed <= 0) {
    return null;
  }
  return parsed;
}

function asNonNegativeInt(value: unknown): number {
  const parsed = asFiniteNumber(value);
  if (parsed == null) {
    return 0;
  }
  return Math.max(Math.floor(parsed), 0);
}

export function deriveLiveRuntimeDiagnostics(
  liveState: MT5LiveStateResponse | null,
  transport: DeskLiveTransport,
): LiveRuntimeDiagnostics {
  if (liveState == null) {
    return {
      phase: "pending",
      marketClosed: false,
      isRetrying: transport === "connecting",
      retryInSeconds: null,
      failureCount: 0,
    };
  }

  const rawLiveState = liveState as Record<string, unknown>;
  const health = (rawLiveState.health as Record<string, unknown> | null | undefined) ?? null;
  const healthStatus = String(health?.status ?? "").trim().toLowerCase();
  const connected = Boolean(liveState.connected);
  const stale = Boolean(liveState.stale);
  const degraded = Boolean(liveState.degraded);
  const marketClosed = Boolean(liveState.market_closed);
  const fallbackSnapshotUsed = Boolean(rawLiveState.fallback_snapshot_used);
  const status = String(liveState.status ?? "").trim().toLowerCase();
  const hasError = typeof liveState.last_error === "string" && liveState.last_error.trim().length > 0;
  const errorRetryable =
    health && typeof health.error_retryable === "boolean" ? Boolean(health.error_retryable) : null;
  const failureCount = Math.max(
    asNonNegativeInt(health?.bridge_consecutive_failures),
    asNonNegativeInt(rawLiveState.bridge_consecutive_failures),
  );
  const retryInSeconds = asPositiveNumber(
    health?.bridge_next_poll_delay_seconds
      ?? rawLiveState.bridge_next_poll_delay_seconds
      ?? null,
  );
  const hasRetryPressure = !marketClosed && (errorRetryable === true || failureCount > 0);

  if (marketClosed && connected) {
    return {
      phase: "live",
      marketClosed,
      isRetrying: false,
      retryInSeconds: null,
      failureCount,
    };
  }

  if (!connected) {
    const retrying = hasRetryPressure || transport === "connecting";
    return {
      phase: retrying ? "recovering" : "offline",
      marketClosed,
      isRetrying: retrying,
      retryInSeconds: retrying ? retryInSeconds : null,
      failureCount,
    };
  }

  if (fallbackSnapshotUsed || degraded || stale || healthStatus === "degraded" || healthStatus === "stale") {
    return {
      phase: "degraded",
      marketClosed,
      isRetrying: hasRetryPressure,
      retryInSeconds: hasRetryPressure ? retryInSeconds : null,
      failureCount,
    };
  }

  if (status === "ok" || status === "market_closed" || healthStatus === "healthy") {
    return {
      phase: "live",
      marketClosed,
      isRetrying: false,
      retryInSeconds: null,
      failureCount,
    };
  }

  const recovering = hasRetryPressure || transport === "connecting" || hasError;
  return {
    phase: recovering ? "recovering" : "live",
    marketClosed,
    isRetrying: recovering && hasRetryPressure,
    retryInSeconds: recovering && hasRetryPressure ? retryInSeconds : null,
    failureCount,
  };
}

export function deriveLiveRuntimePhase(
  liveState: MT5LiveStateResponse | null,
  transport: DeskLiveTransport,
): LiveRuntimePhase {
  return deriveLiveRuntimeDiagnostics(liveState, transport).phase;
}

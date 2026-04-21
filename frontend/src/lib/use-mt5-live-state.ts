"use client";

import { useEffect, useRef, useState } from "react";

import type { MT5LiveEventResponse, MT5LiveStateResponse } from "@/lib/api/types";

type LiveDetailLevel = "summary" | "full" | "inspector";

const LIVE_STATE_FETCH_TIMEOUT_MS = 8_000;
const LIVE_STATE_POLL_INTERVAL_MS = 1_000;
const LIVE_STATE_POLL_BACKOFF_MAX_MS = (() => {
  const parsed = Number(process.env.NEXT_PUBLIC_MT5_POLL_BACKOFF_MAX_MS ?? "4000");
  if (!Number.isFinite(parsed)) {
    return 4_000;
  }
  return Math.max(parsed, 1_000);
})();
const LIVE_STATE_HIDDEN_TAB_MULTIPLIER = 1;
const LIVE_STATE_POLL_JITTER_RATIO = 0.08;
const LIVE_STATE_POLL_FAILURE_STEP = 0.5;
const LIVE_STATE_MAX_HEALTHY_POLL_INTERVAL_MS = (() => {
  const parsed = Number(process.env.NEXT_PUBLIC_MT5_MAX_POLL_INTERVAL_MS ?? "1000");
  if (!Number.isFinite(parsed)) {
    return 1_000;
  }
  return Math.max(parsed, 500);
})();
const LIVE_STATE_FORCED_RESYNC_STALE_MS = (() => {
  const parsed = Number(process.env.NEXT_PUBLIC_MT5_FORCE_RESYNC_STALE_MS ?? "5000");
  if (!Number.isFinite(parsed)) {
    return 5_000;
  }
  return Math.max(parsed, 2_000);
})();
const LIVE_STATE_STREAM_WATCHDOG_INTERVAL_MS = 3_000;
const LIVE_STATE_STREAM_STALL_TIMEOUT_MS = (() => {
  const parsed = Number(process.env.NEXT_PUBLIC_MT5_STREAM_STALL_TIMEOUT_MS ?? "25000");
  if (!Number.isFinite(parsed)) {
    return 25_000;
  }
  return Math.max(parsed, 5_000);
})();
const LIVE_STATE_STREAM_RECONNECT_INTERVAL_MS = (() => {
  const parsed = Number(process.env.NEXT_PUBLIC_MT5_STREAM_RECONNECT_MS ?? "30000");
  if (!Number.isFinite(parsed)) {
    return 30_000;
  }
  return Math.max(parsed, 5_000);
})();
const MT5_STREAM_MODE = (process.env.NEXT_PUBLIC_MT5_STREAM_MODE ?? "polling").toLowerCase();
const MT5_STREAM_ENABLED = MT5_STREAM_MODE === "stream" || MT5_STREAM_MODE === "auto";
const MT5_POLL_WHILE_STREAMING = MT5_STREAM_MODE === "auto";

function withTimeout(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  if (!signal) {
    return timeoutSignal;
  }
  if (typeof AbortSignal.any === "function") {
    return AbortSignal.any([signal, timeoutSignal]);
  }
  return signal;
}

function buildStateUrl(
  portfolioSlug?: string,
  detailLevel: LiveDetailLevel = "full",
  accountId?: string,
) {
  const params = new URLSearchParams();
  if (portfolioSlug) {
    params.set("portfolio_slug", portfolioSlug);
  }
  if (accountId) {
    params.set("account_id", accountId);
  }
  params.set("detail_level", detailLevel);
  const query = params.toString();
  return query ? `/api/proxy/mt5/live/state?${query}` : "/api/proxy/mt5/live/state";
}

function buildStreamUrl(
  portfolioSlug?: string,
  detailLevel: LiveDetailLevel = "full",
  accountId?: string,
) {
  const params = new URLSearchParams();
  if (portfolioSlug) {
    params.set("portfolio_slug", portfolioSlug);
  }
  if (accountId) {
    params.set("account_id", accountId);
  }
  params.set("detail_level", detailLevel);
  const query = params.toString();
  return query ? `/api/proxy/mt5/live/stream?${query}` : "/api/proxy/mt5/live/stream";
}

export function useMt5LiveState(
  portfolioSlug?: string,
  options?: {
    initialState?: MT5LiveStateResponse | null;
    detailLevel?: LiveDetailLevel;
    accountId?: string;
  },
) {
  const detailLevel = options?.detailLevel ?? "full";
  const accountId = options?.accountId;
  const [liveState, setLiveState] = useState<MT5LiveStateResponse | null>(
    options?.initialState ?? null,
  );
  const [heartbeatAt, setHeartbeatAt] = useState<string | null>(
    options?.initialState?.generated_at ?? null,
  );
  const [transport, setTransport] = useState<"stream" | "polling" | "connecting">(
    "connecting",
  );
  const pollIntervalSecondsRef = useRef<number | null>(null);
  const etagRef = useRef<string | null>(null);
  const headerSuggestedPollMsRef = useRef<number | null>(null);
  const lastResponseAtMsRef = useRef<number>(0);
  const lastStateChangeAtMsRef = useRef<number>(0);
  const initialGeneratedAt = options?.initialState?.generated_at ?? null;

  useEffect(() => {
    pollIntervalSecondsRef.current =
      typeof liveState?.poll_interval_seconds === "number"
        ? liveState.poll_interval_seconds
        : null;
  }, [liveState?.poll_interval_seconds]);

  useEffect(() => {
    const stateUrl = buildStateUrl(portfolioSlug, detailLevel, accountId);
    const streamUrl = buildStreamUrl(portfolioSlug, detailLevel, accountId);
    etagRef.current = null;
    headerSuggestedPollMsRef.current = null;
    lastResponseAtMsRef.current = 0;
    lastStateChangeAtMsRef.current = initialGeneratedAt ? Date.now() : 0;

    let eventSource: EventSource | null = null;
    let pollTimer: number | null = null;
    let streamWatchdogTimer: number | null = null;
    let streamReconnectTimer: number | null = null;
    let pollAbort: AbortController | null = null;
    let initialAbort: AbortController | null = null;
    let pollingStarted = false;
    let pollingBackground = false;
    let cancelled = false;
    let consecutivePollFailures = 0;
    let pollLoop: (() => Promise<void>) | null = null;
    let lastStreamActivityAt = Date.now();
    let attachEventStream: (() => void) | null = null;

    const clearPollTimer = () => {
      if (pollTimer != null) {
        window.clearTimeout(pollTimer);
        pollTimer = null;
      }
    };

    const clearStreamWatchdog = () => {
      if (streamWatchdogTimer != null) {
        window.clearInterval(streamWatchdogTimer);
        streamWatchdogTimer = null;
      }
    };

    const clearStreamReconnectTimer = () => {
      if (streamReconnectTimer != null) {
        window.clearTimeout(streamReconnectTimer);
        streamReconnectTimer = null;
      }
    };

    const closeStream = () => {
      if (eventSource != null) {
        eventSource.close();
        eventSource = null;
      }
    };

    const refreshState = async (
      signal?: AbortSignal,
      options?: { forceUnconditional?: boolean },
    ): Promise<boolean> => {
      try {
        const requestHeaders = new Headers();
        if (!options?.forceUnconditional && etagRef.current) {
          requestHeaders.set("If-None-Match", etagRef.current);
        }
        const response = await fetch(stateUrl, {
          cache: "no-store",
          signal: withTimeout(signal, LIVE_STATE_FETCH_TIMEOUT_MS),
          headers: requestHeaders,
        });
        const responseAtMs = Date.now();
        lastResponseAtMsRef.current = responseAtMs;
        if (!cancelled) {
          setHeartbeatAt(new Date(responseAtMs).toISOString());
        }
        const etag = response.headers.get("etag");
        if (etag) {
          etagRef.current = etag;
        }
        const nextPollSecondsHeader = response.headers.get("x-live-next-poll-seconds");
        if (nextPollSecondsHeader) {
          const parsedSeconds = Number(nextPollSecondsHeader);
          if (Number.isFinite(parsedSeconds) && parsedSeconds > 0) {
            headerSuggestedPollMsRef.current = Math.max(parsedSeconds * 1000, 250);
          }
        }
        if (response.status === 304) {
          return true;
        }
        if (!response.ok) {
          return false;
        }
        const payload = (await response.json()) as MT5LiveStateResponse;
        lastStateChangeAtMsRef.current = responseAtMs;
        if (!cancelled) {
          setLiveState(payload);
        }
        return true;
      } catch {
        // Keep the previous state on transient fetch failures.
        return false;
      }
    };

    const stopPolling = () => {
      if (!pollingStarted) {
        return;
      }
      pollingStarted = false;
      pollingBackground = false;
      clearPollTimer();
      if (pollAbort) {
        pollAbort.abort();
        pollAbort = null;
      }
    };

    const scheduleStreamReconnect = () => {
      if (!MT5_STREAM_ENABLED || cancelled || !pollingStarted || streamReconnectTimer != null || eventSource != null) {
        return;
      }
      const visibilityMultiplier =
        typeof document !== "undefined" && document.visibilityState === "hidden"
          ? 2
          : 1;
      const delayMs = Math.max(LIVE_STATE_STREAM_RECONNECT_INTERVAL_MS * visibilityMultiplier, 5_000);
      streamReconnectTimer = window.setTimeout(() => {
        streamReconnectTimer = null;
        if (cancelled || !pollingStarted || eventSource != null || attachEventStream == null) {
          return;
        }
        attachEventStream();
      }, delayMs);
    };

    const startPolling = (options?: { background?: boolean }) => {
      const nextBackground = Boolean(options?.background);
      if (pollingStarted) {
        if (!nextBackground && pollingBackground) {
          pollingBackground = false;
          if (!cancelled) {
            setTransport("polling");
          }
        }
        scheduleStreamReconnect();
        return;
      }
      pollingStarted = true;
      pollingBackground = nextBackground;
      if (!cancelled && !pollingBackground) {
        setTransport("polling");
      }
      pollLoop = async () => {
        if (cancelled || !pollingStarted) {
          return;
        }
        if (pollAbort) {
          pollAbort.abort();
        }
        pollAbort = new AbortController();
        const nowBeforeRefresh = Date.now();
        const stateAgeMs =
          lastStateChangeAtMsRef.current > 0
            ? nowBeforeRefresh - lastStateChangeAtMsRef.current
            : Number.POSITIVE_INFINITY;
        const responseAgeMs =
          lastResponseAtMsRef.current > 0
            ? nowBeforeRefresh - lastResponseAtMsRef.current
            : Number.POSITIVE_INFINITY;
        const forceUnconditional =
          stateAgeMs > LIVE_STATE_FORCED_RESYNC_STALE_MS
          || responseAgeMs > LIVE_STATE_FORCED_RESYNC_STALE_MS;
        if (forceUnconditional) {
          etagRef.current = null;
        }
        const ok = await refreshState(pollAbort.signal, {
          forceUnconditional,
        });
        consecutivePollFailures = ok ? 0 : consecutivePollFailures + 1;
        if (cancelled || !pollingStarted) {
          return;
        }
        const serverSuggestedMs = Math.max(
          Number(headerSuggestedPollMsRef.current ?? 0),
          Number(pollIntervalSecondsRef.current ?? 0) * 1000,
          LIVE_STATE_POLL_INTERVAL_MS,
        );
        const uncappedBaseDelayMs = Number.isFinite(serverSuggestedMs) && serverSuggestedMs > 0
          ? serverSuggestedMs
          : LIVE_STATE_POLL_INTERVAL_MS;
        const baseDelayMs = Math.min(
          Math.max(uncappedBaseDelayMs, LIVE_STATE_POLL_INTERVAL_MS),
          LIVE_STATE_MAX_HEALTHY_POLL_INTERVAL_MS,
        );
        const backoffMultiplier =
          consecutivePollFailures <= 0
            ? 1
            : Math.min(1 + (consecutivePollFailures * LIVE_STATE_POLL_FAILURE_STEP), 2.5);
        const visibilityMultiplier =
          typeof document !== "undefined" && document.visibilityState === "hidden"
            ? LIVE_STATE_HIDDEN_TAB_MULTIPLIER
            : 1;
        const effectiveVisibilityMultiplier =
          pollingBackground && eventSource != null ? 1 : visibilityMultiplier;
        const jitterFactor = 1 + ((Math.random() * 2 - 1) * LIVE_STATE_POLL_JITTER_RATIO);
        let delayMs = Math.min(
          Math.max(baseDelayMs * backoffMultiplier * effectiveVisibilityMultiplier * jitterFactor, 250),
          LIVE_STATE_POLL_BACKOFF_MAX_MS,
        );
        const nowAfterRefresh = Date.now();
        const stateDriftMs =
          lastStateChangeAtMsRef.current > 0
            ? nowAfterRefresh - lastStateChangeAtMsRef.current
            : Number.POSITIVE_INFINITY;
        if (
          stateDriftMs > LIVE_STATE_FORCED_RESYNC_STALE_MS
          && (typeof document === "undefined" || document.visibilityState === "visible")
        ) {
          delayMs = Math.min(delayMs, LIVE_STATE_POLL_INTERVAL_MS);
        }
        pollTimer = window.setTimeout(() => {
          if (pollingStarted && pollLoop) {
            void pollLoop();
          }
        }, delayMs);
        scheduleStreamReconnect();
      };
      if (pollLoop) {
        void pollLoop();
      }
      scheduleStreamReconnect();
    };

    const startStreamWatchdog = () => {
      clearStreamWatchdog();
      streamWatchdogTimer = window.setInterval(() => {
        if (cancelled || eventSource == null) {
          return;
        }
        const silenceMs = Date.now() - lastStreamActivityAt;
        if (silenceMs < LIVE_STATE_STREAM_STALL_TIMEOUT_MS) {
          return;
        }
        closeStream();
        startPolling({ background: false });
      }, LIVE_STATE_STREAM_WATCHDOG_INTERVAL_MS);
    };

    attachEventStream = () => {
      if (!MT5_STREAM_ENABLED || cancelled || eventSource != null) {
        return;
      }
      try {
        const stream = new EventSource(streamUrl);
        eventSource = stream;

        stream.onopen = () => {
          lastStreamActivityAt = Date.now();
          if (cancelled) {
            stream.close();
            return;
          }
          setHeartbeatAt(new Date(lastStreamActivityAt).toISOString());
          clearStreamReconnectTimer();
          if (MT5_POLL_WHILE_STREAMING) {
            startPolling({ background: true });
          } else {
            stopPolling();
          }
          setTransport("stream");
        };

        stream.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data) as MT5LiveEventResponse;
            lastStreamActivityAt = Date.now();
            if (!cancelled) {
              setHeartbeatAt(new Date(lastStreamActivityAt).toISOString());
              setLiveState(payload.state);
              setTransport("stream");
            }
          } catch {
            // Ignore malformed keep-alive payloads.
          }
        };

        stream.onerror = () => {
          lastStreamActivityAt = Date.now();
          if (eventSource === stream) {
            stream.close();
            eventSource = null;
          }
          if (!cancelled) {
            startPolling({ background: false });
          }
        };
      } catch {
        startPolling({ background: false });
      }
    };

    const handleVisibilityChange = () => {
      if (cancelled || typeof document === "undefined") {
        return;
      }
      if (document.visibilityState !== "visible") {
        return;
      }
      if (MT5_STREAM_ENABLED && eventSource == null && attachEventStream != null) {
        attachEventStream();
      }
      if (pollingStarted && pollLoop) {
        clearPollTimer();
        void pollLoop();
      }
    };

    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }

    initialAbort = new AbortController();
    void refreshState(initialAbort.signal);

    if (!MT5_STREAM_ENABLED) {
      startPolling({ background: false });
      return () => {
        cancelled = true;
        clearPollTimer();
        clearStreamWatchdog();
        clearStreamReconnectTimer();
        if (typeof document !== "undefined") {
          document.removeEventListener("visibilitychange", handleVisibilityChange);
        }
        if (initialAbort) {
          initialAbort.abort();
        }
        stopPolling();
        closeStream();
      };
    }

    if (MT5_POLL_WHILE_STREAMING) {
      startPolling({ background: true });
    }
    attachEventStream();
    startStreamWatchdog();

    return () => {
      cancelled = true;
      clearPollTimer();
      clearStreamWatchdog();
      clearStreamReconnectTimer();
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      }
      if (initialAbort) {
        initialAbort.abort();
      }
      stopPolling();
      closeStream();
    };
  }, [accountId, detailLevel, initialGeneratedAt, portfolioSlug]);

  return { liveState, transport, heartbeatAt };
}

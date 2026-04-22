"use client";

import { startTransition, useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { ApiError, api } from "@/lib/api/client";
import type { OperatorRunResponse } from "@/lib/api/types";

const TERMINAL_STATUSES = new Set(["succeeded", "failed"]);
const DEFAULT_POLL_MS = 1_500;
const MAX_POLL_BACKOFF_MS = 15_000;

const FALLBACK_SLA_SECONDS: Record<string, number> = {
  sync: 120,
  snapshot: 240,
  backtest: 420,
  report: 360,
};

type UseOperatorRunActionOptions<TPayload> = {
  action: string;
  portfolioSlug?: string;
  accountId?: string;
  enqueue: (payload: TPayload) => Promise<OperatorRunResponse>;
  onSucceeded?: (run: OperatorRunResponse) => void | Promise<void>;
};

export type OperatorActionUiState = "idle" | "queued" | "running" | "success" | "error";

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableApiError(error: unknown): boolean {
  if (error instanceof ApiError) {
    if ([429, 502, 503, 504].includes(error.status)) {
      return true;
    }
    const errorCode = String(error.errorCode ?? "").toLowerCase();
    return [
      "frontend_request_timeout",
      "frontend_request_failed",
      "backend_timeout",
      "backend_unreachable",
    ].includes(errorCode);
  }
  const message = error instanceof Error ? error.message.toLowerCase() : "";
  return (
    message.includes("timeout")
    || message.includes("timed out")
    || message.includes("network")
    || message.includes("fetch failed")
    || message.includes("socket")
    || message.includes("connect")
  );
}

function operatorRunStorageKey(action: string, portfolioSlug: string) {
  return `operator:run:${portfolioSlug}:${action}`;
}

function operatorRunStorageKeyWithAccount(
  action: string,
  portfolioSlug: string,
  accountId?: string,
) {
  const normalizedAccountId = String(accountId ?? "").trim() || "default";
  return `${operatorRunStorageKey(action, portfolioSlug)}:${normalizedAccountId}`;
}

export function useOperatorRunAction<TPayload>({
  action,
  portfolioSlug,
  accountId,
  enqueue,
  onSucceeded,
}: UseOperatorRunActionOptions<TPayload>) {
  const [run, setRun] = useState<OperatorRunResponse | null>(null);
  const [manualError, setManualError] = useState<Error | null>(null);
  const completedRunIds = useRef<Set<number>>(new Set());
  const onSucceededRef = useRef(onSucceeded);
  const lastPayloadRef = useRef<TPayload | null>(null);
  const lastExecuteAtRef = useRef<number>(0);

  useEffect(() => {
    onSucceededRef.current = onSucceeded;
  }, [onSucceeded]);

  async function settleRun(nextRun: OperatorRunResponse) {
    if (completedRunIds.current.has(nextRun.id)) {
      return;
    }
    completedRunIds.current.add(nextRun.id);
    if (nextRun.status === "succeeded") {
      await onSucceededRef.current?.(nextRun);
    }
  }

  useEffect(() => {
    let cancelled = false;
    if (!portfolioSlug) {
      return undefined;
    }
    const storageKey = operatorRunStorageKeyWithAccount(action, portfolioSlug, accountId);

    const loadActiveRun = async () => {
      let recoveredFromStorage = false;
      if (typeof window !== "undefined") {
        const rawRunId = window.sessionStorage.getItem(storageKey);
        const runId = Number(rawRunId);
        if (Number.isFinite(runId) && runId > 0) {
          try {
            const existing = await api.operatorRun(runId);
            if (
              !cancelled
              && existing.action === action
              && !TERMINAL_STATUSES.has(String(existing.status))
            ) {
              recoveredFromStorage = true;
              startTransition(() => setRun(existing));
            }
          } catch {
            // Ignore stale storage and continue with server lookup.
          }
        }
      }

      if (recoveredFromStorage) {
        return;
      }

      try {
        const runs = await api.operatorRuns({
          portfolioSlug,
          accountId,
          action,
          statuses: ["queued", "running"],
          limit: 1,
        });
        if (cancelled || runs.length === 0) {
          return;
        }
        startTransition(() => setRun(runs[0]));
      } catch {
        // Best-effort hydration only.
      }
    };

    void loadActiveRun();
    return () => {
      cancelled = true;
    };
  }, [accountId, action, portfolioSlug]);

  const currentRun = run;
  useEffect(() => {
    if (!portfolioSlug || typeof window === "undefined") {
      return;
    }
    const storageKey = operatorRunStorageKeyWithAccount(action, portfolioSlug, accountId);
    if (currentRun && !TERMINAL_STATUSES.has(String(currentRun.status))) {
      window.sessionStorage.setItem(storageKey, String(currentRun.id));
      return;
    }
    window.sessionStorage.removeItem(storageKey);
  }, [accountId, action, currentRun, portfolioSlug]);

  const activeRunId =
    run && !TERMINAL_STATUSES.has(run.status)
      ? run.id
      : null;

  const pollQuery = useQuery({
    queryKey: ["operator-run", action, accountId ?? "default", activeRunId],
    enabled: activeRunId != null,
    queryFn: async () => api.operatorRun(activeRunId as number),
    retry: 2,
    retryDelay: (attempt) => Math.min(2_000, 250 * (2 ** Math.max(attempt - 1, 0))),
    refetchInterval: (query) => {
      const current = query.state.data as OperatorRunResponse | undefined;
      if (current && TERMINAL_STATUSES.has(current.status)) {
        return false;
      }
      const basePollMs = Number(current?.poll_after_ms ?? run?.poll_after_ms ?? DEFAULT_POLL_MS);
      const normalizedBaseMs = Number.isFinite(basePollMs) ? Math.max(500, basePollMs) : DEFAULT_POLL_MS;
      const failureCount = Math.max(0, Number(query.state.fetchFailureCount ?? 0));
      const multiplier = failureCount <= 0 ? 1 : Math.min(2 ** failureCount, 8);
      return Math.min(MAX_POLL_BACKOFF_MS, Math.round(normalizedBaseMs * multiplier));
    },
    refetchIntervalInBackground: true,
  });

  useEffect(() => {
    const nextRun = pollQuery.data;
    if (nextRun == null) {
      return;
    }
    startTransition(() => setRun(nextRun));
    if (TERMINAL_STATUSES.has(nextRun.status)) {
      void settleRun(nextRun);
    }
  }, [pollQuery.data]);

  const mutation = useMutation({
    mutationFn: async (payload: TPayload) => {
      const maxAttempts = 2;
      let lastError: unknown = null;
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        try {
          return await enqueue(payload);
        } catch (error) {
          lastError = error;
          if (attempt >= maxAttempts || !isRetryableApiError(error)) {
            throw error;
          }
          await delay(Math.min(1_200, 200 * (2 ** Math.max(attempt - 1, 0))));
        }
      }
      throw (lastError instanceof Error ? lastError : new Error("Failed to enqueue operator action."));
    },
    onSuccess: (nextRun) => {
      setManualError(null);
      startTransition(() => setRun(nextRun));
      if (TERMINAL_STATUSES.has(nextRun.status)) {
        void settleRun(nextRun);
      }
    },
    onError: (error) => {
      void (async () => {
        if (portfolioSlug) {
          try {
            const runs = await api.operatorRuns({
              portfolioSlug,
              accountId,
              action,
              statuses: ["queued", "running"],
              limit: 1,
            });
            if (runs.length > 0) {
              setManualError(null);
              startTransition(() => setRun(runs[0]));
              return;
            }
          } catch {
            // Keep original enqueue error below.
          }
        }
        setManualError(
          error instanceof Error ? error : new Error("Failed to enqueue operator action."),
        );
      })();
    },
  });

  const interruptMutation = useMutation({
    mutationFn: async (reason?: string) => {
      const active = pollQuery.data ?? run;
      if (!active || TERMINAL_STATUSES.has(String(active.status))) {
        throw new Error("No running operator action to interrupt.");
      }
      return api.interruptOperatorRun(active.id, reason ?? null);
    },
    onSuccess: (nextRun) => {
      setManualError(null);
      startTransition(() => setRun(nextRun));
      if (TERMINAL_STATUSES.has(String(nextRun.status))) {
        void settleRun(nextRun);
      }
    },
    onError: (error) => {
      setManualError(
        error instanceof Error ? error : new Error("Failed to interrupt operator run."),
      );
    },
  });

  const mergedRun = (pollQuery.data ?? run) as OperatorRunResponse | null;
  const elapsedSeconds = mergedRun?.elapsed_seconds ?? null;
  const actionKey = String(action || "").toLowerCase();
  const fallbackSlaSeconds = FALLBACK_SLA_SECONDS[actionKey] ?? 300;
  const runSlaSeconds = Number(
    mergedRun?.sla_seconds
    ?? mergedRun?.running_timeout_seconds
    ?? mergedRun?.queued_timeout_seconds
    ?? fallbackSlaSeconds,
  );
  const deadlineExceeded = Boolean(
    mergedRun
    && !TERMINAL_STATUSES.has(String(mergedRun.status))
    && Number.isFinite(Number(elapsedSeconds ?? 0))
    && Number.isFinite(runSlaSeconds)
    && (Number(elapsedSeconds ?? 0) >= Number(runSlaSeconds)),
  );

  const pollingError =
    pollQuery.error == null
      ? null
      : pollQuery.error instanceof Error
        ? pollQuery.error
        : new Error("Failed to load operator run status.");

  const mutationError =
    mutation.error instanceof Error
      ? mutation.error
      : mutation.error
        ? new Error("Failed to enqueue operator action.")
        : null;

  const interruptError =
    interruptMutation.error instanceof Error
      ? interruptMutation.error
      : interruptMutation.error
        ? new Error("Failed to interrupt operator action.")
        : null;

  const activeRun = mergedRun && !TERMINAL_STATUSES.has(String(mergedRun.status))
    ? mergedRun
    : null;
  const hasQueued = Boolean(activeRun && String(activeRun.status) === "queued");
  const hasRunning = Boolean(activeRun && String(activeRun.status) === "running");
  const hasSucceeded = Boolean(mergedRun && String(mergedRun.status) === "succeeded");
  const hasFailed = Boolean(mergedRun && String(mergedRun.status) === "failed");

  const baseError =
    manualError
    ?? mutationError
    ?? interruptError
    ?? (
      hasFailed
        ? new Error(
          [
            mergedRun?.error_message ?? "Operator action failed.",
            mergedRun?.error_code ? `code ${mergedRun.error_code}` : null,
            mergedRun?.hint ?? null,
            mergedRun?.request_id ? `request ${mergedRun.request_id}` : null,
            mergedRun?.id ? `run ${mergedRun.id}` : null,
          ].filter(Boolean).join(" "),
        )
        : null
    )
    ?? (mergedRun == null ? pollingError : null);

  const progressPercent = (() => {
    if (hasSucceeded) {
      return 100;
    }
    if (!activeRun) {
      return null;
    }
    const activeElapsed = Number(activeRun.elapsed_seconds ?? 0);
    const activeSla = Number(
      activeRun.sla_seconds
      ?? activeRun.running_timeout_seconds
      ?? activeRun.queued_timeout_seconds
      ?? fallbackSlaSeconds,
    );
    if (!Number.isFinite(activeElapsed) || activeElapsed <= 0) {
      return hasQueued ? 5 : 10;
    }
    if (!Number.isFinite(activeSla) || activeSla <= 0) {
      return hasQueued ? 10 : null;
    }
    const ratio = Math.max(0, Math.min(activeElapsed / activeSla, 1));
    const ceiling = hasQueued ? 35 : 95;
    return Math.max(hasQueued ? 5 : 10, Math.min(Math.round(ratio * 100), ceiling));
  })();

  const uiState: OperatorActionUiState = (
    hasRunning ? "running"
      : hasQueued ? "queued"
        : hasSucceeded ? "success"
          : (hasFailed || baseError != null) ? "error"
            : "idle"
  );

  const execute = (payload: TPayload) => {
    const now = Date.now();
    if (now - lastExecuteAtRef.current < 450) {
      return false;
    }
    const latestRun = (pollQuery.data ?? run) as OperatorRunResponse | null;
    if (
      mutation.isPending
      || interruptMutation.isPending
      || (latestRun != null && !TERMINAL_STATUSES.has(String(latestRun.status)))
    ) {
      const currentStatus = String(latestRun?.status ?? "running");
      setManualError(new Error(`${action} is already ${currentStatus}. Wait for completion.`));
      return false;
    }
    lastExecuteAtRef.current = now;
    lastPayloadRef.current = payload;
    setManualError(null);
    mutation.mutate(payload);
    return true;
  };

  const retry = () => {
    if (lastPayloadRef.current == null) {
      setManualError(new Error("No previous payload available to retry."));
      return false;
    }
    return execute(lastPayloadRef.current);
  };

  return {
    run: mergedRun,
    elapsedSeconds,
    deadlineExceeded,
    uiState,
    progressPercent,
    statusLabel: hasRunning ? "running" : hasQueued ? "queued" : hasSucceeded ? "succeeded" : hasFailed ? "failed" : "idle",
    lastUpdatedAt: mergedRun?.updated_at ?? mergedRun?.finished_at ?? mergedRun?.created_at ?? null,
    pending:
      mutation.isPending
      || interruptMutation.isPending
      || Boolean(mergedRun && !TERMINAL_STATUSES.has(String(mergedRun.status))),
    interrupting: interruptMutation.isPending,
    canInterrupt: Boolean(mergedRun && !TERMINAL_STATUSES.has(String(mergedRun.status))),
    error: baseError,
    execute,
    retry,
    interrupt: (reason?: string) => {
      setManualError(null);
      interruptMutation.mutate(reason);
    },
    reset: () => {
      setManualError(null);
      startTransition(() => setRun(null));
      if (portfolioSlug && typeof window !== "undefined") {
        window.sessionStorage.removeItem(
          operatorRunStorageKeyWithAccount(action, portfolioSlug, accountId),
        );
      }
    },
  };
}

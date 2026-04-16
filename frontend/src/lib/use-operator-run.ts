"use client";

import { startTransition, useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api/client";
import type { OperatorRunResponse } from "@/lib/api/types";

const TERMINAL_STATUSES = new Set(["succeeded", "failed"]);
const OPERATOR_DEADLINE_MS = 120_000;
const DEFAULT_POLL_MS = 1_500;

type UseOperatorRunActionOptions<TPayload> = {
  action: string;
  portfolioSlug?: string;
  enqueue: (payload: TPayload) => Promise<OperatorRunResponse>;
  onSucceeded?: (run: OperatorRunResponse) => void | Promise<void>;
};

export function useOperatorRunAction<TPayload>({
  action,
  portfolioSlug,
  enqueue,
  onSucceeded,
}: UseOperatorRunActionOptions<TPayload>) {
  const [run, setRun] = useState<OperatorRunResponse | null>(null);
  const [manualError, setManualError] = useState<Error | null>(null);
  const completedRunIds = useRef<Set<number>>(new Set());
  const onSucceededRef = useRef(onSucceeded);

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
    void api.operatorRuns({
      portfolioSlug,
      action,
      statuses: ["queued", "running"],
      limit: 1,
    })
      .then((runs) => {
        if (cancelled || runs.length === 0) {
          return;
        }
        startTransition(() => setRun(runs[0]));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [action, portfolioSlug]);

  const baseElapsedSeconds = Number(run?.elapsed_seconds ?? 0);
  const baseDeadlineExceeded = Boolean(
    run
    && !TERMINAL_STATUSES.has(run.status)
    && Number.isFinite(baseElapsedSeconds)
    && (baseElapsedSeconds * 1000) >= OPERATOR_DEADLINE_MS,
  );

  const activeRunId =
    run && !TERMINAL_STATUSES.has(run.status) && !baseDeadlineExceeded
      ? run.id
      : null;

  const pollQuery = useQuery({
    queryKey: ["operator-run", action, activeRunId],
    enabled: activeRunId != null,
    queryFn: async () => api.operatorRun(activeRunId as number),
    refetchInterval: (query) => {
      const current = query.state.data as OperatorRunResponse | undefined;
      const currentElapsedSeconds = Number(current?.elapsed_seconds ?? run?.elapsed_seconds ?? 0);
      if (
        Number.isFinite(currentElapsedSeconds)
        && (currentElapsedSeconds * 1000) >= OPERATOR_DEADLINE_MS
      ) {
        return false;
      }
      if (current && TERMINAL_STATUSES.has(current.status)) {
        return false;
      }
      const pollAfter = Number(current?.poll_after_ms ?? run?.poll_after_ms ?? DEFAULT_POLL_MS);
      return Number.isFinite(pollAfter) ? Math.max(500, pollAfter) : DEFAULT_POLL_MS;
    },
    retry: 1,
    refetchIntervalInBackground: true,
  });

  useEffect(() => {
    const nextRun = pollQuery.data;
    if (nextRun == null) {
      return;
    }
    if (TERMINAL_STATUSES.has(nextRun.status)) {
      void settleRun(nextRun);
    }
  }, [pollQuery.data]);

  const mutation = useMutation({
    mutationFn: enqueue,
    onSuccess: (nextRun) => {
      setManualError(null);
      startTransition(() => setRun(nextRun));
      if (TERMINAL_STATUSES.has(nextRun.status)) {
        void settleRun(nextRun);
      }
    },
    onError: (error) => {
      setManualError(error instanceof Error ? error : new Error("Failed to enqueue operator action."));
    },
  });

  const currentRun = (pollQuery.data ?? run) as OperatorRunResponse | null;
  const elapsedSeconds = currentRun?.elapsed_seconds ?? null;
  const deadlineExceeded = Boolean(
    currentRun
    && !TERMINAL_STATUSES.has(currentRun.status)
    && Number.isFinite(Number(elapsedSeconds ?? 0))
    && (Number(elapsedSeconds ?? 0) * 1000) >= OPERATOR_DEADLINE_MS,
  );
  const deadlineError = (() => {
    if (currentRun == null || TERMINAL_STATUSES.has(currentRun.status) || !deadlineExceeded) {
      return null;
    }
    const stage = (currentRun.stage ?? "processing").replaceAll("_", " ");
    const hint = currentRun.hint ?? "Retry the action or inspect backend logs using this run id.";
    return new Error(`Run ${currentRun.id} is still ${stage} after 120s. ${hint}`);
  })();
  const pollingError =
    pollQuery.error == null
      ? null
      : pollQuery.error instanceof Error
        ? pollQuery.error
        : new Error("Failed to load operator run status.");
  const hasMutationError = mutation.error != null;
  const hasBlockingError = manualError != null || hasMutationError || pollingError != null;

  return {
    run: currentRun,
    elapsedSeconds,
    deadlineExceeded,
    pending:
      mutation.isPending
      || Boolean(
        currentRun
        && !TERMINAL_STATUSES.has(currentRun.status)
        && !deadlineExceeded
        && !hasBlockingError,
      ),
    error:
      manualError
      ?? (mutation.error instanceof Error
        ? mutation.error
        : mutation.error
          ? new Error("Failed to enqueue operator action.")
          : currentRun?.status === "failed"
            ? new Error(
              [
                currentRun.error_message ?? "Operator action failed.",
                currentRun.error_code ? `code ${currentRun.error_code}` : null,
                currentRun.hint ?? null,
                currentRun.request_id ? `request ${currentRun.request_id}` : null,
                currentRun.id ? `run ${currentRun.id}` : null,
              ].filter(Boolean).join(" "),
            )
            : deadlineError
              ?? pollingError),
    execute: (payload: TPayload) => {
      setManualError(null);
      mutation.mutate(payload);
    },
    reset: () => {
      setManualError(null);
      startTransition(() => setRun(null));
    },
  };
}

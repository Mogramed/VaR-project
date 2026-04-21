"use client";

import { startTransition, useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api/client";
import type { ExecutionFillResponse, ExecutionResultResponse } from "@/lib/api/types";

export function useRecentExecutionActivity({
  portfolioSlug,
  accountId,
  initialExecutions,
  initialFills,
  liveSequence,
  executionLimit = 20,
  fillLimit = 20,
}: {
  portfolioSlug: string;
  accountId?: string;
  initialExecutions: ExecutionResultResponse[];
  initialFills: ExecutionFillResponse[];
  liveSequence?: number;
  executionLimit?: number;
  fillLimit?: number;
}) {
  const [executions, setExecutions] = useState<ExecutionResultResponse[]>(initialExecutions);
  const [fills, setFills] = useState<ExecutionFillResponse[]>(initialFills);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [nextExecutions, nextFills] = await Promise.all([
          api.recentExecutionResults(portfolioSlug, executionLimit, accountId),
          api.recentExecutionFills(portfolioSlug, fillLimit, accountId),
        ]);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setExecutions(nextExecutions);
          setFills(nextFills);
        });
      } catch {
        // Keep the current activity on transient API failures.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accountId, executionLimit, fillLimit, liveSequence, portfolioSlug]);

  const pushExecutionResult = useCallback(
    (result: ExecutionResultResponse) => {
      setExecutions((current) => {
        const next = [result, ...current.filter((item) => item.id !== result.id)];
        return next.slice(0, executionLimit);
      });
      if ((result.fills ?? []).length === 0) {
        return;
      }
      setFills((current) => {
        const incoming = (result.fills ?? []).map((fill) => ({
          ...fill,
          execution_result_id: fill.execution_result_id ?? result.id ?? null,
          portfolio_id: fill.portfolio_id ?? result.portfolio_id ?? null,
          created_at: fill.created_at ?? result.created_at ?? result.time_utc,
        }));
        const existing = current.filter(
          (item) =>
            !incoming.some(
              (fill) =>
                (fill.id != null && item.id === fill.id) ||
                (fill.deal_ticket != null && item.deal_ticket === fill.deal_ticket),
            ),
        );
        return [...incoming, ...existing].slice(0, fillLimit);
      });
    },
    [executionLimit, fillLimit],
  );

  return { executions, fills, pushExecutionResult };
}

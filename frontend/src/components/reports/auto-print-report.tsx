"use client";

import { useEffect } from "react";

const PRINT_READY_TIMEOUT_MS = 12_000;

function chartsReady() {
  const charts = Array.from(document.querySelectorAll("[data-chart-surface]"));
  if (charts.length === 0) {
    return true;
  }
  return charts.every((node) => node.querySelector("svg"));
}

export function AutoPrintReport({ enabled }: { enabled: boolean }) {
  useEffect(() => {
    if (!enabled || typeof window === "undefined") {
      return;
    }

    let cancelled = false;
    let printed = false;
    let timer: number | null = null;
    let timeout: number | null = null;

    const clearScheduledChecks = () => {
      if (timer != null) {
        window.clearTimeout(timer);
        timer = null;
      }
      if (timeout != null) {
        window.clearTimeout(timeout);
        timeout = null;
      }
    };

    const triggerPrint = () => {
      if (cancelled || printed) {
        return;
      }
      printed = true;
      clearScheduledChecks();
      window.requestAnimationFrame(() => {
        if (!cancelled) {
          window.print();
        }
      });
    };

    const pollUntilReady = () => {
      if (cancelled) {
        return;
      }
      if (chartsReady()) {
        triggerPrint();
        return;
      }
      timer = window.setTimeout(pollUntilReady, 250);
    };

    pollUntilReady();
    timeout = window.setTimeout(triggerPrint, PRINT_READY_TIMEOUT_MS);

    return () => {
      cancelled = true;
      clearScheduledChecks();
    };
  }, [enabled]);

  return null;
}

"use client";

import { useEffect, useState } from "react";
import { formatRelativeTime } from "@/lib/utils";

/**
 * Returns a ticking relative time string ("3s ago", "2m ago") that updates
 * every `intervalMs` milliseconds. Useful for showing freshness of live data.
 */
export function useRelativeTime(
  timestamp: string | null | undefined,
  intervalMs = 1_000,
): string {
  const [text, setText] = useState(() => formatRelativeTime(timestamp));

  useEffect(() => {
    if (!timestamp) return;
    const id = setInterval(() => {
      setText(formatRelativeTime(timestamp));
    }, intervalMs);
    return () => clearInterval(id);
  }, [timestamp, intervalMs]);

  return text;
}

"use client";

import type { HeadlineRiskPointResponse } from "@/lib/api/types";

export function pickHeadlineRisk(
  points: HeadlineRiskPointResponse[] | null | undefined,
  key: string,
) {
  return (points ?? []).find((item) => item.key === key) ?? null;
}

export function preferredHeadlineRisk(
  points: HeadlineRiskPointResponse[] | null | undefined,
  keys: string[],
) {
  for (const key of keys) {
    const found = pickHeadlineRisk(points, key);
    if (found) return found;
  }
  return (points ?? [])[0] ?? null;
}

export function formatRiskConfidence(alpha: number | null | undefined) {
  if (alpha == null) return "n/a";
  return `${Math.round(alpha * 100)}%`;
}

export function formatRiskHorizon(days: number | null | undefined) {
  if (days == null) return "n/a";
  return `${days}d`;
}

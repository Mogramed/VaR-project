"use client";

function parseMs(value: string | undefined, fallback: number, minimum: number) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(Math.round(parsed), minimum);
}

export const DESK_ARTIFACT_QUERY_STALE_MS = parseMs(
  process.env.NEXT_PUBLIC_DESK_ARTIFACT_QUERY_STALE_MS,
  2_500,
  500,
);
export const DESK_ARTIFACT_QUERY_GC_MS = 5 * 60_000;
export const DESK_ARTIFACT_QUERY_REFETCH_MS = parseMs(
  process.env.NEXT_PUBLIC_DESK_ARTIFACT_QUERY_REFETCH_MS,
  2_500,
  500,
);

export const deskArtifactQueryMeta = {
  deskArtifact: true,
} as const;

export const deskArtifactQueryOptions = {
  staleTime: DESK_ARTIFACT_QUERY_STALE_MS,
  gcTime: DESK_ARTIFACT_QUERY_GC_MS,
  refetchInterval: DESK_ARTIFACT_QUERY_REFETCH_MS,
  refetchIntervalInBackground: false,
  refetchOnWindowFocus: false,
  refetchOnReconnect: true,
  meta: deskArtifactQueryMeta,
} as const;

export function deskArtifactQueryKey(...parts: Array<string | number | null | undefined>) {
  return ["desk-artifact", ...parts.map((item) => (item == null ? "none" : String(item)))] as const;
}

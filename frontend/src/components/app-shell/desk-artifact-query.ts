"use client";

export const DESK_ARTIFACT_QUERY_STALE_MS = 1_000;
export const DESK_ARTIFACT_QUERY_GC_MS = 5 * 60_000;
export const DESK_ARTIFACT_QUERY_REFETCH_MS = 1_000;

export const deskArtifactQueryMeta = {
  deskArtifact: true,
} as const;

export const deskArtifactQueryOptions = {
  staleTime: DESK_ARTIFACT_QUERY_STALE_MS,
  gcTime: DESK_ARTIFACT_QUERY_GC_MS,
  refetchInterval: DESK_ARTIFACT_QUERY_REFETCH_MS,
  refetchIntervalInBackground: true,
  refetchOnWindowFocus: false,
  refetchOnReconnect: true,
  meta: deskArtifactQueryMeta,
} as const;

export function deskArtifactQueryKey(...parts: Array<string | number | null | undefined>) {
  return ["desk-artifact", ...parts.map((item) => (item == null ? "none" : String(item)))] as const;
}

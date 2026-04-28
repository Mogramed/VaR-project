"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

function parseMs(value: string | undefined, fallback: number, minimum: number) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(Math.round(parsed), minimum);
}

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const defaultStaleMs = parseMs(process.env.NEXT_PUBLIC_QUERY_STALE_MS, 2_000, 500);
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: defaultStaleMs,
            gcTime: 5 * 60_000,
            refetchOnWindowFocus: false,
            refetchOnReconnect: true,
            retry: 1,
          },
        },
      }),
  );

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

"use client";

import { startTransition, createContext, useContext, useRef, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";

import type { MT5LiveStateResponse, OperatorRunResponse } from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";

type DeskLiveTransport = "stream" | "polling" | "connecting";

interface DeskLiveContextValue {
  portfolioSlug?: string;
  liveState: MT5LiveStateResponse | null;
  transport: DeskLiveTransport;
  artifactVersion: number;
  lastCompletedRun: OperatorRunResponse | null;
  notifyOperatorRunCompleted: (run: OperatorRunResponse) => void;
}

const DeskLiveContext = createContext<DeskLiveContextValue | null>(null);

export function DeskLiveProvider({
  portfolioSlug,
  children,
}: {
  portfolioSlug?: string;
  children: ReactNode;
}) {
  const pathname = usePathname();
  const [artifactVersion, setArtifactVersion] = useState(0);
  const [lastCompletedRun, setLastCompletedRun] = useState<OperatorRunResponse | null>(null);
  const lastCompletedSignatureRef = useRef<string>("");
  const detailLevel =
    pathname === "/desk/live"
      ? "inspector"
      : pathname === "/desk/blotter" || pathname === "/desk/incidents"
        ? "full"
      : "summary";
  const { liveState, transport } = useMt5LiveState(portfolioSlug, {
    detailLevel,
  });

  const notifyOperatorRunCompleted = (run: OperatorRunResponse) => {
    const signature = `${run.id}:${run.status}:${run.updated_at ?? run.finished_at ?? ""}`;
    if (lastCompletedSignatureRef.current === signature) {
      return;
    }
    lastCompletedSignatureRef.current = signature;
    startTransition(() => {
      setLastCompletedRun(run);
      setArtifactVersion((current) => current + 1);
    });
  };

  return (
    <DeskLiveContext.Provider
      value={{
        portfolioSlug,
        liveState,
        transport,
        artifactVersion,
        lastCompletedRun,
        notifyOperatorRunCompleted,
      }}
    >
      {children}
    </DeskLiveContext.Provider>
  );
}

export function useDeskLive() {
  const context = useContext(DeskLiveContext);
  if (context == null) {
    throw new Error("useDeskLive must be used inside DeskLiveProvider.");
  }
  return context;
}

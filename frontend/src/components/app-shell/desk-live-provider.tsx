"use client";

import { startTransition, createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";

import type { MT5LiveStateResponse, OperatorRunResponse } from "@/lib/api/types";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";

type DeskLiveTransport = "stream" | "polling" | "connecting";

interface DeskLiveContextValue {
  portfolioSlug?: string;
  liveState: MT5LiveStateResponse | null;
  heartbeatAt: string | null;
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
  const lastHeartbeatSignatureRef = useRef<string>("");
  const lastHeartbeatAtMsRef = useRef<number>(0);
  const queryClient = useQueryClient();
  const detailLevel =
    pathname === "/desk/live"
      ? "full"
      : pathname === "/desk/blotter" || pathname === "/desk/incidents"
        ? "full"
        : "summary";
  const { liveState, transport, heartbeatAt } = useMt5LiveState(portfolioSlug, {
    detailLevel,
  });

  useEffect(() => {
    if (artifactVersion <= 0) {
      return;
    }
    void queryClient.invalidateQueries({
      predicate: (query) => {
        const meta = query.meta as Record<string, unknown> | undefined;
        return Boolean(meta?.deskArtifact);
      },
    });
  }, [artifactVersion, queryClient]);

  useEffect(() => {
    if (liveState == null) {
      return;
    }
    const heartbeatSignature = [
      portfolioSlug ?? "default",
      String(liveState.sequence ?? "0"),
      String(liveState.generated_at ?? ""),
      String(liveState.status ?? ""),
    ].join(":");
    if (lastHeartbeatSignatureRef.current === heartbeatSignature) {
      return;
    }
    const nowMs = Date.now();
    if (nowMs - lastHeartbeatAtMsRef.current < 800) {
      return;
    }
    lastHeartbeatAtMsRef.current = nowMs;
    lastHeartbeatSignatureRef.current = heartbeatSignature;
    startTransition(() => {
      setArtifactVersion((current) => current + 1);
    });
  }, [liveState, portfolioSlug]);

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
        heartbeatAt,
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

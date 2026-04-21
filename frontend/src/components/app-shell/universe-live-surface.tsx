"use client";

import { useQuery } from "@tanstack/react-query";

import { LivePostureBanner } from "@/components/app-shell/live-posture-banner";
import { LiveRuntimeBadgeGroup } from "@/components/app-shell/live-runtime-badge-group";
import { PageHeader } from "@/components/app-shell/page-header";
import {
  deskArtifactQueryKey,
  deskArtifactQueryOptions,
} from "@/components/app-shell/desk-artifact-query";
import { ChartSurface } from "@/components/charts/chart-surface";
import { InstrumentUniverseTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { CHART_PALETTE, makeBarOption } from "@/lib/chart-options";
import type {
  InstrumentDefinitionResponse,
  MarketDataSyncStatusResponse,
  MT5TerminalStatusResponse,
} from "@/lib/api/types";
import { buildInstrumentClassCounts } from "@/lib/view-models";
import { formatTimestamp } from "@/lib/utils";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";

export function UniverseLiveSurface({
  portfolioSlug,
  initialInstruments,
  initialMarketStatus,
  initialMt5Status,
}: {
  portfolioSlug: string;
  initialInstruments: InstrumentDefinitionResponse[];
  initialMarketStatus: MarketDataSyncStatusResponse | null;
  initialMt5Status: MT5TerminalStatusResponse | null;
}) {
  const { liveState, transport, accountId } = useDeskLive();
  const instrumentsQuery = useQuery({
    queryKey: deskArtifactQueryKey("universe", "instruments", portfolioSlug),
    queryFn: () => api.instruments(portfolioSlug),
    initialData: initialInstruments,
    ...deskArtifactQueryOptions,
  });
  const marketStatusQuery = useQuery({
    queryKey: deskArtifactQueryKey("universe", "market-status", portfolioSlug),
    queryFn: () => api.marketDataStatus(portfolioSlug),
    initialData: initialMarketStatus,
    ...deskArtifactQueryOptions,
  });
  const mt5StatusQuery = useQuery({
    queryKey: deskArtifactQueryKey("universe", "mt5-status", portfolioSlug, accountId ?? "default"),
    queryFn: () => api.mt5Status(accountId),
    initialData: initialMt5Status,
    ...deskArtifactQueryOptions,
  });

  const instruments = instrumentsQuery.data ?? initialInstruments;
  const marketStatus = marketStatusQuery.data ?? initialMarketStatus;
  const mt5Status = mt5StatusQuery.data ?? initialMt5Status;

  const classCounts = buildInstrumentClassCounts(instruments);
  const missingSymbols = marketStatus?.missing_symbols ?? [];
  const missingBars = marketStatus?.missing_bars ?? [];
  const trackedSymbols = marketStatus?.symbols ?? [];
  const syncedSymbols = instruments.length - missingSymbols.length;
  const retentionTiers = marketStatus?.retention_tiers ?? {};
  const tickArchive = marketStatus?.tick_archive ?? null;
  const liveBridgeStatus = marketStatus?.live_bridge_status
    ?? liveState?.status
    ?? (mt5Status?.connected ? "ok" : mt5Status?.message ?? null);
  const liveBridgeTimestamp = marketStatus?.live_bridge_generated_at
    ?? liveState?.generated_at
    ?? mt5Status?.timestamp_utc
    ?? null;
  const tierSummary = Object.entries(retentionTiers)
    .map(([timeframe, days]) => `${timeframe} ${days}d`)
    .join(" / ");

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Universe"
        title="Instrument definitions and tradability constraints from MT5."
        aside={(
          <>
            <StatusBadge
              label={liveBridgeStatus ?? marketStatus?.status ?? "unknown"}
              tone={(liveBridgeStatus ?? marketStatus?.status) === "ok" ? "success" : "warning"}
            />
            <LiveRuntimeBadgeGroup liveState={liveState} transport={transport} showBridge={false} />
          </>
        )}
      />
      <LivePostureBanner liveState={liveState} transport={transport} />

      <section className="grid gap-4 xl:grid-cols-6">
        <MetricBlock
          label="Tracked symbols"
          value={String(trackedSymbols.length || instruments.length)}
          hint={`${instruments.length} cached definitions`}
          tone="accent"
        />
        <MetricBlock
          label="Synced"
          value={String(syncedSymbols)}
          hint={`${missingSymbols.length} missing`}
          tone={syncedSymbols > 0 ? "success" : "warning"}
        />
        <MetricBlock
          label="Asset classes"
          value={String(classCounts.length)}
          hint="Derived from MT5 metadata"
        />
        <MetricBlock
          label="Missing bars"
          value={String(missingBars.length)}
          hint={
            liveBridgeTimestamp
              ? `Live ${formatTimestamp(liveBridgeTimestamp)}`
              : "Current default timeframe"
          }
          tone={missingBars.length > 0 ? "warning" : "success"}
        />
        <MetricBlock
          label="Retention tiers"
          value={tierSummary || "n/a"}
          hint={marketStatus?.coverage_status ?? "coverage"}
          tone="accent"
        />
        <MetricBlock
          label="Tick archive"
          value={String(tickArchive?.row_count ?? 0)}
          hint={
            typeof tickArchive?.latest_tick_at === "string"
              ? `Latest ${formatTimestamp(tickArchive.latest_tick_at)}`
              : "No ticks archived yet"
          }
          tone={tickArchive?.row_count ? "success" : "warning"}
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        <ChartSurface
          option={makeBarOption(classCounts, {
            color: CHART_PALETTE.gold,
            negativeColor: CHART_PALETTE.green,
            mode: classCounts.length <= 3 ? "sparse" : "comparison",
          })}
          mode={classCounts.length <= 3 ? "sparse" : "comparison"}
          dataCount={classCounts.length}
          eyebrow="Coverage"
          title="Instrument mix by asset class"
          description="Universe onboarding is now visible as a desk concern, not just a connector implementation detail."
          meta={marketStatus?.latest_sync_at ?? "Awaiting MT5 sync"}
          emptyState={(
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No instrument definitions have been synchronized yet.
            </div>
          )}
        />

        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Instrument registry
          </div>
          <InstrumentUniverseTable rows={instruments} />
        </div>
      </section>
    </div>
  );
}

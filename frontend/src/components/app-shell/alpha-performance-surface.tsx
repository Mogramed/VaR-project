"use client";

import { useQuery } from "@tanstack/react-query";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { DashboardActiveFilters } from "@/components/app-shell/dashboard-active-filters";
import {
  DecisionAlphaForecastPanel,
  DecisionAlphaPortfolioForecastPanel,
  DecisionAlphaProjectionPanel,
  DecisionAlphaReplayPanel,
  DecisionAlphaTrajectoryPanel,
} from "@/components/app-shell/decision-alpha-panels";
import { PageHeader } from "@/components/app-shell/page-header";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type {
  DecisionBacktestTrajectoryResponse,
  DecisionForecastResponse,
  DecisionPortfolioForecastResponse,
  DecisionReplayResponse,
} from "@/lib/api/types";
import {
  deskArtifactQueryKey,
  deskArtifactQueryOptions,
} from "@/components/app-shell/desk-artifact-query";
import { useDashboardPrefs } from "@/lib/dashboard-preferences-context";

const PAST_LOOKBACK_DAYS = 90;
const PROJECTION_LOOKBACK_DAYS = 30;
const LONG_HORIZON_DAYS = 150;

export function AlphaPerformanceSurface({
  portfolioSlug,
  initialReplay,
  initialForecast,
  initialTrajectory,
  initialProjectionForecast,
  initialProjectionTrajectory,
  initialPortfolioForecast,
}: {
  portfolioSlug: string;
  initialReplay: DecisionReplayResponse | null;
  initialForecast: DecisionForecastResponse | null;
  initialTrajectory?: DecisionBacktestTrajectoryResponse | null;
  initialProjectionForecast?: DecisionForecastResponse | null;
  initialProjectionTrajectory?: DecisionBacktestTrajectoryResponse | null;
  initialPortfolioForecast?: DecisionPortfolioForecastResponse | null;
}) {
  const { accountId } = useDeskLive();
  const { preferredHorizonDays } = useDashboardPrefs();
  const replayQuery = useQuery({
    queryKey: deskArtifactQueryKey("alpha-performance-replay", portfolioSlug, accountId ?? "default", 600),
    queryFn: () =>
      api.decisionAlphaReplayWindow(portfolioSlug, {
        limit: 600,
        lookbackDays: PAST_LOOKBACK_DAYS,
      }),
    initialData: initialReplay ?? undefined,
    ...deskArtifactQueryOptions,
  });
  const replay = replayQuery.data ?? initialReplay ?? null;
  const forecastSymbol =
    replay?.predicted_vs_realized?.[replay.predicted_vs_realized.length - 1]?.symbol
    ?? initialForecast?.symbol
    ?? "EURUSD";
  const forecastQuery = useQuery({
    queryKey: [
      "alpha-performance-forecast",
      portfolioSlug,
      accountId ?? "default",
      forecastSymbol,
      preferredHorizonDays,
    ],
    queryFn: () =>
      api.decisionAlphaForecast(forecastSymbol, {
        portfolioSlug,
        horizonDays: preferredHorizonDays,
      }),
    initialData:
      initialForecast != null
      && initialForecast.symbol === forecastSymbol
      && initialForecast.horizon_days === preferredHorizonDays
        ? initialForecast
        : undefined,
    ...deskArtifactQueryOptions,
  });
  const forecast = forecastQuery.data ?? null;
  const trajectoryQuery = useQuery({
    queryKey: [
      "alpha-performance-trajectory",
      portfolioSlug,
      accountId ?? "default",
      forecastSymbol,
      PAST_LOOKBACK_DAYS,
    ],
    queryFn: () =>
      api.decisionAlphaTrajectory(forecastSymbol, {
        portfolioSlug,
        lookbackDays: PAST_LOOKBACK_DAYS,
      }),
    initialData:
      initialTrajectory != null
      && initialTrajectory.symbol === forecastSymbol
      && initialTrajectory.lookback_days === PAST_LOOKBACK_DAYS
        ? initialTrajectory
        : undefined,
    ...deskArtifactQueryOptions,
  });
  const trajectory = trajectoryQuery.data ?? null;
  const projectionForecastQuery = useQuery({
    queryKey: [
      "alpha-performance-projection-forecast",
      portfolioSlug,
      accountId ?? "default",
      forecastSymbol,
      LONG_HORIZON_DAYS,
    ],
    queryFn: () =>
      api.decisionAlphaForecast(forecastSymbol, {
        portfolioSlug,
        horizonDays: LONG_HORIZON_DAYS,
      }),
    initialData:
      initialProjectionForecast != null
      && initialProjectionForecast.symbol === forecastSymbol
      && initialProjectionForecast.horizon_days === LONG_HORIZON_DAYS
        ? initialProjectionForecast
        : undefined,
    ...deskArtifactQueryOptions,
  });
  const projectionForecast = projectionForecastQuery.data ?? null;
  const projectionTrajectoryQuery = useQuery({
    queryKey: [
      "alpha-performance-projection-trajectory",
      portfolioSlug,
      accountId ?? "default",
      forecastSymbol,
      PROJECTION_LOOKBACK_DAYS,
    ],
    queryFn: () =>
      api.decisionAlphaTrajectory(forecastSymbol, {
        portfolioSlug,
        lookbackDays: PROJECTION_LOOKBACK_DAYS,
      }),
    initialData:
      initialProjectionTrajectory != null
      && initialProjectionTrajectory.symbol === forecastSymbol
      && initialProjectionTrajectory.lookback_days === PROJECTION_LOOKBACK_DAYS
        ? initialProjectionTrajectory
        : undefined,
    ...deskArtifactQueryOptions,
  });
  const projectionTrajectory = projectionTrajectoryQuery.data ?? null;
  const portfolioForecastQuery = useQuery({
    queryKey: [
      "alpha-performance-portfolio-forecast",
      portfolioSlug,
      accountId ?? "default",
      LONG_HORIZON_DAYS,
    ],
    queryFn: () =>
      api.decisionAlphaPortfolioForecast({
        portfolioSlug,
        horizonDays: LONG_HORIZON_DAYS,
      }),
    initialData:
      initialPortfolioForecast != null
      && initialPortfolioForecast.horizon_days === LONG_HORIZON_DAYS
        ? initialPortfolioForecast
        : undefined,
    ...deskArtifactQueryOptions,
  });
  const portfolioForecast = portfolioForecastQuery.data ?? null;

  return (
    <div className="desk-page space-y-4">
      <PageHeader
        eyebrow="Alpha"
        title="Replay & forecast"
        aside={<StatusBadge label={portfolioSlug} tone="accent" />}
      />
      <DashboardActiveFilters showModel={false} />

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Replay (historical proof)
          </h4>
          <DecisionAlphaReplayPanel replay={replay} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Forecast scenarios ({forecastSymbol})
          </h4>
          <DecisionAlphaForecastPanel forecast={forecast} />
        </div>
      </div>

      <div className="space-y-2">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
          Last month + next 5 months ({forecastSymbol})
        </h4>
        <DecisionAlphaProjectionPanel
          trajectory={projectionTrajectory}
          forecast={projectionForecast}
          historyWindowDays={PROJECTION_LOOKBACK_DAYS}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Past {PAST_LOOKBACK_DAYS}d trajectory ({forecastSymbol})
          </h4>
          <DecisionAlphaTrajectoryPanel trajectory={trajectory} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Portfolio forecast ({LONG_HORIZON_DAYS}d) + PnL
          </h4>
          <DecisionAlphaPortfolioForecastPanel portfolioForecast={portfolioForecast} />
        </div>
      </div>
    </div>
  );
}

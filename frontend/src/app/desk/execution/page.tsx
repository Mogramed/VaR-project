import { ExecutionLiveSurface } from "@/components/app-shell/execution-live-surface";
import { api } from "@/lib/api/client";

export default async function DeskExecutionPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const initialSymbol =
    typeof query.symbol === "string" && query.symbol.trim().length > 0
      ? query.symbol.toUpperCase()
      : undefined;
  const initialExposureRaw =
    typeof query.exposure === "string" ? Number(query.exposure) : undefined;
  const initialExposureChange =
    initialExposureRaw != null && Number.isFinite(initialExposureRaw)
      ? initialExposureRaw
      : undefined;
  const initialSide =
    typeof query.side === "string" && (query.side === "buy" || query.side === "sell")
      ? query.side
      : undefined;
  const health = await api.safeHealth();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;

  const [liveState, recentExecutions, recentFills] = await Promise.all([
    api.mt5LiveState(resolvedPortfolio).catch(() => null),
    api.recentExecutionResults(resolvedPortfolio, 12).catch(() => []),
    api.recentExecutionFills(resolvedPortfolio, 12).catch(() => []),
  ]);
  const status = liveState?.terminal_status ?? (await api.mt5Status());

  return (
    <ExecutionLiveSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
      initialLiveState={liveState}
      initialTerminalStatus={status}
      initialExecutions={recentExecutions}
      initialFills={recentFills}
      initialSymbol={initialSymbol}
      initialExposureChange={initialExposureChange}
      initialSide={initialSide}
    />
  );
}

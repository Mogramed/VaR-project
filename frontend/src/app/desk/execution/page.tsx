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
  const accountId =
    typeof query.account === "string" ? query.account : undefined;
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

  const [recentExecutions, recentFills] = await Promise.all([
    api.recentExecutionResults(resolvedPortfolio, 12, accountId).catch(() => []),
    api.recentExecutionFills(resolvedPortfolio, 12, accountId).catch(() => []),
  ]);
  const status = await api.mt5Status(accountId);

  return (
    <ExecutionLiveSurface
      key={`${resolvedPortfolio}:${accountId ?? "default"}`}
      portfolioSlug={resolvedPortfolio}
      initialTerminalStatus={status}
      initialExecutions={recentExecutions}
      initialFills={recentFills}
      initialSymbol={initialSymbol}
      initialExposureChange={initialExposureChange}
      initialSide={initialSide}
    />
  );
}

import { Mt5BlotterLiveSurface } from "@/components/app-shell/mt5-blotter-live-surface";
import { api } from "@/lib/api/client";

export default async function DeskBlotterPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const resolvedPortfolio = portfolioSlug ?? (await api.safeHealth()).portfolio_slug;

  const [liveState, executions, fills, audit] = await Promise.all([
    api.mt5LiveState(resolvedPortfolio).catch(() => null),
    api.recentExecutionResults(resolvedPortfolio, 20).catch(() => []),
    api.recentExecutionFills(resolvedPortfolio, 20).catch(() => []),
    api.recentAudit(resolvedPortfolio, 120).catch(() => []),
  ]);

  return (
    <Mt5BlotterLiveSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
      initialLiveState={liveState}
      initialExecutions={executions}
      initialFills={fills}
      initialAudit={audit}
    />
  );
}

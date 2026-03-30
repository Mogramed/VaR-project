import { CapitalLiveSurface } from "@/components/app-shell/capital-live-surface";
import { api } from "@/lib/api/client";

export default async function DeskCapitalPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const health = await api.health();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;

  const [liveState, capital, history] = await Promise.all([
    api.mt5LiveState(resolvedPortfolio).catch(() => null),
    api.latestCapital(resolvedPortfolio).catch(() => null),
    api.capitalHistory(resolvedPortfolio, 18).catch(() => []),
  ]);

  return (
    <CapitalLiveSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
      initialLiveState={liveState}
      initialCapital={capital}
      initialHistory={history}
    />
  );
}

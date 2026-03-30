import { DecisionsLiveSurface } from "@/components/app-shell/decisions-live-surface";
import { api } from "@/lib/api/client";

export default async function DeskDecisionsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const health = await api.health();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;

  const [liveState, decisions] = await Promise.all([
    api.mt5LiveState(resolvedPortfolio).catch(() => null),
    api.recentDecisions(resolvedPortfolio, 12).catch(() => []),
  ]);

  return (
    <DecisionsLiveSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
      initialLiveState={liveState}
      initialDecisions={decisions}
    />
  );
}

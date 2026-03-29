import { Mt5LiveOpsFeed } from "@/components/app-shell/mt5-live-ops-feed";
import { api } from "@/lib/api/client";

export default async function DeskMt5OpsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const health = await api.health();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;
  const liveState = await api.mt5LiveState(resolvedPortfolio);

  return <Mt5LiveOpsFeed initialState={liveState} />;
}

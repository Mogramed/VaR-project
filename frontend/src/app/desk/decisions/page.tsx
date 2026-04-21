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
  const accountId =
    typeof query.account === "string" ? query.account : undefined;
  const health = await api.safeHealth();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;

  const decisions = await api.recentDecisions(resolvedPortfolio, 12, accountId).catch(() => []);

  return (
    <DecisionsLiveSurface
      key={`${resolvedPortfolio}:${accountId ?? "default"}`}
      portfolioSlug={resolvedPortfolio}
      initialDecisions={decisions}
    />
  );
}

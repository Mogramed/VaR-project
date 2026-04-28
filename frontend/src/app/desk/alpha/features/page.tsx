import { AlphaFeaturesSurface } from "@/components/app-shell/alpha-features-surface";
import { api } from "@/lib/api/client";

export default async function DeskAlphaFeaturesPage({
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
  const decisions = await api.recentDecisions(resolvedPortfolio, 40, accountId).catch(() => []);

  return (
    <AlphaFeaturesSurface
      key={`${resolvedPortfolio}:${accountId ?? "default"}`}
      portfolioSlug={resolvedPortfolio}
      initialDecisions={decisions}
    />
  );
}

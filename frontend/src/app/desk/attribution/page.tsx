import { AttributionLiveSurface } from "@/components/app-shell/attribution-live-surface";
import { api } from "@/lib/api/client";

export default async function DeskAttributionPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const preferredModel = typeof query.model === "string" ? query.model : undefined;
  const health = await api.safeHealth();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;

  const [liveState, attribution, comparison] = await Promise.all([
    api.mt5LiveState(resolvedPortfolio).catch(() => null),
    api.latestAttribution(resolvedPortfolio).catch(() => null),
    api.latestModelComparison(resolvedPortfolio).catch(() => null),
  ]);

  return (
    <AttributionLiveSurface
      key={`${resolvedPortfolio}:${preferredModel ?? "auto"}`}
      portfolioSlug={resolvedPortfolio}
      preferredModel={preferredModel}
      initialLiveState={liveState}
      initialAttribution={attribution}
      initialComparison={comparison}
    />
  );
}

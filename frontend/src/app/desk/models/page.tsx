import { ModelsLiveSurface } from "@/components/app-shell/models-live-surface";
import { api } from "@/lib/api/client";

export default async function DeskModelsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const health = await api.safeHealth();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;

  const [liveState, comparison, validation, frame] = await Promise.all([
    api.mt5LiveState(resolvedPortfolio).catch(() => null),
    api.latestModelComparison(resolvedPortfolio).catch(() => null),
    api.latestValidation(resolvedPortfolio).catch(() => null),
    api.latestBacktestFrame(resolvedPortfolio, 240).catch(() => null),
  ]);

  return (
    <ModelsLiveSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
      initialLiveState={liveState}
      initialComparison={comparison}
      initialValidation={validation}
      initialFrame={frame}
    />
  );
}

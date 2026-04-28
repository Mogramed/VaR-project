import { AlphaPerformanceSurface } from "@/components/app-shell/alpha-performance-surface";
import { api } from "@/lib/api/client";

export default async function DeskAlphaPerformancePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const health = await api.safeHealth();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;
  const replay = await api.decisionAlphaReplayWindow(resolvedPortfolio, {
    limit: 600,
    lookbackDays: 90,
  }).catch(() => null);
  const forecastSymbol =
    replay?.predicted_vs_realized?.[replay.predicted_vs_realized.length - 1]?.symbol ?? "EURUSD";
  const forecast = await api.decisionAlphaForecast(forecastSymbol, {
    portfolioSlug: resolvedPortfolio,
    horizonDays: 5,
  }).catch(() => null);
  const projectionForecast = await api.decisionAlphaForecast(forecastSymbol, {
    portfolioSlug: resolvedPortfolio,
    horizonDays: 150,
  }).catch(() => null);
  const trajectory = await api.decisionAlphaTrajectory(forecastSymbol, {
    portfolioSlug: resolvedPortfolio,
    lookbackDays: 90,
  }).catch(() => null);
  const projectionTrajectory = await api.decisionAlphaTrajectory(forecastSymbol, {
    portfolioSlug: resolvedPortfolio,
    lookbackDays: 30,
  }).catch(() => null);
  const portfolioForecast = await api.decisionAlphaPortfolioForecast({
    portfolioSlug: resolvedPortfolio,
    horizonDays: 150,
  }).catch(() => null);

  return (
    <AlphaPerformanceSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
      initialReplay={replay}
      initialForecast={forecast}
      initialTrajectory={trajectory}
      initialProjectionForecast={projectionForecast}
      initialProjectionTrajectory={projectionTrajectory}
      initialPortfolioForecast={portfolioForecast}
    />
  );
}

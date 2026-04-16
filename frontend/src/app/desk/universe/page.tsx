import { UniverseLiveSurface } from "@/components/app-shell/universe-live-surface";
import { api } from "@/lib/api/client";

export default async function DeskUniversePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const resolvedPortfolio = portfolioSlug ?? (await api.safeHealth()).portfolio_slug;

  const [instruments, marketStatus, mt5Status] = await Promise.all([
    api.instruments(resolvedPortfolio).catch(() => []),
    api.marketDataStatus(resolvedPortfolio).catch(() => null),
    api.mt5Status().catch(() => null),
  ]);

  return (
    <UniverseLiveSurface
      portfolioSlug={resolvedPortfolio}
      initialInstruments={instruments}
      initialMarketStatus={marketStatus}
      initialMt5Status={mt5Status}
    />
  );
}

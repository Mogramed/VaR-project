import { StressLiveSurface } from "@/components/app-shell/stress-live-surface";
import { api } from "@/lib/api/client";

export default async function DeskStressPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const resolvedPortfolio = portfolioSlug ?? (await api.safeHealth()).portfolio_slug;

  return (
    <StressLiveSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
    />
  );
}

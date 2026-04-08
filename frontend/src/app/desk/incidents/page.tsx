import { IncidentCenterSurface } from "@/components/app-shell/incident-center-surface";
import { api } from "@/lib/api/client";

export default async function DeskIncidentsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const resolvedPortfolio = portfolioSlug ?? (await api.safeHealth()).portfolio_slug;

  const [incidents, audit] = await Promise.all([
    api.reconciliationIncidents({ portfolioSlug: resolvedPortfolio, includeResolved: true, limit: 200 }).catch(() => []),
    api.recentAudit(resolvedPortfolio, 150).catch(() => []),
  ]);

  return (
    <IncidentCenterSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
      initialIncidents={incidents}
      initialAudit={audit}
    />
  );
}

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

  const [liveState, incidents, audit] = await Promise.all([
    api.mt5LiveState(resolvedPortfolio).catch(() => null),
    api.reconciliationIncidents({ portfolioSlug: resolvedPortfolio, includeResolved: true, limit: 200 }).catch(() => []),
    api.recentAudit(resolvedPortfolio, 150).catch(() => []),
  ]);

  return (
    <IncidentCenterSurface
      key={resolvedPortfolio}
      portfolioSlug={resolvedPortfolio}
      initialLiveState={liveState}
      initialIncidents={incidents}
      initialAudit={audit}
    />
  );
}

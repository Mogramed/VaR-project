import { OverviewLiveDashboard } from "@/components/app-shell/overview-live-dashboard";
import { PageHeader } from "@/components/app-shell/page-header";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { buildAlertSeverityCounts } from "@/lib/view-models";

export default async function DeskOverviewPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;

  const [health, alerts, capital, comparison, snapshot] = await Promise.all([
    api.health(),
    api.recentAlerts(12).catch(() => []),
    api.latestCapital(portfolioSlug).catch(() => null),
    api.latestModelComparison(portfolioSlug).catch(() => null),
    api.latestSnapshot(portfolioSlug).catch(() => null),
  ]);
  const liveState = await api.mt5LiveState(portfolioSlug).catch(() => null);

  const deskSlug = health.desk_slug ?? "main";
  const desk = await api.deskOverview(deskSlug).catch(() => null);
  const payload = (snapshot?.payload ?? {}) as {
    var?: Record<string, number>;
    es?: Record<string, number>;
  };
  const selectedModel =
    comparison?.champion_model ?? capital?.reference_model ?? "hist";
  const varValue = Number(
    payload.var?.[selectedModel] ?? Object.values(payload.var ?? {})[0] ?? 0,
  );
  const esValue = Number(
    payload.es?.[selectedModel] ?? Object.values(payload.es ?? {})[0] ?? 0,
  );
  const alertCounts = buildAlertSeverityCounts(alerts);

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Desk Overview"
        title="Current risk posture across the FX desk."
        description="A denser operator view: portfolio load, model leadership, capital headroom and alert pressure all read in a few seconds."
        aside={capital ? <StatusBadge label={capital.status} tone="accent" /> : null}
      />

      <OverviewLiveDashboard
        deskSlug={deskSlug}
        portfolioSlug={portfolioSlug ?? health.portfolio_slug}
        initialDesk={desk}
        initialLiveState={liveState}
        initialCapital={capital}
        fallbackSelectedModel={selectedModel}
        fallbackVarValue={varValue}
        fallbackEsValue={esValue}
        alertCounts={alertCounts}
        championModel={comparison?.champion_model ?? null}
        snapshotCreatedAt={snapshot?.created_at ?? null}
      />
    </div>
  );
}

import { ReportsLiveSurface } from "@/components/app-shell/reports-live-surface";
import { loadDeskReportViewModel } from "@/lib/report-view-model";

export default async function DeskReportsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const accountId =
    typeof query.account === "string" ? query.account : undefined;

  const reportView = await loadDeskReportViewModel(portfolioSlug, {
    liveState: null,
    accountId,
    freezeToReportScope: false,
  });

  return (
    <ReportsLiveSurface
      key={`${reportView.resolvedPortfolio}:${accountId ?? "default"}`}
      portfolioSlug={reportView.resolvedPortfolio}
      initialView={reportView}
    />
  );
}

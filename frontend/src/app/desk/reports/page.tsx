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

  const reportView = await loadDeskReportViewModel(portfolioSlug);

  return (
    <ReportsLiveSurface
      key={reportView.resolvedPortfolio}
      portfolioSlug={reportView.resolvedPortfolio}
      initialView={reportView}
    />
  );
}

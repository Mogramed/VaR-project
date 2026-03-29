import { ChartSurface } from "@/components/charts/chart-surface";
import { ReportDocument } from "@/components/reports/report-document";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { makeBacktestOption, makeGroupedBarOption, makeLineOption } from "@/lib/chart-options";
import { loadDeskReportViewModel } from "@/lib/report-view-model";
import { formatCurrency, formatTimestamp } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function ReportExportPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const report = await loadDeskReportViewModel(portfolioSlug);

  return (
    <main
      data-report-export-root="true"
      className="pdf-report min-h-screen bg-[#090a0d] px-8 py-8 text-[var(--color-text)]"
    >
      <div className="mx-auto flex max-w-[1120px] flex-col gap-8">
        <section className="surface-strong rounded-[2rem] border border-white/10 px-8 py-10">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div className="max-w-3xl">
              <div className="mono text-[11px] uppercase tracking-[0.3em] text-[var(--color-accent)]">
                FX Risk Desk Platform
              </div>
              <h1 className="mt-5 text-balance text-5xl font-semibold tracking-[-0.06em] text-white">
                Daily FX risk report
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-8 text-[var(--color-text-soft)]">
                Portfolio {report.resolvedPortfolio}. This downloadable report packages
                the current risk posture, capital usage, governance flow and the latest
                written desk narrative into a standalone PDF.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <StatusBadge label={report.selectedModel.toUpperCase()} tone="accent" />
                <StatusBadge label={report.capital?.status ?? "pending"} tone="success" />
                <StatusBadge label={report.meta.reportTimestamp} />
              </div>
            </div>
            <div className="grid w-full gap-4 sm:max-w-[460px] sm:grid-cols-2">
              {report.executiveSummary.map((item) => (
                <MetricBlock
                  key={item.label}
                  label={item.label}
                  value={item.value}
                  hint={item.copy}
                  tone={item.tone}
                />
              ))}
            </div>
          </div>
        </section>

        <section className="pdf-avoid-break grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="surface rounded-[1.8rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Executive summary
            </div>
            <div className="mt-5 space-y-4">
              {report.narrativeSummary.map((item) => (
                <div
                  key={item}
                  className="rounded-[1.2rem] border border-white/8 bg-black/18 px-4 py-4 text-sm leading-7 text-[var(--color-text-soft)]"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>

          <div className="surface rounded-[1.8rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Report metadata
            </div>
            <div className="mt-5 grid gap-4 text-sm text-[var(--color-text-soft)]">
              <ReportMetaRow label="Portfolio" value={report.resolvedPortfolio} />
              <ReportMetaRow label="Report generated" value={report.meta.reportTimestamp} />
              <ReportMetaRow label="Source markdown" value={report.meta.reportPath || "n/a"} mono />
              <ReportMetaRow label="Charts embedded" value={String(report.meta.chartCount)} />
            </div>
          </div>
        </section>

        <section className="pdf-avoid-break grid gap-6 lg:grid-cols-2">
          <ChartSurface
            option={makeBacktestOption(report.derived.backtestSeries)}
            height={420}
            mode="trace"
            dataCount={report.derived.backtestSeries.length}
            eyebrow="Analytics"
            title="Backtest trace"
            description="Portfolio PnL against the current VaR stack."
            meta={`${report.derived.backtestSeries.length} rows`}
          />
          <ChartSurface
            option={makeLineOption(report.capitalSeries, "#5fd4a6", { mode: "standard" })}
            height={420}
            mode="standard"
            dataCount={report.capitalSeries.length}
            eyebrow="Capital"
            title="Capital usage trajectory"
            description="Consumed capital over the latest persisted snapshots."
            meta={report.capital?.reference_model.toUpperCase() ?? "n/a"}
          />
        </section>

        <ChartSurface
          option={makeGroupedBarOption(
            report.decisionSizeSeries.labels,
            [
              {
                name: "Requested",
                data: report.decisionSizeSeries.requested,
                color: "#d89b49",
              },
              {
                name: "Approved",
                data: report.decisionSizeSeries.approved,
                color: "#5fd4a6",
              },
            ],
            { mode: "comparison" },
          )}
          height={360}
          mode="comparison"
          dataCount={report.decisionSizeSeries.labels.length}
          eyebrow="Decision continuity"
          title="Requested versus approved notional"
          description="Advisory friction remains visible inside the downloadable report."
        />

        {report.report ? (
          <ReportDocument
            content={report.normalizedReportContent}
            reportPath={report.report.report_markdown}
            showSource={false}
          />
        ) : (
          <section className="surface rounded-[1.8rem] p-6 text-sm text-[var(--color-text-muted)]">
            No persisted report narrative is available yet.
          </section>
        )}

        <section className="pdf-avoid-break grid gap-6 lg:grid-cols-2">
          <StaticTableCard
            title="Recent decisions"
            headers={["Time", "Symbol", "Decision", "Approved"]}
            rows={report.decisions.slice(0, 8).map((item) => [
              formatTimestamp(item.created_at ?? item.time_utc),
              item.symbol,
              item.decision,
              formatCurrency(item.approved_delta_position_eur),
            ])}
          />
          <StaticTableCard
            title="Audit trail"
            headers={["Time", "Actor", "Action", "Object"]}
            rows={report.audit.slice(0, 8).map((item) => [
              formatTimestamp(item.created_at),
              item.actor,
              item.action_type,
              item.object_type ?? "n/a",
            ])}
          />
        </section>
      </div>
    </main>
  );
}

function ReportMetaRow({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-white/8 pb-4 last:border-b-0 last:pb-0">
      <span>{label}</span>
      <span className={mono ? "mono break-all text-right text-white" : "text-right text-white"}>
        {value}
      </span>
    </div>
  );
}

function StaticTableCard({
  title,
  headers,
  rows,
}: {
  title: string;
  headers: string[];
  rows: string[][];
}) {
  return (
    <section className="surface rounded-[1.8rem] p-6">
      <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
        {title}
      </div>
      <div className="mt-5 overflow-hidden rounded-[1.3rem] border border-white/8">
        <table className="min-w-full border-collapse">
          <thead className="bg-white/[0.03]">
            <tr>
              {headers.map((header) => (
                <th
                  key={header}
                  className="mono border-b border-white/8 px-4 py-3 text-left text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={`${title}-${rowIndex}`} className="border-b border-white/6 last:border-b-0">
                {row.map((cell, cellIndex) => (
                  <td
                    key={`${title}-${rowIndex}-${cellIndex}`}
                    className="px-4 py-3 text-sm text-[var(--color-text-soft)]"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

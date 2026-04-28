import { AutoPrintReport } from "@/components/reports/auto-print-report";
import {
  ReportBacktestChart,
  ReportCapitalChart,
  ReportDecisionContinuityChart,
} from "@/components/reports/report-export-charts";
import { ReportDocument } from "@/components/reports/report-document";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import {
  formatReportStatusLabel,
  getReportExportSection,
  REPORT_EXPORT_SECTIONS,
  reportStatusTone,
} from "@/lib/report-export";
import { loadDeskReportViewModel } from "@/lib/report-view-model";
import { formatCurrency, formatSourceLabel, formatTimestamp } from "@/lib/utils";

export const dynamic = "force-dynamic";

const REPORT_DATE_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  day: "2-digit",
  month: "long",
  year: "numeric",
});

export default async function ReportExportPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const accountId =
    typeof query.account === "string" ? query.account : undefined;
  const autoPrint = query.print === "1";
  const report = await loadDeskReportViewModel(portfolioSlug, {
    liveState: null,
    accountId,
    freezeToReportScope: true,
  });

  const snapshotSourceLabel = formatSourceLabel(report.meta.preferredSnapshotSource);
  const isLiveSnapshot = String(report.meta.preferredSnapshotSource ?? "").startsWith("mt5_live");
  const reportDate = REPORT_DATE_FORMATTER.format(new Date());
  const executiveSection = getReportExportSection("executive");
  const analyticsSection = getReportExportSection("analytics");
  const capitalSection = getReportExportSection("capital");
  const governanceSection = getReportExportSection("governance");
  const narrativeSection = getReportExportSection("narrative");
  const auditSection = getReportExportSection("audit");
  const selectedCapitalModel = report.capital?.reference_model?.toUpperCase() ?? report.selectedModel.toUpperCase();

  return (
    <main
      data-report-export-root="true"
      className="pdf-report min-h-screen bg-[#090a0d] px-8 py-8 text-[var(--color-text)]"
    >
      <AutoPrintReport enabled={autoPrint} />
      <div className="mx-auto flex max-w-[1120px] flex-col gap-8">
        <section className="pdf-page-break-after surface-strong rounded-[2rem] border border-white/10 px-10 py-12">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div className="max-w-3xl">
              <div className="h-[2px] w-16 rounded-full bg-[var(--color-accent)]" />
              <div className="mt-5 mono text-[11px] uppercase tracking-[0.3em] text-[var(--color-accent)]">
                FX Risk Desk Platform
              </div>
              <h1 className="mt-5 text-balance text-5xl font-semibold tracking-[-0.06em] text-white">
                Daily FX Risk Report
              </h1>
              <p className="mt-2 text-lg font-medium text-white/70">
                {reportDate}
              </p>
              <p className="mt-4 max-w-2xl text-base leading-8 text-[var(--color-text-soft)]">
                Portfolio <span className="font-semibold text-white">{report.resolvedPortfolio}</span>.
                This report packages the current risk posture, capital usage, governance
                flow and the latest written desk narrative into a standalone document.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <StatusBadge label={report.selectedModel.toUpperCase()} tone="accent" />
                <StatusBadge
                  label={snapshotSourceLabel}
                  tone={isLiveSnapshot ? "success" : "neutral"}
                />
                <StatusBadge
                  label={formatReportStatusLabel(report.capital?.status)}
                  tone={reportStatusTone(report.capital?.status)}
                />
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

          <div className="mt-10 flex items-center gap-3 border-t border-white/8 pt-5">
            <span className="mono text-[9px] uppercase tracking-[0.25em] text-[var(--color-text-muted)]">
              Confidential
            </span>
            <span className="h-px flex-1 bg-white/6" />
            <span className="text-[10px] text-[var(--color-text-muted)]">
              Internal use only - Do not distribute
            </span>
          </div>
        </section>

        <section className="pdf-avoid-break surface rounded-[1.8rem] p-6">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Contents
          </div>
          <div className="mt-4 grid gap-1.5 text-sm">
            {REPORT_EXPORT_SECTIONS.map((section) => (
              <div key={section.number} className="flex items-center gap-3 py-1.5">
                <span className="mono w-6 text-[12px] font-bold text-[var(--color-accent)]">{section.number}</span>
                <span className="text-[var(--color-text-soft)]">{section.title}</span>
                <span className="h-px flex-1 border-b border-dotted border-white/10" />
              </div>
            ))}
            {report.headings.filter((h) => h.level <= 2).slice(0, 8).map((h) => (
              <div key={h.id} className="flex items-center gap-3 py-1 pl-9">
                <span className="text-[12px] text-[var(--color-text-muted)]">{h.text}</span>
                <span className="h-px flex-1 border-b border-dotted border-white/8" />
              </div>
            ))}
          </div>
        </section>

        <SectionDivider number={executiveSection.number} title={executiveSection.title} />
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
              <ReportMetaRow label="Snapshot source" value={snapshotSourceLabel} />
              <ReportMetaRow label="Source markdown" value={report.meta.reportPath || "n/a"} mono />
              <ReportMetaRow label="Charts embedded" value={String(report.meta.chartCount)} />
            </div>
          </div>
        </section>

        <SectionDivider number={analyticsSection.number} title={analyticsSection.title} />
        <section className="pdf-avoid-break grid gap-6 lg:grid-cols-2">
          <ReportBacktestChart
            points={report.derived.backtestSeries}
            height={420}
            title="Backtest trace"
            description="Portfolio PnL against the current VaR stack."
            meta={`${report.derived.backtestSeries.length} rows`}
          />
          <section className="surface rounded-[1.8rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Risk analytics snapshot
            </div>
            <div className="mt-5 grid gap-4 sm:grid-cols-2">
              <MetricBlock
                label="Selected model"
                value={report.selectedModel.toUpperCase()}
                hint="Model frozen into the versioned report contract."
                tone="accent"
              />
              <MetricBlock
                label="Validation champion"
                value={(report.comparison?.champion_model ?? report.selectedModel).toUpperCase()}
                hint="Champion/challenger state captured at report generation time."
                tone="neutral"
              />
              <MetricBlock
                label="VaR"
                value={report.varDisplay}
                hint={report.report?.report_contract?.metrics?.var?.as_of_utc
                  ? `As of ${formatTimestamp(report.report.report_contract.metrics.var.as_of_utc)}.`
                  : "Latest report-scoped VaR value."}
                tone="accent"
              />
              <MetricBlock
                label="Expected shortfall"
                value={report.esDisplay}
                hint={report.report?.report_contract?.metrics?.es?.as_of_utc
                  ? `As of ${formatTimestamp(report.report.report_contract.metrics.es.as_of_utc)}.`
                  : "Latest report-scoped ES value."}
                tone="warning"
              />
              <MetricBlock
                label="Last PnL"
                value={report.pnlDisplay}
                hint={report.pnlTimestamp
                  ? `Frozen at ${formatTimestamp(report.pnlTimestamp)} for this report export.`
                  : "No report-scoped PnL timestamp available."}
                tone="neutral"
              />
              <MetricBlock
                label="Contract version"
                value={report.meta.reportContractVersion ?? "report.v1"}
                hint="Versioned report contract used by the export surface."
                tone="success"
              />
            </div>
          </section>
        </section>

        <SectionDivider number={capitalSection.number} title={capitalSection.title} />
        <section className="pdf-avoid-break grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <ReportCapitalChart
            points={report.capitalSeries}
            height={420}
            title="Capital usage trajectory"
            description="Consumed capital over the latest persisted snapshots."
            meta={selectedCapitalModel}
          />
          <section className="surface rounded-[1.8rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Capital posture
            </div>
            <div className="mt-5 grid gap-4 sm:grid-cols-2">
              <MetricBlock
                label="Budget"
                value={report.capital ? formatCurrency(report.capital.total_capital_budget_eur, report.meta.moneyDecimals) : "n/a"}
                hint="Total capital envelope persisted with the report snapshot."
                tone="accent"
              />
              <MetricBlock
                label="Consumed"
                value={report.capital ? formatCurrency(report.capital.total_capital_consumed_eur, report.meta.moneyDecimals) : "n/a"}
                hint="Capital currently consumed by active exposures."
                tone="warning"
              />
              <MetricBlock
                label="Reserved"
                value={report.capital ? formatCurrency(report.capital.total_capital_reserved_eur, report.meta.moneyDecimals) : "n/a"}
                hint="Capital reserve carved out by current desk policy."
                tone="neutral"
              />
              <MetricBlock
                label="Headroom"
                value={report.capital ? `${Math.round((report.capital.headroom_ratio ?? 0) * 100)}%` : "n/a"}
                hint={report.capital
                  ? `${formatCurrency(report.capital.total_capital_remaining_eur, report.meta.moneyDecimals)} remaining before the configured boundary.`
                  : "No report-scoped capital snapshot available."}
                tone={reportStatusTone(report.capital?.status)}
              />
            </div>
            <div className="mt-6 grid gap-4 text-sm text-[var(--color-text-soft)]">
              <ReportMetaRow label="Status" value={formatReportStatusLabel(report.capital?.status)} />
              <ReportMetaRow label="Reference model" value={selectedCapitalModel} />
              <ReportMetaRow label="Base currency" value={report.meta.baseCurrency} />
              <ReportMetaRow label="Snapshot source" value={snapshotSourceLabel} />
            </div>
          </section>
        </section>

        <SectionDivider number={governanceSection.number} title={governanceSection.title} />
        <ReportDecisionContinuityChart
          labels={report.decisionSizeSeries.labels}
          requested={report.decisionSizeSeries.requested}
          approved={report.decisionSizeSeries.approved}
          height={360}
          title="Requested versus approved exposure"
          description="Advisory friction remains visible inside the downloadable report."
        />

        <SectionDivider number={narrativeSection.number} title={narrativeSection.title} />
        {report.report ? (
          <ReportDocument
            content={report.normalizedReportContent}
            reportPath={report.report.report_markdown}
            showSource={false}
            portfolioSlug={report.resolvedPortfolio}
            reportId={report.report.report_id}
            chartPaths={report.report.chart_paths ?? []}
          />
        ) : (
          <section className="surface rounded-[1.8rem] p-6 text-sm text-[var(--color-text-muted)]">
            No persisted report narrative is available yet.
          </section>
        )}

        <SectionDivider number={auditSection.number} title={auditSection.title} />
        <section className="pdf-avoid-break grid gap-6 lg:grid-cols-2">
          <StaticTableCard
            title="Recent decisions"
            headers={["Time", "Symbol", "Decision", "Approved exposure"]}
            rows={report.decisions.slice(0, 8).map((item) => [
              formatTimestamp(item.created_at ?? item.time_utc),
              item.symbol,
              item.decision,
              formatCurrency(item.approved_exposure_change),
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

function SectionDivider({ number, title }: { number: string; title: string }) {
  return (
    <div className="pdf-avoid-break flex items-center gap-4 pb-1 pt-4">
      <span className="mono text-[14px] font-bold text-[var(--color-accent)]">{number}</span>
      <div className="h-[1px] flex-1 bg-gradient-to-r from-[var(--color-accent)]/30 to-transparent" />
      <span className="text-[11px] uppercase tracking-[0.2em] text-[var(--color-text-muted)]">{title}</span>
    </div>
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

function isNumericCell(cell: string): boolean {
  return /^[\dA-Za-z$%,.\-+\s]+$/.test(cell.trim());
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
      <div className="mt-5 overflow-hidden rounded-[1.3rem] border border-white/8 border-l-[2px] border-l-[var(--color-accent)]/30">
        <table className="min-w-full border-collapse">
          <thead className="bg-white/[0.04]">
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
              <tr
                key={`${title}-${rowIndex}`}
                className={`border-b border-white/6 last:border-b-0 ${rowIndex % 2 === 1 ? "bg-white/[0.02]" : ""}`}
              >
                {row.map((cell, cellIndex) => (
                  <td
                    key={`${title}-${rowIndex}-${cellIndex}`}
                    className={`px-4 py-3 text-sm text-[var(--color-text-soft)] ${
                      isNumericCell(cell) ? "mono text-right font-medium" : ""
                    }`}
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

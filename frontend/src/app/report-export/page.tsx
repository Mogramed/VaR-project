import { ChartSurface } from "@/components/charts/chart-surface";
import { ReportDocument } from "@/components/reports/report-document";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { CHART_PALETTE, makeBacktestOption, makeGroupedBarOption, makeLineOption } from "@/lib/chart-options";
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
  const snapshotSourceLabel =
    report.meta.preferredSnapshotSource === "mt5_live_bridge"
      ? "mt5 live snapshot"
      : "historical snapshot";
  const reportDate = new Date().toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "long",
    year: "numeric",
  });

  return (
    <main
      data-report-export-root="true"
      className="pdf-report min-h-screen bg-[#090a0d] px-8 py-8 text-[var(--color-text)]"
    >
      <div className="mx-auto flex max-w-[1120px] flex-col gap-8">
        {/* ─── Cover Page ─── */}
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
                  tone={report.meta.preferredSnapshotSource === "mt5_live_bridge" ? "success" : "neutral"}
                />
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

          {/* Confidential notice */}
          <div className="mt-10 flex items-center gap-3 border-t border-white/8 pt-5">
            <span className="mono text-[9px] uppercase tracking-[0.25em] text-[var(--color-text-muted)]">
              Confidential
            </span>
            <span className="h-px flex-1 bg-white/6" />
            <span className="text-[10px] text-[var(--color-text-muted)]">
              Internal use only — Do not distribute
            </span>
          </div>
        </section>

        {/* ─── Table of Contents ─── */}
        <section className="pdf-avoid-break surface rounded-[1.8rem] p-6">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Contents
          </div>
          <div className="mt-4 grid gap-1.5 text-sm">
            {[
              { num: "01", title: "Executive Summary" },
              { num: "02", title: "Analytics" },
              { num: "03", title: "Capital" },
              { num: "04", title: "Governance" },
              { num: "05", title: "Desk Narrative" },
              { num: "06", title: "Audit Trail" },
            ].map((s) => (
              <div key={s.num} className="flex items-center gap-3 py-1.5">
                <span className="mono w-6 text-[12px] font-bold text-[var(--color-accent)]">{s.num}</span>
                <span className="text-[var(--color-text-soft)]">{s.title}</span>
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

        {/* ─── 01 Executive Summary ─── */}
        <SectionDivider number="01" title="Executive Summary" />
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

        {/* ─── 02 Analytics ─── */}
        <SectionDivider number="02" title="Analytics" />
        <section className="pdf-avoid-break grid gap-6 lg:grid-cols-2">
          <ChartSurface
            option={makeBacktestOption(report.derived.backtestSeries)}
            height={420}
            mode="trace"
            dataCount={report.derived.backtestSeries.length}
            eyebrow="Analytics"
            title="Backtest trace"
            description="Portfolio PnL against the current VaR stack."
            showDescription
            meta={`${report.derived.backtestSeries.length} rows`}
          />
          <ChartSurface
            option={makeLineOption(report.capitalSeries, CHART_PALETTE.green, { mode: "standard" })}
            height={420}
            mode="standard"
            dataCount={report.capitalSeries.length}
            eyebrow="Capital"
            title="Capital usage trajectory"
            description="Consumed capital over the latest persisted snapshots."
            showDescription
            meta={report.capital?.reference_model.toUpperCase() ?? "n/a"}
          />
        </section>

        {/* ─── 03 Capital ─── */}
        <SectionDivider number="03" title="Governance" />
        <ChartSurface
          option={makeGroupedBarOption(
            report.decisionSizeSeries.labels,
            [
              {
                name: "Requested",
                data: report.decisionSizeSeries.requested,
                color: CHART_PALETTE.gold,
              },
              {
                name: "Approved",
                data: report.decisionSizeSeries.approved,
                color: CHART_PALETTE.green,
              },
            ],
            { mode: "comparison" },
          )}
          height={360}
          mode="comparison"
          dataCount={report.decisionSizeSeries.labels.length}
          eyebrow="Decision continuity"
          title="Requested versus approved exposure"
          description="Advisory friction remains visible inside the downloadable report."
          showDescription
        />

        {/* ─── 04 Desk Narrative ─── */}
        <SectionDivider number="04" title="Desk Narrative" />
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

        {/* ─── 05 Audit Trail ─── */}
        <SectionDivider number="05" title="Audit Trail" />
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

/* ─── Local components ─── */

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
  return /^[\d€$£¥%,.\-+\s]+$/.test(cell.trim());
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

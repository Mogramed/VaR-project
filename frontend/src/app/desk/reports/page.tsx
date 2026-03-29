import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { AuditTrailTable, DecisionHistoryTable } from "@/components/data/risk-tables";
import { ReportActions } from "@/components/reports/report-actions";
import {
  ReportDocument,
  ReportTableOfContents,
} from "@/components/reports/report-document";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import {
  makeBacktestOption,
  makeGroupedBarOption,
  makeLineOption,
} from "@/lib/chart-options";
import { loadDeskReportViewModel } from "@/lib/report-view-model";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

function statusTone(status: string | null | undefined) {
  const normalized = (status ?? "").toLowerCase();
  if (normalized.includes("breach") || normalized.includes("critical")) {
    return "danger" as const;
  }
  if (normalized.includes("warn") || normalized.includes("hold")) {
    return "warning" as const;
  }
  if (normalized.includes("ok") || normalized.includes("stable")) {
    return "success" as const;
  }
  return "neutral" as const;
}

export default async function DeskReportsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;

  const reportView = await loadDeskReportViewModel(portfolioSlug);
  const topRecommendation = reportView.capital?.recommendations?.[0];

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Reports & Governance"
        title="A real desk report, readable in the platform and exportable as a branded PDF."
        description="Narrative, analytics, decision continuity and audit history now live in one editorial surface instead of a stack of disconnected widgets."
        aside={
          <div className="flex flex-col items-start gap-3 lg:items-end">
            <StatusBadge label={reportView.resolvedPortfolio} tone="accent" />
            <ReportActions portfolioSlug={reportView.resolvedPortfolio} />
          </div>
        }
      />

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_320px]">
        <div className="surface-strong rounded-[2rem] border border-white/10 px-6 py-7 md:px-8">
          <div className="flex flex-wrap items-start justify-between gap-5">
            <div className="max-w-3xl">
              <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-accent)]">
                Daily desk report
              </div>
              <h2 className="mt-4 max-w-3xl text-balance text-4xl font-semibold tracking-[-0.05em] text-white md:text-[3.35rem]">
                FX VaR posture, capital pressure and governance continuity.
              </h2>
              <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--color-text-soft)] md:text-base">
                Generated for {reportView.resolvedPortfolio}. The report now reads like a
                desk note: model lead, capital headroom, decision friction and audit
                continuity are all on the same canvas.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <StatusBadge label={reportView.selectedModel.toUpperCase()} tone="accent" />
              <StatusBadge
                label={reportView.capital?.status ?? "snapshot pending"}
                tone={statusTone(reportView.capital?.status)}
              />
              <StatusBadge label={reportView.meta.reportTimestamp} />
            </div>
          </div>

          <div className="mt-7 grid gap-4 lg:grid-cols-2 xl:grid-cols-4">
            {reportView.executiveSummary.map((item) => (
              <MetricBlock
                key={item.label}
                label={item.label}
                value={item.value}
                hint={item.copy}
                tone={item.tone}
                className="h-full"
              />
            ))}
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)]">
            <div className="rounded-[1.6rem] border border-white/8 bg-black/16 px-5 py-5">
              <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
                Executive reading
              </div>
              <div className="mt-4 space-y-3">
                {reportView.narrativeSummary.map((item) => (
                  <div
                    key={item}
                    className="rounded-[1.2rem] border border-white/7 bg-white/[0.02] px-4 py-4 text-sm leading-7 text-[var(--color-text-soft)]"
                  >
                    {item}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[1.6rem] border border-white/8 bg-black/16 px-5 py-5">
              <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
                Report highlights
              </div>
              <div className="mt-4 space-y-4 text-sm text-[var(--color-text-soft)]">
                <ReportHighlight
                  label="Generated"
                  value={reportView.meta.reportTimestamp}
                />
                <ReportHighlight
                  label="Charts embedded"
                  value={String(reportView.meta.chartCount)}
                />
                <ReportHighlight
                  label="Backtest rows"
                  value={String(reportView.derived.backtestSeries.length)}
                />
                <ReportHighlight
                  label="Decision history"
                  value={`${reportView.decisions.length} persisted items`}
                />
                {topRecommendation ? (
                  <div className="rounded-[1.2rem] border border-[var(--color-border-strong)] bg-[var(--color-accent-soft)] px-4 py-4 text-sm leading-6 text-[var(--color-text)]">
                    <div className="mono text-[11px] uppercase tracking-[0.22em] text-[var(--color-accent)]">
                      Top rebalance signal
                    </div>
                    <div className="mt-2 font-semibold text-white">
                      Shift {formatCurrency(topRecommendation.amount_eur)} from{" "}
                      {topRecommendation.symbol_from} to {topRecommendation.symbol_to}.
                    </div>
                    <div className="mt-2 text-[var(--color-text-soft)]">
                      {topRecommendation.reason}
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>

        <ReportTableOfContents
          headings={reportView.headings}
          reportPath={reportView.meta.reportPath || "No report path"}
          chartCount={reportView.meta.chartCount}
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.08fr)_minmax(320px,0.92fr)]">
        <ChartSurface
          option={makeBacktestOption(reportView.derived.backtestSeries)}
          mode="trace"
          dataCount={reportView.derived.backtestSeries.length}
          eyebrow="Analytics"
          title="Backtest trace"
          description="PnL and VaR remain a primary reading surface here, not a tiny appendix chart."
          meta={
            reportView.derived.backtestSeries.length
              ? `${reportView.derived.backtestSeries.length} rows`
              : "No backtest frame"
          }
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No backtest trace is available yet.
            </div>
          }
        />

        <ChartSurface
          option={makeLineOption(reportView.capitalSeries, "#5fd4a6", { mode: "standard" })}
          mode="standard"
          dataCount={reportView.capitalSeries.length}
          eyebrow="Capital"
          title="Capital usage trajectory"
          description="Budget consumption is shown alongside the report so risk posture and capital posture stay visually linked."
          meta={
            reportView.capital?.reference_model
              ? reportView.capital.reference_model.toUpperCase()
              : "No capital snapshot"
          }
          footer={
            <div className="space-y-3 text-sm text-[var(--color-text-soft)]">
              <div className="flex items-center justify-between gap-3">
                <span>Remaining capital</span>
                <span className="mono text-white">
                  {reportView.capital
                    ? formatCurrency(reportView.capital.total_capital_remaining_eur)
                    : "n/a"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span>Headroom ratio</span>
                <span className="mono text-white">
                  {reportView.capital
                    ? formatPercent(reportView.capital.headroom_ratio ?? 0, 0)
                    : "n/a"}
                </span>
              </div>
              <div className="text-[var(--color-text-muted)]">
                {topRecommendation
                  ? `Priority move: ${topRecommendation.symbol_from} -> ${topRecommendation.symbol_to}.`
                  : "No active reallocation recommendation is persisted."}
              </div>
            </div>
          }
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No capital history is available yet.
            </div>
          }
        />
      </section>

      <ChartSurface
        option={makeGroupedBarOption(
          reportView.decisionSizeSeries.labels,
          [
            {
              name: "Requested",
              data: reportView.decisionSizeSeries.requested,
              color: "#d89b49",
            },
            {
              name: "Approved",
              data: reportView.decisionSizeSeries.approved,
              color: "#5fd4a6",
            },
          ],
          { mode: "comparison" },
        )}
        mode="comparison"
        dataCount={reportView.decisionSizeSeries.labels.length}
        eyebrow="Decision continuity"
        title="Requested versus approved notional"
        description="Sizing friction, reductions and approvals remain readable in the report itself instead of being buried in history tables."
        meta={`${reportView.decisions.length} recent decisions`}
        footer={
          <div className="grid gap-4 lg:grid-cols-3">
            <MetricBlock
              label="Average fill"
              value={
                reportView.fillRatio == null
                  ? "n/a"
                  : formatPercent(reportView.fillRatio, 0)
              }
              hint="Approved vs requested"
              className="bg-transparent"
            />
            <MetricBlock
              label="Recent decisions"
              value={String(reportView.decisions.length)}
              hint="Persisted advisory outcomes"
              className="bg-transparent"
            />
            <MetricBlock
              label="Audit events"
              value={String(reportView.audit.length)}
              hint="Operator continuity"
              className="bg-transparent"
            />
          </div>
        }
        emptyState={
          <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
            No decision history is available yet.
          </div>
        }
      />

      {reportView.report ? (
        <ReportDocument
          content={reportView.normalizedReportContent}
          reportPath={reportView.report.report_markdown}
        />
      ) : (
        <div className="surface-strong rounded-[2rem] border border-white/10 px-8 py-10 text-sm leading-7 text-[var(--color-text-muted)]">
          No report content is available yet. Generate a fresh report to populate the
          written desk narrative and unlock the PDF export.
        </div>
      )}

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Decision appendix
            </div>
            <div className="text-xs text-[var(--color-text-muted)]">
              Last update{" "}
              {reportView.latestReportEvent?.created_at
                ? formatTimestamp(reportView.latestReportEvent.created_at)
                : "n/a"}
            </div>
          </div>
          <DecisionHistoryTable rows={reportView.decisions} />
        </div>

        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Audit appendix
          </div>
          <AuditTrailTable rows={reportView.audit} />
        </div>
      </section>
    </div>
  );
}

function ReportHighlight({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/8 pb-4 last:border-b-0 last:pb-0">
      <span>{label}</span>
      <span className="mono text-right text-white">{value}</span>
    </div>
  );
}

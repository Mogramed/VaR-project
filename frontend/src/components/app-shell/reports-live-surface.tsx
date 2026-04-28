"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";

import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { LivePostureBanner } from "@/components/app-shell/live-posture-banner";
import { LiveRuntimeBadgeGroup } from "@/components/app-shell/live-runtime-badge-group";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { AuditTrailTable, DecisionHistoryTable } from "@/components/data/risk-tables";
import { ReportActions } from "@/components/reports/report-actions";
import {
  ReportDocument,
  ReportTableOfContents,
} from "@/components/reports/report-document";
import {
  deskArtifactQueryKey,
  deskArtifactQueryOptions,
} from "@/components/app-shell/desk-artifact-query";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import {
  CHART_PALETTE,
  makeBacktestOption,
  makeGroupedBarOption,
  makeLineOption,
} from "@/lib/chart-options";
import {
  type DeskReportViewModel,
  loadDeskReportViewModel,
} from "@/lib/report-view-model";
import { formatCurrency, formatPercent, formatSourceLabel, formatTimestamp, humanizeIdentifier } from "@/lib/utils";

function snapshotLabel(src: string | null | undefined) {
  return formatSourceLabel(src ?? "auto");
}

function verdictTone(verdict: string | null | undefined) {
  const normalized = String(verdict || "").toUpperCase();
  if (normalized === "FAIL") return "danger" as const;
  if (normalized === "WARN") return "warning" as const;
  if (normalized === "PASS") return "success" as const;
  return "neutral" as const;
}

function confidenceTone(level: string | null | undefined) {
  const normalized = String(level || "").toUpperCase();
  if (normalized === "LOW") return "danger" as const;
  if (normalized === "MEDIUM") return "warning" as const;
  if (normalized === "HIGH") return "success" as const;
  return "neutral" as const;
}

function trafficLightTone(value: string | null | undefined) {
  const normalized = String(value || "").toUpperCase();
  if (normalized === "RED") return "danger" as const;
  if (normalized === "YELLOW") return "warning" as const;
  if (normalized === "GREEN") return "success" as const;
  return "neutral" as const;
}

function esShortfallTone(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "neutral" as const;
  const gap = Math.abs(value - 1);
  if (gap <= 0.15) return "success" as const;
  if (gap <= 0.30) return "warning" as const;
  return "danger" as const;
}

function esBreachTone(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "neutral" as const;
  if (value <= 0.15) return "success" as const;
  if (value <= 0.25) return "warning" as const;
  return "danger" as const;
}

const REPORTS_NOISE_ALERT_CODES = new Set([
  "MT5_RECONCILIATION_INCOMPLETE",
  "MT5_RECONCILIATION_WINDOW_EXPIRED",
  "MT5_MANUAL_EVENTS",
  "EXECUTION_UNMATCHED",
  "PENDING_BROKER_ACTIVITY",
  "PARTIAL_FILL_ACTIVE",
  "PNL_DRIFT",
]);

function includeReportsAlert(code: string | null | undefined): boolean {
  const normalized = String(code ?? "").toUpperCase();
  if (!normalized) {
    return false;
  }
  if (normalized.startsWith("VALIDATION_")) {
    return false;
  }
  return !REPORTS_NOISE_ALERT_CODES.has(normalized);
}

export function ReportsLiveSurface({
  portfolioSlug,
  initialView,
}: {
  portfolioSlug: string;
  initialView: DeskReportViewModel;
}) {
  const { liveState, transport, accountId } = useDeskLive();
  const queryClient = useQueryClient();
  const reportViewQueryKey = deskArtifactQueryKey("reports", portfolioSlug, accountId ?? "default");
  const reportViewQuery = useQuery({
    queryKey: reportViewQueryKey,
    queryFn: () => loadDeskReportViewModel(portfolioSlug, {
      liveState: liveState ?? null,
      accountId,
      freezeToReportScope: false,
    }),
    initialData: initialView,
    ...deskArtifactQueryOptions,
  });
  const view = reportViewQuery.data ?? initialView;

  const resolved = liveState ?? view.liveState;
  const decisions = view.decisions;
  const audit = view.audit;
  const reportCapital = view.capital;
  const selectedModel = view.selectedModel;
  const varValue = Number(view.varValue);
  const esValue = Number(view.esValue);
  const pnlValue = Number(view.pnlValue);
  const fillRatio = view.fillRatio;
  const capitalSeries = view.capitalSeries;
  const decisionSeries = view.decisionSizeSeries;
  const resolvedSource = view.meta.preferredSnapshotSource ?? resolved?.risk_summary?.source;
  const label = snapshotLabel(resolvedSource);
  const reportLiveAlerts = (resolved?.operator_alerts ?? []).filter((alert) => includeReportsAlert(alert.code));

  return (
    <div className="desk-page space-y-4">
      <PageHeader
        eyebrow="Reports"
        title="Desk report, analytics and PDF export"
        aside={(
          <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-1.5">
              <StatusBadge label={humanizeIdentifier(view.resolvedPortfolio)} tone="accent" />
              <StatusBadge
                label={label}
                tone={String(resolvedSource ?? "").startsWith("mt5_live") ? "success" : "neutral"}
              />
              <LiveRuntimeBadgeGroup
                liveState={liveState}
                transport={transport}
                showBridge={false}
              />
            </div>
            <ReportActions
              portfolioSlug={view.resolvedPortfolio}
              onGenerated={async () => {
                await queryClient.invalidateQueries({ queryKey: reportViewQueryKey });
              }}
            />
          </div>
        )}
      />
      <LivePostureBanner liveState={liveState} transport={transport} />

      <LiveOperatorAlerts alerts={reportLiveAlerts} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricBlock
          label={`VaR / ${selectedModel.toUpperCase()}`}
          value={view.varDisplay ?? formatCurrency(varValue, view.meta.moneyDecimals)}
          tone="accent"
        />
        <MetricBlock
          label={`ES / ${selectedModel.toUpperCase()}`}
          value={view.esDisplay ?? formatCurrency(esValue, view.meta.moneyDecimals)}
          tone="warning"
        />
        <MetricBlock
          label="Latest PnL"
          value={view.pnlDisplay ?? formatCurrency(pnlValue, view.meta.moneyDecimals)}
          hint={view.pnlTimestamp ? `As of ${formatTimestamp(view.pnlTimestamp)}` : "Latest report dataset point"}
          tone={pnlValue < 0 ? "warning" : "neutral"}
        />
        <MetricBlock
          label="Headroom"
          value={reportCapital ? formatPercent(reportCapital.headroom_ratio ?? 0, 0) : "n/a"}
          tone="success"
        />
        <MetricBlock
          label="Avg fill"
          value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)}
        />
      </section>

      <section className="rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 p-4 shadow-[var(--shadow-soft)]">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Model Validation
            </h4>
            <p className="mt-1 text-[12px] text-[var(--color-text-soft)]">
              Academic checks (Kupiec, Independence, Conditional Coverage) and governance surface.
            </p>
          </div>
          {view.validationAcademic.available ? (
            <div className="flex flex-wrap items-center gap-1.5">
              <StatusBadge label={view.validationAcademic.championModel ?? "n/a"} tone="accent" />
              <StatusBadge
                label={`Global ${view.validationAcademic.globalVerdict}`}
                tone={verdictTone(view.validationAcademic.globalVerdict)}
              />
              <StatusBadge label={view.validationAcademic.championVerdict} tone={verdictTone(view.validationAcademic.championVerdict)} />
              <StatusBadge
                label={
                  view.validationAcademic.confidenceScore == null
                    ? `Confidence ${view.validationAcademic.confidenceLevel}`
                    : `Confidence ${view.validationAcademic.confidenceLevel} ${view.validationAcademic.confidenceScore.toFixed(0)}/100`
                }
                tone={confidenceTone(view.validationAcademic.confidenceLevel)}
              />
              <StatusBadge label={`p-threshold ${(view.validationAcademic.threshold * 100).toFixed(1)}%`} tone="neutral" />
            </div>
          ) : (
            <StatusBadge label="No validation data" tone="neutral" />
          )}
        </div>

        {view.validationAcademic.available ? (
          <div className="space-y-3">
            {view.validationAcademic.insufficientSampleCount > 0 ? (
              <div className="rounded-[var(--radius-lg)] border border-amber-500/25 bg-amber-500/8 px-3 py-2 text-[12px] text-[var(--color-text-soft)]">
                Zero rejection counts do not mean the models are fully validated yet.{" "}
                <span className="font-semibold text-[var(--color-text)]">
                  {view.validationAcademic.effectivePoints}/{view.validationAcademic.totalPoints}
                </span>{" "}
                points currently clear the sample floor, and{" "}
                <span className="font-semibold text-[var(--color-text)]">
                  {view.validationAcademic.insufficientSampleCount}
                </span>{" "}
                remain statistically under-sampled.
              </div>
            ) : null}

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-8">
              <MetricBlock
                label="Effective pass rate"
                value={
                  view.validationAcademic.effectivePoints <= 0
                    ? "n/a"
                    : formatPercent(view.validationAcademic.passCount / view.validationAcademic.effectivePoints, 0)
                }
                hint={`${view.validationAcademic.passCount} PASS on ${view.validationAcademic.effectivePoints}/${view.validationAcademic.totalPoints} sampled points`}
                tone={
                  view.validationAcademic.effectivePoints <= 0
                    ? "neutral"
                    : view.validationAcademic.failCount > 0
                      ? "warning"
                      : "success"
                }
                className="bg-transparent"
              />
              <MetricBlock
                label="Sample confidence"
                value={
                  view.validationAcademic.confidenceScore == null
                    ? view.validationAcademic.confidenceLevel
                    : `${view.validationAcademic.confidenceLevel} ${view.validationAcademic.confidenceScore.toFixed(0)}/100`
                }
                hint={view.validationAcademic.confidenceReason ?? "Statistical confidence from sample-size guardrails."}
                tone={confidenceTone(view.validationAcademic.confidenceLevel)}
                className="bg-transparent"
              />
              <MetricBlock
                label="Sampled points"
                value={`${view.validationAcademic.effectivePoints}/${view.validationAcademic.totalPoints}`}
                hint={`${view.validationAcademic.insufficientSampleCount} points still below the formal sample floor`}
                tone={
                  view.validationAcademic.insufficientSampleCount > 0
                    ? "warning"
                    : view.validationAcademic.effectivePoints > 0
                      ? "success"
                      : "neutral"
                }
                className="bg-transparent"
              />
              <MetricBlock
                label="Coverage rejects"
                value={String(view.validationAcademic.coverageFailCount)}
                hint="Kupiec UC rejections among sufficiently sampled points only"
                tone={
                  view.validationAcademic.effectivePoints <= 0
                    ? "neutral"
                    : view.validationAcademic.coverageFailCount > 0
                      ? "danger"
                      : "success"
                }
                className="bg-transparent"
              />
              <MetricBlock
                label="Independence rejects"
                value={String(view.validationAcademic.independenceFailCount)}
                hint="Christoffersen IND rejections among sufficiently sampled points only"
                tone={
                  view.validationAcademic.effectivePoints <= 0
                    ? "neutral"
                    : view.validationAcademic.independenceFailCount > 0
                      ? "warning"
                      : "success"
                }
                className="bg-transparent"
              />
              <MetricBlock
                label="Conditional rejects"
                value={String(view.validationAcademic.conditionalFailCount)}
                hint="Christoffersen CC rejections among sufficiently sampled points only"
                tone={
                  view.validationAcademic.effectivePoints <= 0
                    ? "neutral"
                    : view.validationAcademic.conditionalFailCount > 0
                      ? "danger"
                      : "success"
                }
                className="bg-transparent"
              />
              <MetricBlock
                label="Champion ES tails"
                value={view.validationAcademic.championEsTailObservations == null ? "n/a" : String(view.validationAcademic.championEsTailObservations)}
                hint="Tail observations"
                tone={view.validationAcademic.championEsTailObservations == null ? "neutral" : view.validationAcademic.championEsTailObservations > 0 ? "accent" : "warning"}
                className="bg-transparent"
              />
              <MetricBlock
                label="Champion ES shortfall"
                value={view.validationAcademic.championEsShortfallRatio == null ? "n/a" : view.validationAcademic.championEsShortfallRatio.toFixed(3)}
                hint={
                  view.validationAcademic.championEsBreachRate == null
                    ? "Observed tail / forecast ES"
                    : `Breach ${(view.validationAcademic.championEsBreachRate * 100).toFixed(2)}%`
                }
                tone={esShortfallTone(view.validationAcademic.championEsShortfallRatio)}
                className="bg-transparent"
              />
            </div>

            {view.validationAcademic.horizonRows.length > 0 ? (
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {view.validationAcademic.horizonRows.map((row) => (
                  <div
                    key={row.horizonDays}
                    className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)]/75 px-3 py-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                        Validation {row.horizonDays}d
                      </p>
                      <StatusBadge label={row.verdict} tone={verdictTone(row.verdict)} />
                    </div>
                    <p className="mt-1 text-[11px] text-[var(--color-text-soft)]">
                      Champion {row.championModel ?? "n/a"} | PASS {row.passCount} / {row.totalPoints}
                    </p>
                    <p className="mt-0.5 text-[11px] text-[var(--color-text-muted)]">
                      Warn {row.warnCount} | Fail {row.failCount} | Pass rate {row.passRate == null ? "n/a" : `${(row.passRate * 100).toFixed(0)}%`}
                    </p>
                    <p className="mt-0.5 text-[11px] text-[var(--color-text-muted)]">
                      Confidence {row.confidenceScore == null ? row.confidenceLevel : `${row.confidenceLevel} ${row.confidenceScore.toFixed(0)}/100`}
                    </p>
                  </div>
                ))}
              </div>
            ) : null}

            <div className="overflow-x-auto rounded-[var(--radius-lg)] border border-[var(--color-border)]">
              <table className="w-full min-w-[980px] text-left text-[12px]">
                <thead className="bg-[var(--color-surface)]/85 text-[10px] uppercase tracking-wide text-[var(--color-text-muted)]">
                  <tr>
                    <th className="px-3 py-2">Model</th>
                    <th className="px-3 py-2">Rank</th>
                    <th className="px-3 py-2">Score</th>
                    <th className="px-3 py-2">Exceptions</th>
                    <th className="px-3 py-2">Rate</th>
                    <th className="px-3 py-2">p(UC)</th>
                    <th className="px-3 py-2">p(IND)</th>
                    <th className="px-3 py-2">p(CC)</th>
                    <th className="px-3 py-2">ES tails</th>
                    <th className="px-3 py-2">ES shortfall</th>
                    <th className="px-3 py-2">ES breach</th>
                    <th className="px-3 py-2">Traffic</th>
                    <th className="px-3 py-2">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {view.validationAcademic.rows.map((row) => (
                    <tr key={row.model} className="border-t border-[var(--color-border)]">
                      <td className="px-3 py-2 font-semibold text-[var(--color-text)]">{row.model}</td>
                      <td className="px-3 py-2 text-[var(--color-text-soft)]">{row.rank ?? "n/a"}</td>
                      <td className="px-3 py-2 text-[var(--color-text-soft)]">{row.score == null ? "n/a" : row.score.toFixed(2)}</td>
                      <td className="px-3 py-2 text-[var(--color-text-soft)]">
                        {row.exceptions == null ? "n/a" : row.n == null ? row.exceptions : `${row.exceptions}/${row.n}`}
                      </td>
                      <td className="px-3 py-2 text-[var(--color-text-soft)]">
                        {row.actualRate == null ? "n/a" : `${(row.actualRate * 100).toFixed(2)}%`}
                        {row.expectedRate == null ? "" : ` (exp ${(row.expectedRate * 100).toFixed(2)}%)`}
                      </td>
                      <td className="px-3 py-2 text-[var(--color-text-soft)]">{row.pUc == null ? "n/a" : row.pUc.toFixed(4)}</td>
                      <td className="px-3 py-2 text-[var(--color-text-soft)]">{row.pInd == null ? "n/a" : row.pInd.toFixed(4)}</td>
                      <td className="px-3 py-2 text-[var(--color-text-soft)]">{row.pCc == null ? "n/a" : row.pCc.toFixed(4)}</td>
                      <td className="px-3 py-2 text-[var(--color-text-soft)]">{row.esTailObservations == null ? "n/a" : row.esTailObservations}</td>
                      <td className="px-3 py-2">
                        <StatusBadge
                          label={row.esShortfallRatio == null ? "n/a" : row.esShortfallRatio.toFixed(3)}
                          tone={esShortfallTone(row.esShortfallRatio)}
                        />
                      </td>
                      <td className="px-3 py-2">
                        <StatusBadge
                          label={row.esBreachRate == null ? "n/a" : `${(row.esBreachRate * 100).toFixed(2)}%`}
                          tone={esBreachTone(row.esBreachRate)}
                        />
                      </td>
                      <td className="px-3 py-2">
                        <StatusBadge label={row.trafficLight ?? "n/a"} tone={trafficLightTone(row.trafficLight)} />
                      </td>
                      <td className="px-3 py-2">
                        <StatusBadge label={row.verdict} tone={verdictTone(row.verdict)} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <p className="text-[12px] text-[var(--color-text-muted)]">
            No validation output found yet. Run a backtest to populate Kupiec/Christoffersen diagnostics.
          </p>
        )}
      </section>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartSurface
          option={makeBacktestOption(view.derived.backtestSeries)}
          mode="trace"
          dataCount={view.derived.backtestSeries.length}
          title="Backtest trace"
          meta={
            view.derived.backtestSeries.length
              ? `${view.derived.backtestSeries.length} rows`
              : undefined
          }
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No backtest data.</p>}
        />
        <ChartSurface
          option={makeLineOption(capitalSeries, CHART_PALETTE.green, { mode: "standard" })}
          mode="standard"
          dataCount={capitalSeries.length}
          title="Capital trajectory"
          meta={reportCapital?.reference_model?.toUpperCase()}
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No capital history.</p>}
        />
      </div>

      <ChartSurface
        option={makeGroupedBarOption(
          decisionSeries.labels,
          [
            {
              name: "Requested",
              data: decisionSeries.requested,
              color: CHART_PALETTE.gold,
            },
            {
              name: "Approved",
              data: decisionSeries.approved,
              color: CHART_PALETTE.green,
            },
          ],
          { mode: "comparison" },
        )}
        mode="comparison"
        dataCount={decisionSeries.labels.length}
        title="Requested vs approved exposure"
        meta={`${decisions.length} decisions`}
        emptyState={<p className="text-xs text-[var(--color-text-muted)]">No decisions.</p>}
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_260px]">
        {view.report ? (
          <ReportDocument
            content={view.normalizedReportContent}
            reportPath={view.report.report_markdown}
            portfolioSlug={view.resolvedPortfolio}
            reportId={view.report.report_id}
            chartPaths={view.report.chart_paths ?? []}
          />
        ) : (
          <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4 text-xs text-[var(--color-text-muted)]">
            No report content available. Generate a fresh report.
          </div>
        )}
        <ReportTableOfContents
          headings={view.headings}
          reportPath={view.meta.reportPath || "n/a"}
          chartCount={view.meta.chartCount}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Decisions
          </h4>
          <DecisionHistoryTable rows={decisions} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            Audit
          </h4>
          <AuditTrailTable rows={audit} />
        </div>
      </div>
    </div>
  );
}

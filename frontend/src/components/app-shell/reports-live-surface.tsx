"use client";

import { startTransition, useEffect, useState } from "react";

import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
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
import { api } from "@/lib/api/client";
import {
  makeBacktestOption,
  makeGroupedBarOption,
  makeLineOption,
} from "@/lib/chart-options";
import {
  type DeskReportViewModel,
  loadDeskReportViewModel,
} from "@/lib/report-view-model";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";
import {
  averageDecisionFillRatio,
  buildCapitalHistorySeries,
  buildDecisionDeltaComparison,
} from "@/lib/view-models";

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

function preferredReportSource(view: DeskReportViewModel) {
  return view.meta.preferredSnapshotSource === "mt5_live_bridge"
    ? "mt5_live_bridge"
    : undefined;
}

function reportSnapshotLabel(source: string | null | undefined) {
  return source === "mt5_live_bridge" ? "mt5 live snapshot" : "historical snapshot";
}

export function ReportsLiveSurface({
  portfolioSlug,
  initialView,
}: {
  portfolioSlug: string;
  initialView: DeskReportViewModel;
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialView.liveState);
  const [view, setView] = useState(initialView);
  const [decisions, setDecisions] = useState(initialView.decisions);
  const [capitalHistory, setCapitalHistory] = useState(initialView.capitalHistory);
  const [audit, setAudit] = useState(initialView.audit);

  const reportSource = preferredReportSource(view);

  useEffect(() => {
    let cancelled = false;

    const refreshLightweightState = async () => {
      try {
        let nextCapitalHistory = await api.reportCapitalHistory(portfolioSlug, 8, reportSource);
        if (reportSource && nextCapitalHistory.length === 0) {
          nextCapitalHistory = await api.reportCapitalHistory(portfolioSlug, 8);
        }
        const [nextDecisions, nextAudit] = await Promise.all([
          api.reportDecisionHistory(portfolioSlug, 12).catch(() => []),
          api.recentAudit(portfolioSlug, 16).catch(() => []),
        ]);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setDecisions(nextDecisions);
          setCapitalHistory(nextCapitalHistory);
          setAudit(nextAudit);
        });
      } catch {
        // Keep the current report-side activity on transient failures.
      }
    };

    void refreshLightweightState();
    return () => {
      cancelled = true;
    };
  }, [portfolioSlug, liveState?.sequence, reportSource]);

  useEffect(() => {
    let cancelled = false;

    const refreshFullView = async () => {
      try {
        const nextView = await loadDeskReportViewModel(portfolioSlug);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setView(nextView);
          setDecisions(nextView.decisions);
          setCapitalHistory(nextView.capitalHistory);
          setAudit(nextView.audit);
        });
      } catch {
        // Keep the current report view on transient failures.
      }
    };

    const timer = window.setInterval(() => {
      void refreshFullView();
    }, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [portfolioSlug]);

  const resolvedLiveState = liveState ?? view.liveState;
  const liveCapital = resolvedLiveState?.capital_usage ?? view.capital;
  const selectedModel =
    resolvedLiveState?.risk_summary?.reference_model ?? view.selectedModel;
  const liveVarValue = Number(
    resolvedLiveState?.risk_summary?.var?.[selectedModel] ??
      Object.values(resolvedLiveState?.risk_summary?.var ?? {})[0] ??
      view.varValue,
  );
  const liveEsValue = Number(
    resolvedLiveState?.risk_summary?.es?.[selectedModel] ??
      Object.values(resolvedLiveState?.risk_summary?.es ?? {})[0] ??
      view.esValue,
  );
  const fillRatio = averageDecisionFillRatio(decisions);
  const capitalSeries = buildCapitalHistorySeries(capitalHistory);
  const decisionSizeSeries = buildDecisionDeltaComparison(decisions);
  const latestReportEvent = audit.find((event) => event.action_type === "report.run");
  const reportTimestamp = latestReportEvent?.created_at
    ? formatTimestamp(latestReportEvent.created_at)
    : view.meta.reportTimestamp;
  const executiveSummary = [
    {
      label: `VaR / ${selectedModel.toUpperCase()}`,
      value: formatCurrency(liveVarValue),
      copy: resolvedLiveState?.risk_summary?.latest_observation
        ? `Live selected-model risk level from the MT5 bridge sample ending ${formatTimestamp(
            resolvedLiveState.risk_summary.latest_observation,
          )}.`
        : "Current selected model risk level for the latest persisted portfolio snapshot.",
      tone: "accent" as const,
    },
    {
      label: `ES / ${selectedModel.toUpperCase()}`,
      value: formatCurrency(liveEsValue),
      copy: resolvedLiveState?.risk_summary
        ? "Tail-loss expectation derived from the live bridge risk summary."
        : "Tail-loss expectation carried into the report baseline.",
      tone: "warning" as const,
    },
    {
      label: "Capital headroom",
      value: liveCapital ? formatPercent(liveCapital.headroom_ratio ?? 0, 0) : "n/a",
      copy: liveCapital
        ? `Remaining ${formatCurrency(liveCapital.total_capital_remaining_eur)} before hitting current budget boundaries.`
        : "No persisted capital snapshot yet.",
      tone: "success" as const,
    },
    {
      label: "Average fill ratio",
      value: fillRatio == null ? "n/a" : formatPercent(fillRatio, 0),
      copy: "Advisory decisions approved versus requested notionals over recent runs.",
      tone: "neutral" as const,
    },
  ];
  const topRecommendation = liveCapital?.recommendations?.[0] ?? view.capital?.recommendations?.[0];
  const narrativeSummary = [
    `Champion model: ${(view.comparison?.champion_model ?? selectedModel).toUpperCase()} with a score gap of ${
      view.comparison?.score_gap != null ? view.comparison.score_gap.toFixed(1) : "n/a"
    }.`,
    liveCapital
      ? `Capital posture remains ${liveCapital.status.toLowerCase()} with ${formatCurrency(
          liveCapital.total_capital_consumed_eur,
        )} consumed from ${formatCurrency(liveCapital.total_capital_budget_eur)}.`
      : "Capital posture is not yet available.",
    decisions.length
      ? `${decisions.length} recent decisions are persisted, with an average fill ratio of ${
          fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)
        }.`
      : "No recent decisions are persisted yet.",
    resolvedLiveState?.reconciliation
      ? `${resolvedLiveState.reconciliation.manual_event_count} manual MT5 event(s) and ${resolvedLiveState.reconciliation.unmatched_execution_count} unmatched execution attempt(s) are currently visible from the live bridge.`
      : "Live reconciliation telemetry is not currently available.",
  ];
  const reportMeta = {
    ...view.meta,
    reportTimestamp,
    chartCount: view.report?.chart_paths?.length ?? view.meta.chartCount,
  };
  const snapshotLabel = reportSnapshotLabel(
    resolvedLiveState?.risk_summary?.source ?? view.meta.preferredSnapshotSource,
  );

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Reports & Governance"
        title="A real desk report, readable in the platform and exportable as a branded PDF."
        description="Narrative, analytics, decision continuity and audit history now live in one editorial surface, with MT5 bridge posture layered directly into the report workflow."
        aside={
          <div className="flex flex-col items-start gap-3 lg:items-end">
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge label={view.resolvedPortfolio} tone="accent" />
              <StatusBadge
                label={snapshotLabel}
                tone={snapshotLabel === "mt5 live snapshot" ? "success" : "neutral"}
              />
              <StatusBadge
                label={transport === "stream" ? "stream" : transport === "polling" ? "polling" : "connecting"}
                tone={transport === "stream" ? "success" : transport === "polling" ? "warning" : "neutral"}
              />
            </div>
            <ReportActions
              portfolioSlug={view.resolvedPortfolio}
              onGenerated={async () => {
                const nextView = await loadDeskReportViewModel(portfolioSlug);
                startTransition(() => {
                  setView(nextView);
                  setDecisions(nextView.decisions);
                  setCapitalHistory(nextView.capitalHistory);
                  setAudit(nextView.audit);
                });
              }}
            />
          </div>
        }
      />

      <LiveOperatorAlerts
        alerts={resolvedLiveState?.operator_alerts ?? []}
        title="Reporting watchlist"
        copy="Live bridge health, budget pressure and desk-versus-broker drift stay visible while you read and export the report."
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
                Generated for {view.resolvedPortfolio}. The report now reads like a desk note:
                model lead, live capital headroom, decision friction and audit continuity are all
                on the same canvas.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <StatusBadge label={selectedModel.toUpperCase()} tone="accent" />
              <StatusBadge
                label={liveCapital?.status ?? "snapshot pending"}
                tone={statusTone(liveCapital?.status)}
              />
              <StatusBadge label={reportMeta.reportTimestamp} />
            </div>
          </div>

          <div className="mt-7 grid gap-4 lg:grid-cols-2 xl:grid-cols-4">
            {executiveSummary.map((item) => (
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
                {narrativeSummary.map((item) => (
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
                <ReportHighlight label="Generated" value={reportMeta.reportTimestamp} />
                <ReportHighlight label="Snapshot source" value={snapshotLabel} />
                <ReportHighlight label="Charts embedded" value={String(reportMeta.chartCount)} />
                <ReportHighlight
                  label="Backtest rows"
                  value={String(view.derived.backtestSeries.length)}
                />
                <ReportHighlight label="Decision history" value={`${decisions.length} persisted items`} />
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
          headings={view.headings}
          reportPath={reportMeta.reportPath || "No report path"}
          chartCount={reportMeta.chartCount}
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.08fr)_minmax(320px,0.92fr)]">
        <ChartSurface
          option={makeBacktestOption(view.derived.backtestSeries)}
          mode="trace"
          dataCount={view.derived.backtestSeries.length}
          eyebrow="Analytics"
          title="Backtest trace"
          description="PnL and VaR remain a primary reading surface here, not a tiny appendix chart."
          meta={
            view.derived.backtestSeries.length
              ? `${view.derived.backtestSeries.length} rows`
              : "No backtest frame"
          }
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No backtest trace is available yet.
            </div>
          }
        />

        <ChartSurface
          option={makeLineOption(capitalSeries, "#5fd4a6", { mode: "standard" })}
          mode="standard"
          dataCount={capitalSeries.length}
          eyebrow="Capital"
          title="Capital usage trajectory"
          description="Budget consumption is now linked to the live MT5 bridge when the desk runs in live mode."
          meta={liveCapital?.reference_model ? liveCapital.reference_model.toUpperCase() : "No capital snapshot"}
          footer={
            <div className="space-y-3 text-sm text-[var(--color-text-soft)]">
              <div className="flex items-center justify-between gap-3">
                <span>Remaining capital</span>
                <span className="mono text-white">
                  {liveCapital ? formatCurrency(liveCapital.total_capital_remaining_eur) : "n/a"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span>Headroom ratio</span>
                <span className="mono text-white">
                  {liveCapital ? formatPercent(liveCapital.headroom_ratio ?? 0, 0) : "n/a"}
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
          decisionSizeSeries.labels,
          [
            {
              name: "Requested",
              data: decisionSizeSeries.requested,
              color: "#d89b49",
            },
            {
              name: "Approved",
              data: decisionSizeSeries.approved,
              color: "#5fd4a6",
            },
          ],
          { mode: "comparison" },
        )}
        mode="comparison"
        dataCount={decisionSizeSeries.labels.length}
        eyebrow="Decision continuity"
        title="Requested versus approved notional"
        description="Sizing friction, reductions and approvals remain readable in the report itself instead of being buried in history tables."
        meta={`${decisions.length} recent decisions`}
        footer={
          <div className="grid gap-4 lg:grid-cols-3">
            <MetricBlock
              label="Average fill"
              value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)}
              hint="Approved vs requested"
              className="bg-transparent"
            />
            <MetricBlock
              label="Recent decisions"
              value={String(decisions.length)}
              hint="Persisted advisory outcomes"
              className="bg-transparent"
            />
            <MetricBlock
              label="Audit events"
              value={String(audit.length)}
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

      {view.report ? (
        <ReportDocument
          content={view.normalizedReportContent}
          reportPath={view.report.report_markdown}
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
              Last update {latestReportEvent?.created_at ? formatTimestamp(latestReportEvent.created_at) : "n/a"}
            </div>
          </div>
          <DecisionHistoryTable rows={decisions} />
        </div>

        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Audit appendix
          </div>
          <AuditTrailTable rows={audit} />
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

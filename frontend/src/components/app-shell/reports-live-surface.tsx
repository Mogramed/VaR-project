"use client";

import { startTransition, useEffect, useEffectEvent, useState } from "react";

import { useDeskLive } from "@/components/app-shell/desk-live-provider";
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
import { formatCurrency, formatPercent, formatSourceLabel } from "@/lib/utils";
import {
  averageDecisionFillRatio,
  buildCapitalHistorySeries,
  buildDecisionDeltaComparison,
} from "@/lib/view-models";

function snapshotLabel(src: string | null | undefined) {
  return formatSourceLabel(src ?? "auto");
}

export function ReportsLiveSurface({
  portfolioSlug,
  initialView,
}: {
  portfolioSlug: string;
  initialView: DeskReportViewModel;
}) {
  const { liveState, transport, artifactVersion } = useDeskLive();
  const [view, setView] = useState(initialView);

  useEffect(() => {
    startTransition(() => setView(initialView));
  }, [initialView]);
  const refreshView = useEffectEvent(async (isCancelled: () => boolean) => {
    const nextView = await loadDeskReportViewModel(portfolioSlug, {
      liveState: liveState ?? null,
    });
    if (!isCancelled()) {
      startTransition(() => setView(nextView));
    }
  });

  useEffect(() => {
    let cancelled = false;
    void refreshView(() => cancelled).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [artifactVersion, portfolioSlug]);

  const resolved = liveState ?? view.liveState;
  const decisions = view.decisions;
  const capitalHistory = view.capitalHistory;
  const audit = view.audit;
  const liveCapital = resolved?.capital_usage ?? view.capital;
  const selectedModel = resolved?.risk_summary?.reference_model ?? view.selectedModel;
  const varValue = Number(
    resolved?.risk_summary?.var?.[selectedModel]
      ?? Object.values(resolved?.risk_summary?.var ?? {})[0]
      ?? view.varValue,
  );
  const esValue = Number(
    resolved?.risk_summary?.es?.[selectedModel]
      ?? Object.values(resolved?.risk_summary?.es ?? {})[0]
      ?? view.esValue,
  );
  const fillRatio = averageDecisionFillRatio(decisions);
  const capitalSeries = buildCapitalHistorySeries(capitalHistory);
  const decisionSeries = buildDecisionDeltaComparison(decisions);
  const resolvedSource = resolved?.risk_summary?.source ?? view.meta.preferredSnapshotSource;
  const label = snapshotLabel(resolvedSource);

  return (
    <div className="desk-page space-y-4">
      <PageHeader
        eyebrow="Reports"
        title="Desk report, analytics and PDF export"
        aside={(
          <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-1.5">
              <StatusBadge label={view.resolvedPortfolio} tone="accent" />
              <StatusBadge
                label={label}
                tone={String(resolvedSource ?? "").startsWith("mt5_live") ? "success" : "neutral"}
              />
              <StatusBadge
                label={transport}
                tone={transport === "stream" ? "success" : "warning"}
              />
            </div>
            <ReportActions
              portfolioSlug={view.resolvedPortfolio}
              onGenerated={async () => {
                const nextView = await loadDeskReportViewModel(portfolioSlug, {
                  liveState: liveState ?? null,
                });
                startTransition(() => setView(nextView));
              }}
            />
          </div>
        )}
      />

      <LiveOperatorAlerts alerts={resolved?.operator_alerts ?? []} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricBlock
          label={`VaR / ${selectedModel.toUpperCase()}`}
          value={formatCurrency(varValue)}
          tone="accent"
        />
        <MetricBlock
          label={`ES / ${selectedModel.toUpperCase()}`}
          value={formatCurrency(esValue)}
          tone="warning"
        />
        <MetricBlock
          label="Headroom"
          value={liveCapital ? formatPercent(liveCapital.headroom_ratio ?? 0, 0) : "n/a"}
          tone="success"
        />
        <MetricBlock
          label="Avg fill"
          value={fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)}
        />
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
          meta={liveCapital?.reference_model?.toUpperCase()}
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

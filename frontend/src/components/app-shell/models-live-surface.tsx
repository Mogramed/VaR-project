"use client";

import { useQuery } from "@tanstack/react-query";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { DashboardActiveFilters } from "@/components/app-shell/dashboard-active-filters";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { LivePostureBanner } from "@/components/app-shell/live-posture-banner";
import { LiveRuntimeBadgeGroup } from "@/components/app-shell/live-runtime-badge-group";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { ModelRankingTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type { BacktestFrameResponse, ModelComparisonResponse, ValidationRunSummary } from "@/lib/api/types";
import { CHART_PALETTE, makeBacktestOption, makeBarOption } from "@/lib/chart-options";
import {
  deskArtifactQueryKey,
  deskArtifactQueryOptions,
} from "@/components/app-shell/desk-artifact-query";
import { useDashboardPrefs } from "@/lib/dashboard-preferences-context";
import { preferredHeadlineRisk } from "@/lib/risk-surface";
import { formatCurrency, formatPercent, formatTimestamp, joinLabelParts } from "@/lib/utils";
import { buildBacktestSeries, buildModelScoreSeries } from "@/lib/view-models";

type ValidationVerdict = "PASS" | "WARN" | "FAIL" | "N/A";

const ES_ACERBI_WARN_PVALUE = 0.05;
const ES_ACERBI_BREACH_PVALUE = 0.01;
const ES_ACERBI_MIN_OBSERVATIONS = 60;

function verdictTone(verdict: ValidationVerdict) {
  if (verdict === "FAIL") return "danger" as const;
  if (verdict === "WARN") return "warning" as const;
  if (verdict === "PASS") return "success" as const;
  return "neutral" as const;
}

function confidenceTone(level: string | null | undefined) {
  const normalized = String(level ?? "").trim().toUpperCase();
  if (normalized === "LOW") return "danger" as const;
  if (normalized === "MEDIUM") return "warning" as const;
  if (normalized === "HIGH") return "success" as const;
  return "neutral" as const;
}

export function ModelsLiveSurface({
  portfolioSlug, initialComparison, initialValidation, initialFrame,
}: {
  portfolioSlug: string;
  initialComparison: ModelComparisonResponse | null;
  initialValidation: ValidationRunSummary | null;
  initialFrame: BacktestFrameResponse | null;
}) {
  const { liveState, transport } = useDeskLive();
  const { preferredHorizonDays, resolvePreferredModel } = useDashboardPrefs();
  const comparisonQuery = useQuery({
    queryKey: deskArtifactQueryKey("models", "comparison", portfolioSlug),
    queryFn: () => api.latestModelComparison(portfolioSlug),
    initialData: initialComparison,
    ...deskArtifactQueryOptions,
  });
  const validationQuery = useQuery({
    queryKey: deskArtifactQueryKey("models", "validation", portfolioSlug),
    queryFn: () => api.latestValidation(portfolioSlug),
    initialData: initialValidation,
    ...deskArtifactQueryOptions,
  });
  const frameQuery = useQuery({
    queryKey: deskArtifactQueryKey("models", "backtest-frame", portfolioSlug, 240),
    queryFn: () => api.latestBacktestFrame(portfolioSlug, 240),
    initialData: initialFrame,
    ...deskArtifactQueryOptions,
  });
  const comparison = comparisonQuery.data ?? initialComparison;
  const validation = validationQuery.data ?? initialValidation;
  const frame = frameQuery.data ?? initialFrame;

  const ranking = comparison?.ranking ?? [];
  const backtestSeries = frame ? buildBacktestSeries(frame) : [];
  const scoreSeries = comparison ? buildModelScoreSeries(comparison) : [];
  const fallbackSelectedModel = liveState?.risk_summary?.reference_model ?? comparison?.champion_model ?? validation?.best_model ?? "hist";
  const selectedModel = resolvePreferredModel(fallbackSelectedModel) ?? fallbackSelectedModel;
  const liveRisk95 = preferredHeadlineRisk(liveState?.risk_summary?.headline_risk, [`live_${preferredHorizonDays}d_95`, "live_1d_95"]);
  const liveRisk99 = preferredHeadlineRisk(liveState?.risk_summary?.headline_risk, [`live_${preferredHorizonDays}d_99`, "live_1d_99"]);
  const liveCapital = liveState?.capital_usage ?? null;
  const validationSurface = (comparison?.validation_surface ?? null) as {
    points?: unknown[];
    governance_summary?: unknown;
    horizon_governance?: unknown;
  } | null;
  const validationSurfacePoints = Array.isArray(validationSurface?.points)
    ? validationSurface.points
    : [];
  const validationSummary = toRecord(validation?.summary);
  const validationModels = toRecord(validationSummary?.models);
  const selectedValidationModel = resolveValidationModelMetrics(validationModels, selectedModel, comparison?.champion_model ?? undefined);
  const selectedEsShortfallRatio = toNumber(selectedValidationModel?.es_shortfall_ratio);
  const selectedEsBreachRate = toNumber(selectedValidationModel?.es_breach_rate);
  const selectedEsTailObservations = toNumber(selectedValidationModel?.es_tail_observations);
  const selectedEsAcerbiStat = toNumber(selectedValidationModel?.es_acerbi_stat ?? selectedValidationModel?.es_acerbi_z);
  const selectedEsAcerbiPValue = toNumber(selectedValidationModel?.es_acerbi_p_value ?? selectedValidationModel?.es_acerbi_p);
  const selectedEsAcerbiObservations = toNumber(
    selectedValidationModel?.es_acerbi_observations ?? selectedValidationModel?.es_acerbi_n,
  );
  const hasEsAcerbiSample = selectedEsAcerbiObservations != null && selectedEsAcerbiObservations >= ES_ACERBI_MIN_OBSERVATIONS;
  const esAcerbiVerdict: ValidationVerdict = !hasEsAcerbiSample || selectedEsAcerbiPValue == null
    ? "N/A"
    : selectedEsAcerbiPValue <= ES_ACERBI_BREACH_PVALUE
      ? "FAIL"
      : selectedEsAcerbiPValue <= ES_ACERBI_WARN_PVALUE
        ? "WARN"
        : "PASS";
  const governanceSummary = toRecord(validationSurface?.governance_summary);
  const governanceStatusCounts = toRecord(governanceSummary?.status_counts);
  const governanceTotalPoints = toNumber(governanceSummary?.total_points);
  const governancePassCount = toNumber(governanceStatusCounts?.PASS);
  const governanceWarnCount = toNumber(governanceStatusCounts?.WARN);
  const governanceFailCount = toNumber(governanceStatusCounts?.FAIL);
  const governanceCoverageFails = toNumber(governanceSummary?.coverage_fail_count);
  const governanceIndependenceFails = toNumber(governanceSummary?.independence_fail_count);
  const governanceConditionalFails = toNumber(governanceSummary?.conditional_fail_count);
  const governancePvalueThreshold = toNumber(governanceSummary?.pvalue_threshold);
  const governanceConfidenceScore = toNumber(governanceSummary?.confidence_score);
  const governanceConfidenceLevel = toUpper(governanceSummary?.confidence_level) ?? "UNKNOWN";
  const governanceConfidenceReason = typeof governanceSummary?.confidence_reason === "string"
    ? governanceSummary.confidence_reason.trim()
    : null;
  const horizonGovernance = toRecord(validationSurface?.horizon_governance) ?? {};
  const horizonPayload = toRecord(horizonGovernance.horizons) ?? {};
  const horizonOrder = Array.isArray(horizonGovernance.horizon_order)
    ? horizonGovernance.horizon_order
      .map((value) => toInteger(value))
      .filter((value): value is number => value != null && value > 0)
    : [];
  const inferredHorizonOrder = Object.keys(horizonPayload)
    .map((key) => key.match(/^h(\d+)$/))
    .filter((match): match is RegExpMatchArray => match != null)
    .map((match) => Number.parseInt(match[1], 10))
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);
  const selectedHorizonOrder = horizonOrder.length ? horizonOrder : inferredHorizonOrder;
  const horizonRows = selectedHorizonOrder.map((horizonDays) => {
    const row = toRecord(horizonPayload[`h${horizonDays}`]) ?? {};
    const rowStatusCounts = toRecord(row.status_counts) ?? {};
    const rowPassCount = toInteger(rowStatusCounts.PASS) ?? 0;
    const rowWarnCount = toInteger(rowStatusCounts.WARN) ?? 0;
    const rowFailCount = toInteger(rowStatusCounts.FAIL) ?? 0;
    const rowTotalPoints = toInteger(row.total_points) ?? rowPassCount + rowWarnCount + rowFailCount;
    const rowPassRate = toNumber(row.pass_rate) ?? (rowTotalPoints > 0 ? rowPassCount / rowTotalPoints : null);
    const rowVerdict =
      normalizeValidationVerdict(row.verdict)
      ?? (rowFailCount > 0 ? "FAIL" : rowWarnCount > 0 ? "WARN" : rowPassCount > 0 ? "PASS" : "N/A");
    const rowConfidenceLevel = toUpper(row.confidence_level) ?? "UNKNOWN";
    const rowConfidenceScore = toNumber(row.confidence_score);
    return {
      horizonDays,
      championModel: toUpper(row.champion_model),
      verdict: rowVerdict,
      passRate: rowPassRate,
      passCount: rowPassCount,
      warnCount: rowWarnCount,
      failCount: rowFailCount,
      totalPoints: rowTotalPoints,
      confidenceLevel: rowConfidenceLevel,
      confidenceScore: rowConfidenceScore,
    };
  });
  const globalVerdict =
    normalizeValidationVerdict(governanceSummary?.verdict)
    ?? normalizeValidationVerdict(horizonGovernance.overall_verdict)
    ?? (
      governanceFailCount != null && governanceFailCount > 0
        ? "FAIL"
        : governanceWarnCount != null && governanceWarnCount > 0
          ? "WARN"
          : governancePassCount != null && governancePassCount > 0
            ? "PASS"
            : "N/A"
    );
  const governancePassRate = governanceTotalPoints && governanceTotalPoints > 0 && governancePassCount != null
    ? governancePassCount / governanceTotalPoints
    : null;

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Models" title="Champion, challenger and statistical credibility"
        aside={<>
          {comparison?.champion_model ? <StatusBadge label={comparison.champion_model.toUpperCase()} tone="accent" /> : null}
          <LiveRuntimeBadgeGroup liveState={liveState} transport={transport} showBridge={false} />
        </>}
      />
      <DashboardActiveFilters showSymbol={false} />
      <LivePostureBanner liveState={liveState} transport={transport} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-8">
        <MetricBlock label="Champion" value={(comparison?.champion_model ?? "n/a").toUpperCase()} />
        <MetricBlock label="Reporting champion" value={(comparison?.champion_model_reporting ?? comparison?.challenger_model ?? "n/a").toUpperCase()} />
        <MetricBlock label="Exception rate" value={validation ? formatPercent(validation.expected_rate) : "n/a"} tone="warning" />
        <MetricBlock label="Score gap" value={comparison?.score_gap != null ? comparison.score_gap.toFixed(1) : "n/a"} tone="accent" />
        <MetricBlock
          label="Stat checks"
          value={governancePassRate == null ? "n/a" : formatPercent(governancePassRate, 0)}
          hint={
            joinLabelParts(
              globalVerdict === "N/A" ? null : `Global ${globalVerdict}`,
              governanceTotalPoints == null || governancePassCount == null
                ? null
                : `${Math.round(governancePassCount)}/${Math.round(governanceTotalPoints)} pass`,
              governanceConfidenceLevel === "UNKNOWN"
                ? null
                : governanceConfidenceScore == null
                  ? `Confidence ${governanceConfidenceLevel}`
                  : `Confidence ${governanceConfidenceLevel} ${governanceConfidenceScore.toFixed(0)}/100`,
            ) || undefined
          }
          tone={
            governanceFailCount == null
              ? "neutral"
              : governanceFailCount > 0
                ? "danger"
                : governanceWarnCount != null && governanceWarnCount > 0
                  ? "warning"
                  : confidenceTone(governanceConfidenceLevel)
          }
        />
        <MetricBlock
          label="ES tail ratio"
          value={selectedEsShortfallRatio == null ? "n/a" : selectedEsShortfallRatio.toFixed(2)}
          hint={
            joinLabelParts(
              selectedEsBreachRate == null ? null : `ES breach ${formatPercent(selectedEsBreachRate, 0)}`,
              selectedEsTailObservations == null ? null : `${Math.round(selectedEsTailObservations)} tail obs`,
            ) || undefined
          }
          tone={
            selectedEsShortfallRatio == null
              ? "neutral"
              : selectedEsShortfallRatio > 1.1
                ? "warning"
                : "success"
          }
        />
        <MetricBlock
          label="ES Acerbi p-value"
          value={!hasEsAcerbiSample || selectedEsAcerbiPValue == null ? "n/a" : selectedEsAcerbiPValue.toFixed(4)}
          hint={
            joinLabelParts(
              selectedEsAcerbiStat == null ? null : `z ${selectedEsAcerbiStat.toFixed(2)}`,
              selectedEsAcerbiObservations == null ? null : `${Math.round(selectedEsAcerbiObservations)} obs`,
              esAcerbiVerdict === "N/A" ? null : esAcerbiVerdict,
            ) || undefined
          }
          tone={
            esAcerbiVerdict === "FAIL"
              ? "danger"
              : esAcerbiVerdict === "WARN"
                ? "warning"
                : esAcerbiVerdict === "PASS"
                  ? "success"
                  : "neutral"
          }
        />
        <MetricBlock
          label={`Live VaR / ES (${preferredHorizonDays}d)`}
          value={liveRisk95 == null ? "n/a" : formatCurrency(liveRisk95.var)}
          hint={liveRisk99 ? `99% ES ${formatCurrency(liveRisk99.es)}` : liveState?.risk_summary?.latest_observation ? formatTimestamp(liveState.risk_summary.latest_observation) : undefined}
          tone="warning"
        />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_300px]">
        <ChartSurface
          option={makeBacktestOption(backtestSeries)}
          mode="trace" dataCount={backtestSeries.length}
          title="Backtest trace" meta={backtestSeries.length ? `${backtestSeries.length} rows` : undefined}
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No backtest frame available.</p>}
        />
        <div className="space-y-3">
          <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Validation</div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Best model</span>
                <span className="mono font-semibold text-[var(--color-text)]">{(validation?.best_model ?? selectedModel).toUpperCase()}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Reporting champion</span>
                <span className="mono text-[var(--color-text)]">{(comparison?.champion_model_reporting ?? "n/a").toUpperCase()}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Alpha</span>
                <span className="mono text-[var(--color-text)]">{validation ? formatPercent(validation.alpha, 0) : "n/a"}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Headroom</span>
                <span className="mono text-[var(--color-green)]">{liveCapital ? formatPercent(liveCapital.headroom_ratio ?? 0, 0) : "n/a"}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">ES tail ratio</span>
                <span className="mono text-[var(--color-text)]">
                  {selectedEsShortfallRatio == null ? "n/a" : selectedEsShortfallRatio.toFixed(2)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">ES Acerbi z / p</span>
                <span className="mono text-[var(--color-text)]">
                  {selectedEsAcerbiStat == null || selectedEsAcerbiPValue == null
                    ? "n/a"
                    : `${selectedEsAcerbiStat.toFixed(2)} / ${selectedEsAcerbiPValue.toFixed(4)}`}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">ES Acerbi verdict</span>
                <StatusBadge label={esAcerbiVerdict} tone={verdictTone(esAcerbiVerdict)} />
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Stat status</span>
                <span className="mono text-[var(--color-text)]">
                  {governancePassCount == null || governanceTotalPoints == null
                    ? "n/a"
                    : `P${Math.round(governancePassCount)} W${Math.round(governanceWarnCount ?? 0)} F${Math.round(governanceFailCount ?? 0)}`}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Global verdict</span>
                <StatusBadge label={globalVerdict} tone={verdictTone(globalVerdict)} />
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Coverage fails</span>
                <span className="mono text-[var(--color-text)]">
                  {governanceCoverageFails == null ? "n/a" : Math.round(governanceCoverageFails)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Independence fails</span>
                <span className="mono text-[var(--color-text)]">
                  {governanceIndependenceFails == null ? "n/a" : Math.round(governanceIndependenceFails)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Conditional fails</span>
                <span className="mono text-[var(--color-text)]">
                  {governanceConditionalFails == null ? "n/a" : Math.round(governanceConditionalFails)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">P-value threshold</span>
                <span className="mono text-[var(--color-text)]">
                  {governancePvalueThreshold == null ? "n/a" : `${(governancePvalueThreshold * 100).toFixed(1)}%`}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[var(--color-text-muted)]">Sample confidence</span>
                <StatusBadge
                  label={
                    governanceConfidenceScore == null
                      ? governanceConfidenceLevel
                      : `${governanceConfidenceLevel} ${governanceConfidenceScore.toFixed(0)}/100`
                  }
                  tone={confidenceTone(governanceConfidenceLevel)}
                />
              </div>
              {governanceConfidenceReason ? (
                <p className="text-[11px] text-[var(--color-text-soft)]">{governanceConfidenceReason}</p>
              ) : null}
              {horizonRows.length > 0 ? (
                <div className="space-y-1.5 border-t border-[var(--color-border)] pt-2">
                  {horizonRows.map((row) => (
                    <div key={row.horizonDays} className="flex items-center justify-between gap-2 text-xs">
                      <span className="text-[var(--color-text-muted)]">
                        H{row.horizonDays}d {row.championModel ? `(${row.championModel})` : ""}
                      </span>
                      <span className="mono text-[var(--color-text-soft)]">
                        {row.passCount}/{row.totalPoints}
                        {row.passRate == null ? "" : ` ${formatPercent(row.passRate, 0)}`}
                        {row.confidenceScore == null ? "" : ` | ${row.confidenceLevel} ${row.confidenceScore.toFixed(0)}/100`}
                      </span>
                      <StatusBadge label={row.verdict} tone={verdictTone(row.verdict)} />
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
          {ranking.slice(0, 3).map((row) => (
            <div key={row.model} className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--color-text)]">{row.model.toUpperCase()}</span>
                <span className="mono text-sm text-[var(--color-accent)]">{row.score.toFixed(1)}</span>
              </div>
              <div className="mt-1.5 flex gap-4 text-[11px] text-[var(--color-text-muted)]">
                <span>Rate {formatPercent(row.actual_rate)}</span>
                <span>VaR {formatCurrency(row.current_var)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(280px,0.85fr)_minmax(0,1.15fr)]">
        <ChartSurface
          option={makeBarOption(scoreSeries, { color: CHART_PALETTE.gold, negativeColor: CHART_PALETTE.green, mode: ranking.length <= 5 ? "sparse" : "standard" })}
          mode={ranking.length <= 5 ? "sparse" : "standard"} dataCount={ranking.length}
          title="Model scores"
          emptyState={<p className="text-xs text-[var(--color-text-muted)]">No ranking available.</p>}
        />
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Ranking table</h4>
          <ModelRankingTable rows={ranking} />
          {validationSurfacePoints.length > 0 ? (
            <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3 text-[11px] text-[var(--color-text-soft)]">
              Validation surface loaded with {validationSurfacePoints.length} model / alpha / horizon points.
              {governancePassCount != null && governanceWarnCount != null && governanceFailCount != null ? (
                <span className="block pt-1.5">
                  Statistical checks: PASS {Math.round(governancePassCount)} | WARN {Math.round(governanceWarnCount)} | FAIL {Math.round(governanceFailCount)}
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function toRecord(value: unknown): Record<string, unknown> | null {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function toInteger(value: unknown): number | null {
  const parsed = toNumber(value);
  if (parsed == null) {
    return null;
  }
  return Math.trunc(parsed);
}

function toUpper(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toUpperCase();
  return normalized || null;
}

function normalizeValidationVerdict(value: unknown): ValidationVerdict | null {
  const normalized = toUpper(value);
  if (normalized === "PASS" || normalized === "WARN" || normalized === "FAIL") {
    return normalized;
  }
  if (normalized === "N/A" || normalized === "NA") {
    return "N/A";
  }
  return null;
}

function resolveValidationModelMetrics(
  models: Record<string, unknown> | null,
  selectedModel: string,
  championModel?: string,
): Record<string, unknown> | null {
  if (models == null) {
    return null;
  }
  const selected = toRecord(models[selectedModel]);
  if (selected != null) {
    return selected;
  }
  if (championModel) {
    const champion = toRecord(models[championModel]);
    if (champion != null) {
      return champion;
    }
  }
  const fallbackKey = Object.keys(models)[0];
  if (!fallbackKey) {
    return null;
  }
  return toRecord(models[fallbackKey]);
}

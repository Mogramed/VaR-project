import { api } from "@/lib/api/client";
import type { MT5LiveStateResponse } from "@/lib/api/types";
import { formatCurrency, formatPercent, formatTimestamp, slugifyHeading } from "@/lib/utils";
import {
  averageDecisionFillRatio,
  buildBacktestSeries,
  buildCapitalHistorySeries,
  buildDecisionDeltaComparison,
  buildDeskConsumptionSeries,
} from "@/lib/view-models";

export interface ReportHeading {
  level: number;
  text: string;
  id: string;
}

type ValidationVerdict = "PASS" | "WARN" | "FAIL" | "N/A";
type ValidationConfidenceLevel = "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";

interface ValidationAcademicRow {
  model: string;
  rank: number | null;
  score: number | null;
  n: number | null;
  exceptions: number | null;
  expectedRate: number | null;
  actualRate: number | null;
  pUc: number | null;
  pInd: number | null;
  pCc: number | null;
  esTailObservations: number | null;
  esShortfallRatio: number | null;
  esBreachRate: number | null;
  trafficLight: string | null;
  verdict: ValidationVerdict;
}

interface ValidationAcademicHorizonRow {
  horizonDays: number;
  championModel: string | null;
  verdict: ValidationVerdict;
  passRate: number | null;
  passCount: number;
  warnCount: number;
  failCount: number;
  totalPoints: number;
  confidenceScore: number | null;
  confidenceLevel: ValidationConfidenceLevel;
  confidenceReason: string | null;
}

interface ValidationAcademicBlock {
  available: boolean;
  threshold: number;
  globalVerdict: ValidationVerdict;
  championModel: string | null;
  championVerdict: ValidationVerdict;
  championTrafficLight: string | null;
  championEsTailObservations: number | null;
  championEsShortfallRatio: number | null;
  championEsBreachRate: number | null;
  passRate: number | null;
  passCount: number;
  warnCount: number;
  failCount: number;
  totalPoints: number;
  coverageFailCount: number;
  independenceFailCount: number;
  conditionalFailCount: number;
  confidenceScore: number | null;
  confidenceLevel: ValidationConfidenceLevel;
  confidenceReason: string | null;
  horizonRows: ValidationAcademicHorizonRow[];
  rows: ValidationAcademicRow[];
}

const DEFAULT_P_VALUE_THRESHOLD = 0.05;

function toRecord(value: unknown): Record<string, unknown> {
  return value != null && typeof value === "object" ? (value as Record<string, unknown>) : {};
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

function inferValidationVerdict({
  pUc,
  pInd,
  pCc,
  threshold,
}: {
  pUc: number | null;
  pInd: number | null;
  pCc: number | null;
  threshold: number;
}): ValidationVerdict {
  if (pUc == null && pInd == null && pCc == null) {
    return "N/A";
  }
  if ((pUc != null && pUc < threshold) || (pCc != null && pCc < threshold)) {
    return "FAIL";
  }
  if (pInd != null && pInd < threshold) {
    return "WARN";
  }
  return "PASS";
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

function normalizeConfidenceLevel(value: unknown): ValidationConfidenceLevel {
  const normalized = toUpper(value);
  if (normalized === "HIGH" || normalized === "MEDIUM" || normalized === "LOW") {
    return normalized;
  }
  return "UNKNOWN";
}

function buildValidationAcademicBlock(args: {
  comparison: unknown;
  validation: unknown;
  selectedModel: string;
}): ValidationAcademicBlock {
  const comparison = toRecord(args.comparison);
  const validation = toRecord(args.validation);
  const validationSummary = toRecord(validation.summary);

  const validationSurface = toRecord(comparison.validation_surface);
  const governance = toRecord(validationSurface.governance_summary);
  const horizonGovernance = toRecord(validationSurface.horizon_governance);
  const horizonPayload = toRecord(horizonGovernance.horizons);
  const statusCounts = toRecord(governance.status_counts);

  const threshold = toNumber(governance.pvalue_threshold) ?? DEFAULT_P_VALUE_THRESHOLD;
  const passCount = toInteger(statusCounts.PASS) ?? 0;
  const warnCount = toInteger(statusCounts.WARN) ?? 0;
  const failCount = toInteger(statusCounts.FAIL) ?? 0;
  const totalPoints = toInteger(governance.total_points) ?? passCount + warnCount + failCount;
  const passRate = toNumber(governance.pass_rate);
  const coverageFailCount = toInteger(governance.coverage_fail_count) ?? 0;
  const independenceFailCount = toInteger(governance.independence_fail_count) ?? 0;
  const conditionalFailCount = toInteger(governance.conditional_fail_count) ?? 0;
  const confidenceScore = toNumber(governance.confidence_score);
  const confidenceLevel = normalizeConfidenceLevel(governance.confidence_level);
  const confidenceReasonRaw = typeof governance.confidence_reason === "string"
    ? governance.confidence_reason.trim()
    : "";
  const confidenceReason = confidenceReasonRaw || null;
  const globalVerdict =
    normalizeValidationVerdict(governance.verdict)
    ?? normalizeValidationVerdict(horizonGovernance.overall_verdict)
    ?? (failCount > 0 ? "FAIL" : warnCount > 0 ? "WARN" : passCount > 0 ? "PASS" : "N/A");

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
  const horizonRows: ValidationAcademicHorizonRow[] = selectedHorizonOrder.map((horizonDays) => {
    const row = toRecord(horizonPayload[`h${horizonDays}`]);
    const rowStatusCounts = toRecord(row.status_counts);
    const rowPassCount = toInteger(rowStatusCounts.PASS) ?? 0;
    const rowWarnCount = toInteger(rowStatusCounts.WARN) ?? 0;
    const rowFailCount = toInteger(rowStatusCounts.FAIL) ?? 0;
    const rowTotalPoints = toInteger(row.total_points) ?? rowPassCount + rowWarnCount + rowFailCount;
    const rowPassRate = toNumber(row.pass_rate) ?? (rowTotalPoints > 0 ? rowPassCount / rowTotalPoints : null);
    const rowVerdict =
      normalizeValidationVerdict(row.verdict)
      ?? (rowFailCount > 0 ? "FAIL" : rowWarnCount > 0 ? "WARN" : rowPassCount > 0 ? "PASS" : "N/A");
    const rowConfidenceLevel = normalizeConfidenceLevel(row.confidence_level);
    const rowConfidenceScore = toNumber(row.confidence_score);
    const rowConfidenceReasonRaw = typeof row.confidence_reason === "string" ? row.confidence_reason.trim() : "";
    return {
      horizonDays,
      championModel: toUpper(row.champion_model),
      verdict: rowVerdict,
      passRate: rowPassRate,
      passCount: rowPassCount,
      warnCount: rowWarnCount,
      failCount: rowFailCount,
      totalPoints: rowTotalPoints,
      confidenceScore: rowConfidenceScore,
      confidenceLevel: rowConfidenceLevel,
      confidenceReason: rowConfidenceReasonRaw || null,
    };
  });

  const ranking = Array.isArray(comparison.ranking) ? comparison.ranking : [];
  const modelsPayload = toRecord(validationSummary.models ?? validationSummary.model_results ?? {});

  const rows: ValidationAcademicRow[] = ranking.map((row) => {
    const item = toRecord(row);
    const model = String(item.model ?? "").toLowerCase();
    const modelSummary = toRecord(modelsPayload[model]);
    const pUc = toNumber(item.p_uc ?? modelSummary.p_uc);
    const pInd = toNumber(item.p_ind ?? modelSummary.p_ind);
    const pCc = toNumber(item.p_cc ?? modelSummary.p_cc);
    return {
      model: model.toUpperCase(),
      rank: toInteger(item.rank),
      score: toNumber(item.score),
      n: toInteger(modelSummary.n),
      exceptions: toInteger(item.exceptions ?? modelSummary.exceptions),
      expectedRate: toNumber(item.expected_rate ?? modelSummary.expected_rate),
      actualRate: toNumber(item.actual_rate ?? modelSummary.actual_rate),
      pUc,
      pInd,
      pCc,
      esTailObservations: toInteger(modelSummary.es_tail_observations),
      esShortfallRatio: toNumber(modelSummary.es_shortfall_ratio),
      esBreachRate: toNumber(modelSummary.es_breach_rate),
      trafficLight: toUpper(item.traffic_light ?? modelSummary.traffic_light),
      verdict: inferValidationVerdict({ pUc, pInd, pCc, threshold }),
    };
  });

  const selected = rows.find((row) => row.model.toLowerCase() === args.selectedModel.toLowerCase()) ?? rows[0] ?? null;

  return {
    available: rows.length > 0 || totalPoints > 0,
    threshold,
    globalVerdict,
    championModel: selected?.model ?? null,
    championVerdict: selected?.verdict ?? "N/A",
    championTrafficLight: selected?.trafficLight ?? null,
    championEsTailObservations: selected?.esTailObservations ?? null,
    championEsShortfallRatio: selected?.esShortfallRatio ?? null,
    championEsBreachRate: selected?.esBreachRate ?? null,
    passRate: passRate ?? (totalPoints > 0 ? passCount / totalPoints : null),
    passCount,
    warnCount,
    failCount,
    totalPoints,
    coverageFailCount,
    independenceFailCount,
    conditionalFailCount,
    confidenceScore,
    confidenceLevel,
    confidenceReason,
    horizonRows,
    rows,
  };
}

export function normalizeReportContent(content: string) {
  return content
    .replace(/^#\s+Risk Report.*$/m, "# Daily Risk Report")
    .replace(/^[*-]\s*Generated.*(?:\r?\n)?/gm, "")
    .replace(/^[*-]\s*Compare CSV.*(?:\r?\n)?/gm, "")
    .replace(/^Generated.*(?:\r?\n)?/gm, "")
    .replace(/^Compare CSV.*(?:\r?\n)?/gm, "")
    .replace(/^## Charts[\s\S]*?(?=\n##\s|\s*$)/m, "")
    .trim();
}

export function extractMarkdownHeadings(markdown: string): ReportHeading[] {
  return markdown
    .split(/\r?\n/)
    .map((line) => line.match(/^(#{1,4})\s+(.+)$/))
    .filter((value): value is RegExpMatchArray => value !== null)
    .map((match) => ({
      level: match[1].length,
      text: match[2].trim(),
      id: slugifyHeading(match[2]),
    }));
}

function executiveSignal(
  label: string,
  value: string,
  copy: string,
  tone: "accent" | "warning" | "success" | "neutral" = "neutral",
) {
  return { label, value, copy, tone };
}

export async function loadDeskReportViewModel(
  portfolioSlug?: string,
  options?: { liveState?: MT5LiveStateResponse | null; accountId?: string | null },
) {
  const health = await api.safeHealth();
  const resolvedPortfolio = portfolioSlug ?? health.portfolio_slug;
  const accountId = String(options?.accountId ?? "").trim() || undefined;
  const liveState =
    options?.liveState !== undefined
      ? options.liveState
      : await api.mt5LiveState(resolvedPortfolio, {
        detailLevel: "summary",
        accountId,
      }).catch(() => null);
  const persistedLiveSnapshot = await api
    .latestSnapshot(resolvedPortfolio, "mt5_live_bridge")
    .catch(() => null);
  const preferredSnapshotSource =
    liveState?.risk_summary?.source
    ?? (persistedLiveSnapshot ? "mt5_live_bridge" : "auto");
  const [report, decisions, capitalHistory, audit, comparison, validation, frame, desk] = await Promise.all([
    api.latestReport(resolvedPortfolio, undefined, accountId).catch(() => null),
    api.reportDecisionHistory(resolvedPortfolio, 12, accountId).catch(() => []),
    api.reportCapitalHistory(resolvedPortfolio, 8, preferredSnapshotSource).catch(() => []),
    api.recentAudit(resolvedPortfolio, 16).catch(() => []),
    api.latestModelComparison(resolvedPortfolio).catch(() => null),
    api.latestValidation(resolvedPortfolio).catch(() => null),
    api.latestBacktestFrame(resolvedPortfolio, 260).catch(() => null),
    api.deskOverview(health.desk_slug ?? "main").catch(() => null),
  ]);
  const capital = liveState?.capital_usage ?? (await api.latestCapital(resolvedPortfolio).catch(() => null));
  let snapshot =
    preferredSnapshotSource === "mt5_live_bridge"
      ? persistedLiveSnapshot
      : await api.latestSnapshot(resolvedPortfolio, preferredSnapshotSource).catch(() => null);
  if (snapshot == null) {
    snapshot = await api.latestSnapshot(resolvedPortfolio, "auto").catch(() => null);
  }

  const selectedModel =
    liveState?.risk_summary?.reference_model ??
    comparison?.champion_model ??
    capital?.reference_model ??
    "hist";
  const validationAcademic = buildValidationAcademicBlock({
    comparison,
    validation,
    selectedModel,
  });
  const payload = (snapshot?.payload ?? {}) as {
    var?: Record<string, number>;
    es?: Record<string, number>;
  };
  const varValue = Number(
    liveState?.risk_summary?.var?.[selectedModel] ??
      Object.values(liveState?.risk_summary?.var ?? {})[0] ??
      payload.var?.[selectedModel] ??
      Object.values(payload.var ?? {})[0] ??
      0,
  );
  const esValue = Number(
    liveState?.risk_summary?.es?.[selectedModel] ??
      Object.values(liveState?.risk_summary?.es ?? {})[0] ??
      payload.es?.[selectedModel] ??
      Object.values(payload.es ?? {})[0] ??
      0,
  );
  const fillRatio = averageDecisionFillRatio(decisions);
  const decisionSizeSeries = buildDecisionDeltaComparison(decisions);
  const capitalSeries = buildCapitalHistorySeries(capitalHistory);
  const deskSeries = desk ? buildDeskConsumptionSeries(desk) : [];
  const normalizedReportContent = report ? normalizeReportContent(report.content) : "";
  const headings = report ? extractMarkdownHeadings(normalizedReportContent) : [];
  const latestReportEvent = audit.find((event) => event.action_type === "report.run");

  const executiveSummary = [
    executiveSignal(
      `VaR / ${selectedModel.toUpperCase()}`,
      formatCurrency(varValue),
      liveState?.risk_summary?.latest_observation
        ? `Live selected-model risk level from the MT5 bridge sample ending ${formatTimestamp(liveState.risk_summary.latest_observation)}.`
        : "Current selected model risk level for the latest persisted portfolio snapshot.",
      "accent",
    ),
    executiveSignal(
      `ES / ${selectedModel.toUpperCase()}`,
      formatCurrency(esValue),
      liveState?.risk_summary
        ? "Tail-loss expectation derived from the live bridge risk summary."
        : "Tail-loss expectation carried into the report baseline.",
      "warning",
    ),
    executiveSignal(
      "Capital headroom",
      capital ? formatPercent(capital.headroom_ratio ?? 0, 0) : "n/a",
      capital
        ? `Remaining ${formatCurrency(capital.total_capital_remaining_eur)} before hitting current budget boundaries.`
        : "No persisted capital snapshot yet.",
      "success",
    ),
    executiveSignal(
      "Average fill ratio",
      fillRatio == null ? "n/a" : formatPercent(fillRatio, 0),
      "Advisory decisions approved versus requested exposure changes over recent runs.",
      "neutral",
    ),
  ];

  const narrativeSummary = [
    `Champion model: ${(comparison?.champion_model ?? selectedModel).toUpperCase()} with a score gap of ${
      comparison?.score_gap != null ? comparison.score_gap.toFixed(1) : "n/a"
    }.`,
    capital
      ? `Capital posture remains ${capital.status.toLowerCase()} with ${formatCurrency(
          capital.total_capital_consumed_eur,
        )} consumed from ${formatCurrency(capital.total_capital_budget_eur)}.`
      : "Capital posture is not yet available.",
    decisions.length
      ? `${decisions.length} recent decisions are persisted, with an average fill ratio of ${
          fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)
        }.`
      : "No recent decisions are persisted yet.",
    liveState?.reconciliation
      ? `${liveState.reconciliation.manual_event_count} manual MT5 event(s) and ${liveState.reconciliation.unmatched_execution_count} unmatched execution attempt(s) are currently visible from the live bridge.`
      : "Live reconciliation telemetry is not currently available.",
  ];

  return {
    health,
    liveState,
    report,
    decisions,
    capitalHistory,
    audit,
    capital,
    comparison,
    validation,
    validationAcademic,
    snapshot,
    frame,
    desk,
    resolvedPortfolio,
    selectedModel,
    varValue,
    esValue,
    fillRatio,
    decisionSizeSeries,
    capitalSeries,
    deskSeries,
    headings,
    latestReportEvent,
    normalizedReportContent,
    executiveSummary,
    narrativeSummary,
    meta: {
      reportTimestamp: latestReportEvent?.created_at
        ? formatTimestamp(latestReportEvent.created_at)
        : report
          ? "report available"
          : "report pending",
      reportPath: report?.report_markdown ?? "",
      chartCount: report?.chart_paths?.length ?? 0,
      baseCurrency: capital?.base_currency ?? "EUR",
      preferredSnapshotSource,
    },
    derived: {
      backtestSeries: frame ? buildBacktestSeries(frame) : [],
    },
  };
}

export type DeskReportViewModel = Awaited<ReturnType<typeof loadDeskReportViewModel>>;

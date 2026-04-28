import { api } from "@/lib/api/client";
import type { AuditEventResponse, MT5LiveStateResponse, ReportContentResponse } from "@/lib/api/types";
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

interface ResolvedReportContractMetric {
  value: number | null;
  display: string | null;
  asOfUtc: string | null;
}

export interface ResolvedReportContract {
  version: string | null;
  timezone: string;
  generatedAtUtc: string | null;
  selectedModel: string | null;
  snapshotSource: string | null;
  snapshotTimestampUtc: string | null;
  moneyDecimals: number;
  percentDecimals: number;
  metrics: {
    var: ResolvedReportContractMetric;
    es: ResolvedReportContractMetric;
    pnl: ResolvedReportContractMetric;
  };
}

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
  effectivePoints: number;
  insufficientSampleCount: number;
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
const INITIAL_REPORT_DECISION_LIMIT = 12;
const INITIAL_REPORT_CAPITAL_LIMIT = 8;
const INITIAL_REPORT_AUDIT_LIMIT = 16;
const EXPANDED_REPORT_HISTORY_LIMIT = 200;

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

function normalizeDecimalSetting(value: unknown, fallback: number) {
  const parsed = toInteger(value);
  if (parsed == null || parsed < 0 || parsed > 8) {
    return fallback;
  }
  return parsed;
}

function toUtcMillis(value: string | null | undefined): number | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  const parsed = Date.parse(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeSource(value: string | null | undefined): string | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  return value.trim().toLowerCase();
}

function selectLatestReportRunEvent(
  events: AuditEventResponse[],
  report: ReportContentResponse | null,
): AuditEventResponse | undefined {
  const reportPath = typeof report?.report_markdown === "string" ? report.report_markdown.trim() : "";
  const reportId = report?.report_id == null ? null : Number(report.report_id);
  const reportAccountId = typeof report?.account_id === "string" ? report.account_id.trim() : "";
  const hasReportIdentity = Boolean(reportPath) || reportId != null;

  const reportRunEvents = events.filter((event) => event.action_type === "report.run");
  const matching = reportRunEvents.find((event) => {
    const payload = toRecord(event.payload);
    const payloadReportPath =
      typeof payload.report_markdown === "string" ? payload.report_markdown.trim() : "";
    if (reportPath && payloadReportPath && payloadReportPath === reportPath) {
      return true;
    }

    const payloadReportId = toNumber(payload.report_id);
    if (reportId != null && payloadReportId != null && Number(payloadReportId) === reportId) {
      return true;
    }

    if (reportAccountId) {
      const payloadAccountId = typeof payload.account_id === "string" ? payload.account_id.trim() : "";
      if (payloadAccountId && payloadAccountId === reportAccountId && reportPath && payloadReportPath) {
        return payloadReportPath.endsWith(reportPath.split("/").pop() ?? reportPath);
      }
    }
    return false;
  });

  if (matching) {
    return matching;
  }
  return hasReportIdentity ? undefined : reportRunEvents[0];
}

function selectReportCutoffTimestamp(
  reportContract: ResolvedReportContract | null,
  latestReportEvent: Record<string, unknown> | undefined,
): string | null {
  const fromContract = reportContract?.generatedAtUtc ?? null;
  if (fromContract) {
    return fromContract;
  }
  const fromEvent = typeof latestReportEvent?.created_at === "string" && latestReportEvent.created_at.trim()
    ? latestReportEvent.created_at.trim()
    : null;
  if (fromEvent) {
    return fromEvent;
  }
  return reportContract?.snapshotTimestampUtc ?? null;
}

function isAtOrBeforeCutoff(
  timestamp: string | null | undefined,
  cutoffMillis: number | null,
): boolean {
  if (cutoffMillis == null) {
    return true;
  }
  const itemMillis = toUtcMillis(timestamp);
  if (itemMillis == null) {
    return false;
  }
  return itemMillis <= cutoffMillis;
}

function isPlaceholderMetricDisplay(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return true;
  }
  if (normalized === "-" || normalized === "--") {
    return true;
  }
  const compact = normalized.replace(/\s+/g, "");
  return compact === "n/a" || compact === "na" || compact === "null" || compact === "none";
}

function resolveContractMetric(value: unknown): ResolvedReportContractMetric {
  const record = toRecord(value);
  const display = typeof record.display === "string" && record.display.trim()
    ? record.display.trim()
    : null;
  const normalizedDisplay = display != null && isPlaceholderMetricDisplay(display)
    ? null
    : display;
  const asOfUtc = typeof record.as_of_utc === "string" && record.as_of_utc.trim()
    ? record.as_of_utc.trim()
    : null;
  return {
    value: toNumber(record.value),
    display: normalizedDisplay,
    asOfUtc,
  };
}

export function resolveReportContract(report: Pick<ReportContentResponse, "report_contract"> | null | undefined): ResolvedReportContract | null {
  const contract = toRecord(report?.report_contract);
  if (!Object.keys(contract).length) {
    return null;
  }
  const rounding = toRecord(contract.rounding);
  const metrics = toRecord(contract.metrics);
  return {
    version: typeof contract.version === "string" && contract.version.trim() ? contract.version.trim() : null,
    timezone: typeof contract.timezone === "string" && contract.timezone.trim() ? contract.timezone.trim() : "UTC",
    generatedAtUtc: typeof contract.generated_at_utc === "string" && contract.generated_at_utc.trim()
      ? contract.generated_at_utc.trim()
      : null,
    selectedModel: typeof contract.selected_model === "string" && contract.selected_model.trim()
      ? contract.selected_model.trim()
      : null,
    snapshotSource: typeof contract.snapshot_source === "string" && contract.snapshot_source.trim()
      ? contract.snapshot_source.trim()
      : null,
    snapshotTimestampUtc: typeof contract.snapshot_timestamp_utc === "string" && contract.snapshot_timestamp_utc.trim()
      ? contract.snapshot_timestamp_utc.trim()
      : null,
    moneyDecimals: normalizeDecimalSetting(rounding.money_decimals, 2),
    percentDecimals: normalizeDecimalSetting(rounding.percent_decimals, 1),
    metrics: {
      var: resolveContractMetric(metrics.var),
      es: resolveContractMetric(metrics.es),
      pnl: resolveContractMetric(metrics.pnl),
    },
  };
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
  const effectivePoints = toInteger(governance.effective_points) ?? 0;
  const insufficientSampleCount = toInteger(governance.insufficient_sample_count) ?? Math.max(totalPoints - effectivePoints, 0);
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
    effectivePoints,
    insufficientSampleCount,
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

const TABLE_SEPARATOR_PATTERN = /^\|\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/;

function insertBlankLineBeforeTablesAfterBullets(lines: string[]) {
  const output: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("|")) {
      const previousNonEmpty = [...output].reverse().find((item) => item.trim().length > 0) ?? null;
      if (previousNonEmpty?.trim().startsWith("- ") && (output.length === 0 || output[output.length - 1].trim() !== "")) {
        output.push("");
      }
    }
    output.push(line);
  }
  return output;
}

function moveEsTailNoteBelowTable(lines: string[]) {
  const output: string[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    output.push(line);
    if (!TABLE_SEPARATOR_PATTERN.test(line.trim())) {
      index += 1;
      continue;
    }

    let cursor = index + 1;
    while (cursor < lines.length && lines[cursor].trim() === "") {
      cursor += 1;
    }

    let deferredNote: string | null = null;
    if (cursor < lines.length && /^_ES tail diagnostics/i.test(lines[cursor].trim())) {
      deferredNote = lines[cursor];
      cursor += 1;
      while (cursor < lines.length && lines[cursor].trim() === "") {
        cursor += 1;
      }
    }

    while (cursor < lines.length && lines[cursor].trim().startsWith("|")) {
      output.push(lines[cursor]);
      cursor += 1;
    }

    if (deferredNote) {
      if (output[output.length - 1]?.trim() !== "") {
        output.push("");
      }
      output.push(deferredNote);
      output.push("");
    }
    index = cursor;
  }
  return output;
}

function repairLegacyReportTables(markdown: string) {
  const lines = markdown.split(/\r?\n/);
  const withSpacing = insertBlankLineBeforeTablesAfterBullets(lines);
  const withMovedNote = moveEsTailNoteBelowTable(withSpacing);
  return withMovedNote.join("\n");
}

export function normalizeReportContent(content: string) {
  const normalized = content
    .replace(/^#\s+Risk Report.*$/m, "# Daily Risk Report")
    .replace(/^[*-]\s*Generated.*(?:\r?\n)?/gm, "")
    .replace(/^[*-]\s*Compare CSV.*(?:\r?\n)?/gm, "")
    .replace(/^Generated.*(?:\r?\n)?/gm, "")
    .replace(/^Compare CSV.*(?:\r?\n)?/gm, "")
    .replace(/^## Charts[\s\S]*?(?=\n##\s|\s*$)/m, "");
  return repairLegacyReportTables(normalized).trim();
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
  options?: {
    liveState?: MT5LiveStateResponse | null;
    accountId?: string | null;
    freezeToReportScope?: boolean;
  },
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
  const [report, initialDecisionHistory, initialCapitalHistory, initialAuditHistory, desk] = await Promise.all([
    api.latestReport(resolvedPortfolio, undefined, accountId).catch(() => null),
    api.reportDecisionHistory(resolvedPortfolio, INITIAL_REPORT_DECISION_LIMIT, accountId).catch(() => []),
    api.reportCapitalHistory(resolvedPortfolio, INITIAL_REPORT_CAPITAL_LIMIT, preferredSnapshotSource).catch(() => []),
    api.recentAudit(resolvedPortfolio, INITIAL_REPORT_AUDIT_LIMIT).catch(() => []),
    api.deskOverview(health.desk_slug ?? "main").catch(() => null),
  ]);
  const reportId = typeof report?.report_id === "number" && Number.isFinite(report.report_id)
    ? report.report_id
    : undefined;
  const [comparison, validation, frame] = await Promise.all([
    api.latestModelComparison(resolvedPortfolio, reportId).catch(() => null),
    api.latestValidation(resolvedPortfolio, reportId).catch(() => null),
    api.latestBacktestFrame(resolvedPortfolio, 260, reportId).catch(() => null),
  ]);
  let decisionHistoryRaw = initialDecisionHistory;
  let auditRaw = initialAuditHistory;
  const reportContract = resolveReportContract(report);
  const strictReportScope = Boolean(options?.freezeToReportScope ?? true) && report != null;
  let latestReportEvent = selectLatestReportRunEvent(auditRaw, report);
  if (strictReportScope && latestReportEvent == null && auditRaw.length >= INITIAL_REPORT_AUDIT_LIMIT) {
    auditRaw = await api
      .recentAudit(resolvedPortfolio, EXPANDED_REPORT_HISTORY_LIMIT)
      .catch(() => auditRaw);
    latestReportEvent = selectLatestReportRunEvent(auditRaw, report);
  }
  let snapshot =
    preferredSnapshotSource === "mt5_live_bridge"
      ? persistedLiveSnapshot
      : await api.latestSnapshot(resolvedPortfolio, preferredSnapshotSource).catch(() => null);
  if (snapshot == null) {
    snapshot = await api.latestSnapshot(resolvedPortfolio, "auto").catch(() => null);
  }

  const reportSnapshotSource = strictReportScope
    ? (reportContract?.snapshotSource ?? preferredSnapshotSource)
    : preferredSnapshotSource;

  let capitalHistory = initialCapitalHistory;
  if (
    strictReportScope
    && reportContract?.snapshotSource
    && normalizeSource(reportContract.snapshotSource) !== normalizeSource(preferredSnapshotSource)
  ) {
    capitalHistory = await api
      .reportCapitalHistory(resolvedPortfolio, INITIAL_REPORT_CAPITAL_LIMIT, reportContract.snapshotSource)
      .catch(() => initialCapitalHistory);
  }

  if (strictReportScope && normalizeSource(reportSnapshotSource) !== normalizeSource(preferredSnapshotSource)) {
    snapshot = reportSnapshotSource === "mt5_live_bridge"
      ? persistedLiveSnapshot
      : await api.latestSnapshot(resolvedPortfolio, reportSnapshotSource).catch(() => snapshot);
  }

  const reportCutoffTimestamp = strictReportScope
    ? selectReportCutoffTimestamp(reportContract, latestReportEvent)
    : null;
  const reportCutoffMillis = toUtcMillis(reportCutoffTimestamp);
  const reportSourceFilter = strictReportScope ? normalizeSource(reportContract?.snapshotSource ?? null) : null;
  const filterReportScopedDecisions = (rows: typeof decisionHistoryRaw) => strictReportScope
    ? rows.filter((decision) =>
      isAtOrBeforeCutoff(decision.time_utc ?? decision.created_at ?? null, reportCutoffMillis))
    : rows;
  const filterReportScopedAudit = (rows: typeof auditRaw) => strictReportScope
    ? rows.filter((event) => isAtOrBeforeCutoff(event.created_at ?? null, reportCutoffMillis))
    : rows;
  const filterReportScopedCapitalHistory = (rows: typeof capitalHistory) => strictReportScope
    ? rows.filter((entry) => {
      const entrySource = normalizeSource(
        (entry.snapshot_source ?? entry.source ?? null) as string | null | undefined,
      );
      if (reportSourceFilter != null && entrySource != null && entrySource !== reportSourceFilter) {
        return false;
      }
      return isAtOrBeforeCutoff(entry.snapshot_timestamp ?? entry.created_at ?? null, reportCutoffMillis);
    })
    : rows;
  let decisions = filterReportScopedDecisions(decisionHistoryRaw);
  let audit = filterReportScopedAudit(auditRaw);
  let scopedCapitalHistory = filterReportScopedCapitalHistory(capitalHistory);

  if (
    strictReportScope
    && reportCutoffMillis != null
    && decisions.length === 0
    && decisionHistoryRaw.length >= INITIAL_REPORT_DECISION_LIMIT
  ) {
    decisionHistoryRaw = await api
      .reportDecisionHistory(resolvedPortfolio, EXPANDED_REPORT_HISTORY_LIMIT, accountId)
      .catch(() => decisionHistoryRaw);
    decisions = filterReportScopedDecisions(decisionHistoryRaw);
  }

  if (
    strictReportScope
    && reportCutoffMillis != null
    && audit.length === 0
    && auditRaw.length >= INITIAL_REPORT_AUDIT_LIMIT
  ) {
    auditRaw = await api
      .recentAudit(resolvedPortfolio, EXPANDED_REPORT_HISTORY_LIMIT)
      .catch(() => auditRaw);
    audit = filterReportScopedAudit(auditRaw);
  }

  if (
    strictReportScope
    && reportCutoffMillis != null
    && scopedCapitalHistory.length === 0
    && capitalHistory.length >= INITIAL_REPORT_CAPITAL_LIMIT
  ) {
    capitalHistory = await api
      .reportCapitalHistory(resolvedPortfolio, EXPANDED_REPORT_HISTORY_LIMIT, reportSnapshotSource)
      .catch(() => capitalHistory);
    scopedCapitalHistory = filterReportScopedCapitalHistory(capitalHistory);
  }

  const sortedCapitalHistory = scopedCapitalHistory
    .slice()
    .sort((left, right) => {
      const leftMs = toUtcMillis(left.snapshot_timestamp ?? left.created_at ?? null) ?? Number.NEGATIVE_INFINITY;
      const rightMs = toUtcMillis(right.snapshot_timestamp ?? right.created_at ?? null) ?? Number.NEGATIVE_INFINITY;
      return rightMs - leftMs;
    });
  const reportScopedCapital = sortedCapitalHistory[0] ?? null;
  const liveOrLatestCapital = liveState?.capital_usage ?? (await api.latestCapital(resolvedPortfolio).catch(() => null));
  const capital = strictReportScope ? reportScopedCapital : liveOrLatestCapital;

  const payload = (snapshot?.payload ?? {}) as {
    var?: Record<string, number>;
    es?: Record<string, number>;
  };
  const snapshotModel =
    Object.keys(payload.var ?? {}).find((model) => Boolean(model))?.toLowerCase()
    ?? Object.keys(payload.es ?? {}).find((model) => Boolean(model))?.toLowerCase()
    ?? null;
  const contractModel = strictReportScope
    ? (reportContract?.selectedModel?.toLowerCase() ?? null)
    : null;
  const selectedModel =
    contractModel ??
    snapshotModel ??
    comparison?.champion_model ??
    capital?.reference_model ??
    liveState?.risk_summary?.reference_model ??
    "hist";
  const validationAcademic = buildValidationAcademicBlock({
    comparison,
    validation,
    selectedModel,
  });
  const contractVar = reportContract?.metrics.var.value ?? null;
  const contractEs = reportContract?.metrics.es.value ?? null;
  const fallbackVar = Object.values(payload.var ?? {})
    .map((value) => toNumber(value))
    .find((value): value is number => value != null) ?? null;
  const fallbackEs = Object.values(payload.es ?? {})
    .map((value) => toNumber(value))
    .find((value): value is number => value != null) ?? null;
  const varValueRaw = strictReportScope
    ? (contractVar ?? toNumber(payload.var?.[selectedModel]) ?? fallbackVar)
    : (
      toNumber(liveState?.risk_summary?.var?.[selectedModel])
      ?? toNumber(Object.values(liveState?.risk_summary?.var ?? {})[0])
      ?? toNumber(payload.var?.[selectedModel])
      ?? fallbackVar
    );
  const esValueRaw = strictReportScope
    ? (contractEs ?? toNumber(payload.es?.[selectedModel]) ?? fallbackEs)
    : (
      toNumber(liveState?.risk_summary?.es?.[selectedModel])
      ?? toNumber(Object.values(liveState?.risk_summary?.es ?? {})[0])
      ?? toNumber(payload.es?.[selectedModel])
      ?? fallbackEs
    );
  const varValue = Number(varValueRaw ?? 0);
  const esValue = Number(esValueRaw ?? 0);
  const backtestRows = Array.isArray(frame?.rows) ? frame.rows : [];
  let latestBacktestPnl: number | null = null;
  let latestBacktestPnlTimestamp: string | null = null;
  for (let index = backtestRows.length - 1; index >= 0; index -= 1) {
    const row = toRecord(backtestRows[index]);
    const candidatePnl = toNumber(row.pnl);
    if (candidatePnl == null) {
      continue;
    }
    latestBacktestPnl = candidatePnl;
    const candidateTimestamp = [
      row.time_utc,
      row.timestamp,
      row.time,
      row.date,
      row.label,
    ].find((value) => typeof value === "string" && value.trim()) as string | undefined;
    latestBacktestPnlTimestamp = candidateTimestamp?.trim() ?? null;
    break;
  }
  const contractPnl = reportContract?.metrics.pnl.value ?? null;
  const pnlValueRaw = strictReportScope ? contractPnl : latestBacktestPnl;
  const pnlValue = Number(pnlValueRaw ?? 0);
  const pnlTimestamp = reportContract?.metrics.pnl.asOfUtc ?? (strictReportScope ? null : latestBacktestPnlTimestamp);
  const moneyDecimals = reportContract?.moneyDecimals ?? 2;
  const varDisplay = strictReportScope
    ? (reportContract?.metrics.var.display ?? (varValueRaw == null ? "n/a" : formatCurrency(varValue, moneyDecimals)))
    : (varValueRaw == null ? "n/a" : formatCurrency(varValue, moneyDecimals));
  const esDisplay = strictReportScope
    ? (reportContract?.metrics.es.display ?? (esValueRaw == null ? "n/a" : formatCurrency(esValue, moneyDecimals)))
    : (esValueRaw == null ? "n/a" : formatCurrency(esValue, moneyDecimals));
  const pnlDisplay = strictReportScope
    ? (reportContract?.metrics.pnl.display ?? (pnlValueRaw == null ? "n/a" : formatCurrency(pnlValue, moneyDecimals)))
    : (pnlValueRaw == null ? "n/a" : formatCurrency(pnlValue, moneyDecimals));
  const fillRatio = averageDecisionFillRatio(decisions);
  const decisionSizeSeries = buildDecisionDeltaComparison(decisions);
  const capitalSeries = buildCapitalHistorySeries(scopedCapitalHistory);
  const deskSeries = desk ? buildDeskConsumptionSeries(desk) : [];
  const normalizedReportContent = report ? normalizeReportContent(report.content) : "";
  const headings = report ? extractMarkdownHeadings(normalizedReportContent) : [];

  const executiveSummary = [
    executiveSignal(
      `VaR / ${selectedModel.toUpperCase()}`,
      varDisplay,
      reportContract?.metrics.var.asOfUtc
        ? `Canonical report contract value as of ${formatTimestamp(reportContract.metrics.var.asOfUtc)}.`
        : strictReportScope
          ? "Report-scoped risk level reconstructed from persisted snapshot inputs."
        : liveState?.risk_summary?.latest_observation
          ? `Live selected-model risk level from the MT5 bridge sample ending ${formatTimestamp(liveState.risk_summary.latest_observation)}.`
          : "Current selected model risk level for the latest persisted portfolio snapshot.",
      "accent",
    ),
    executiveSignal(
      `ES / ${selectedModel.toUpperCase()}`,
      esDisplay,
      reportContract?.metrics.es.asOfUtc
        ? `Canonical report contract value as of ${formatTimestamp(reportContract.metrics.es.asOfUtc)}.`
        : strictReportScope
          ? "Report-scoped tail-loss expectation derived from persisted snapshot inputs."
        : liveState?.risk_summary
          ? "Tail-loss expectation derived from the live bridge risk summary."
          : "Tail-loss expectation carried into the report baseline.",
      "warning",
    ),
    executiveSignal(
      "Latest PnL",
      pnlDisplay,
      pnlTimestamp
        ? `Latest available PnL point captured at ${formatTimestamp(pnlTimestamp)}.`
        : "Latest available PnL point from the report dataset.",
      "neutral",
    ),
    executiveSignal(
      "Capital headroom",
      capital ? formatPercent(capital.headroom_ratio ?? 0, 0) : "n/a",
      capital
        ? `${strictReportScope ? "Report snapshot remaining" : "Remaining"} ${formatCurrency(
          capital.total_capital_remaining_eur,
          moneyDecimals,
        )} before hitting current budget boundaries.`
        : "No persisted capital snapshot yet.",
      "success",
    ),
    executiveSignal(
      "Average fill ratio",
      fillRatio == null ? "n/a" : formatPercent(fillRatio, 0),
      strictReportScope
        ? "Advisory decisions approved versus requested exposure changes up to the report cutoff."
        : "Advisory decisions approved versus requested exposure changes over recent runs.",
      "neutral",
    ),
  ];

  const narrativeChampion = strictReportScope ? selectedModel : (comparison?.champion_model ?? selectedModel);
  const narrativeSummary = [
    `Champion model: ${narrativeChampion.toUpperCase()} with a score gap of ${
      strictReportScope ? "n/a" : (comparison?.score_gap != null ? comparison.score_gap.toFixed(1) : "n/a")
    }.`,
    `Latest report PnL point: ${pnlDisplay}${pnlTimestamp ? ` at ${formatTimestamp(pnlTimestamp)}` : ""}.`,
    capital
      ? `Capital posture remains ${capital.status.toLowerCase()} with ${formatCurrency(
          capital.total_capital_consumed_eur,
          moneyDecimals,
        )} consumed from ${formatCurrency(capital.total_capital_budget_eur, moneyDecimals)}.`
      : "Capital posture is not yet available.",
    decisions.length
      ? `${decisions.length} recent decisions are persisted, with an average fill ratio of ${
          fillRatio == null ? "n/a" : formatPercent(fillRatio, 0)
        }.`
      : "No recent decisions are persisted yet.",
    !strictReportScope && liveState?.reconciliation
      ? `${liveState.reconciliation.manual_event_count} manual MT5 event(s) and ${liveState.reconciliation.unmatched_execution_count} unmatched execution attempt(s) are currently visible from the live bridge.`
      : strictReportScope
        ? "Live reconciliation telemetry is intentionally excluded from report-scoped narratives."
        : "Live reconciliation telemetry is not currently available.",
  ];

  return {
    health,
    liveState,
    report,
    decisions,
    capitalHistory: scopedCapitalHistory,
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
    pnlValue,
    varDisplay,
    esDisplay,
    pnlDisplay,
    pnlTimestamp,
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
        : reportContract?.generatedAtUtc
          ? formatTimestamp(reportContract.generatedAtUtc)
        : report
          ? "report available"
          : "report pending",
      reportPath: report?.report_markdown ?? "",
      chartCount: report?.chart_paths?.length ?? 0,
      baseCurrency: capital?.base_currency ?? "EUR",
      preferredSnapshotSource: reportSnapshotSource,
      reportContractVersion: reportContract?.version ?? null,
      reportTimezone: reportContract?.timezone ?? "UTC",
      moneyDecimals,
      percentDecimals: reportContract?.percentDecimals ?? 1,
    },
    derived: {
      backtestSeries: frame ? buildBacktestSeries(frame) : [],
    },
  };
}

export type DeskReportViewModel = Awaited<ReturnType<typeof loadDeskReportViewModel>>;

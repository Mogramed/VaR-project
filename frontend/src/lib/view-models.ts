import type {
  AlertSummary,
  BacktestFrameResponse,
  CapitalUsageSnapshotResponse,
  DealHistoryEntryResponse,
  DeskSnapshotResponse,
  HoldingSnapshotResponse,
  InstrumentDefinitionResponse,
  ModelComparisonResponse,
  OrderHistoryEntryResponse,
  ReconciliationSummaryResponse,
  RiskAttributionResponse,
  RiskDecisionResponse,
} from "@/lib/api/types";
import { dedupePersistedAlerts } from "@/lib/alerts";
import { humanizeIdentifier } from "@/lib/utils";

export interface TimeSeriesPoint {
  label: string;
  value: number;
}

export interface BacktestSeriesPoint {
  label: string;
  pnl?: number;
  var_hist?: number;
  var_garch?: number;
  var_fhs?: number;
  var_alpha?: number;
}

export interface FlatAttributionRow {
  symbol: string;
  assetClass: string;
  model: string;
  position: number;
  componentVar: number;
  componentEs: number;
  incrementalVar: number;
  marginalVar: number;
  contributionPctVar: number;
  status: string;
}

export interface FlatCapitalRow {
  symbol: string;
  targetCapital: number;
  consumedCapital: number;
  remainingCapital: number;
  utilization: number;
  action: string;
  status: string;
}

function normalizeEpochLikeValue(value: number): Date | null {
  if (!Number.isFinite(value)) {
    return null;
  }
  const abs = Math.abs(value);
  const millis =
    abs < 1e11
      ? value * 1000
      : abs < 1e14
        ? value
        : abs < 1e17
          ? value / 1000
          : value / 1_000_000;
  const date = new Date(millis);
  if (Number.isNaN(date.getTime()) || date.getUTCFullYear() < 2000) {
    return null;
  }
  return date;
}

function normalizeBacktestLabel(raw: unknown, index: number): string {
  if (raw == null) {
    return String(index + 1);
  }
  if (typeof raw === "number") {
    const normalized = normalizeEpochLikeValue(raw);
    return normalized ? normalized.toISOString() : String(index + 1);
  }
  const text = String(raw).trim();
  if (!text) {
    return String(index + 1);
  }
  if (/^-?\d+(?:\.\d+)?$/.test(text)) {
    const normalized = normalizeEpochLikeValue(Number(text));
    return normalized ? normalized.toISOString() : String(index + 1);
  }
  const date = new Date(text);
  if (!Number.isNaN(date.getTime()) && date.getUTCFullYear() >= 2000) {
    return date.toISOString();
  }
  return text;
}

function normalizeCapitalHistoryLabel(raw: unknown, index: number): string {
  if (raw == null) {
    return String(index + 1);
  }
  const text = String(raw).trim();
  if (!text) {
    return String(index + 1);
  }
  const parsed = new Date(text);
  if (Number.isNaN(parsed.getTime()) || parsed.getUTCFullYear() < 2000) {
    return String(index + 1);
  }
  const year = String(parsed.getUTCFullYear());
  const month = String(parsed.getUTCMonth() + 1).padStart(2, "0");
  const day = String(parsed.getUTCDate()).padStart(2, "0");
  const hour = String(parsed.getUTCHours()).padStart(2, "0");
  const minute = String(parsed.getUTCMinutes()).padStart(2, "0");
  return `${year}-${month}-${day} ${hour}:${minute}Z`;
}

export function buildDeskConsumptionSeries(desk: DeskSnapshotResponse): TimeSeriesPoint[] {
  return (desk.portfolios ?? []).map((portfolio) => ({
    label: humanizeIdentifier(portfolio.portfolio_name),
    value: portfolio.total_capital_consumed_eur,
  }));
}

export function buildAlertSeverityCounts(alerts: AlertSummary[]) {
  return dedupePersistedAlerts(alerts).reduce<Record<string, number>>((acc, alert) => {
    const key = alert.severity.toLowerCase();
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});
}

export function buildBacktestSeries(frame: BacktestFrameResponse): BacktestSeriesPoint[] {
  return frame.rows.map((row, index) => ({
    label: normalizeBacktestLabel(row.date ?? row.time ?? row.time_utc ?? row.timestamp, index),
    pnl: row.pnl == null ? undefined : Number(row.pnl),
    var_hist: row.var_hist == null ? undefined : Number(row.var_hist),
    var_garch: row.var_garch == null ? undefined : Number(row.var_garch),
    var_fhs: row.var_fhs == null ? undefined : Number(row.var_fhs),
    var_alpha: row.var_alpha == null ? undefined : Number(row.var_alpha),
  }));
}

export function flattenAttribution(
  attribution: RiskAttributionResponse,
  model: string,
): FlatAttributionRow[] {
  const selected = attribution.models[model];
  if (!selected) {
    return [];
  }

    return Object.values(selected.positions)
      .map((position) => ({
        symbol: position.symbol,
        assetClass: position.asset_class,
        model,
        position: position.exposure_base_ccy,
        componentVar: position.component_var,
        componentEs: position.component_es,
      incrementalVar: position.incremental_var,
      marginalVar: position.marginal_var,
      contributionPctVar: Number(position.contribution_pct_var ?? 0),
      status:
        Number(position.contribution_pct_var ?? 0) >= 0.35
          ? "critical"
          : Number(position.contribution_pct_var ?? 0) >= 0.2
            ? "watch"
            : "stable",
    }))
    .sort((left, right) => right.componentVar - left.componentVar);
}

export function flattenCapitalAllocations(
  capital: CapitalUsageSnapshotResponse,
): FlatCapitalRow[] {
  return Object.values(capital.allocations ?? {})
    .map((allocation) => ({
      symbol: allocation.symbol,
      targetCapital: allocation.target_capital_eur,
      consumedCapital: allocation.consumed_capital_eur,
      remainingCapital: allocation.remaining_capital_eur,
      utilization: Number(allocation.utilization ?? 0),
      action: allocation.action,
      status: allocation.status,
    }))
    .sort((left, right) => right.utilization - left.utilization);
}

export function buildCapitalHistorySeries(
  history: CapitalUsageSnapshotResponse[],
): TimeSeriesPoint[] {
  return history
    .slice()
    .reverse()
    .map((item, index) => ({
      label: normalizeCapitalHistoryLabel(item.snapshot_timestamp ?? item.created_at, index),
      value: item.total_capital_consumed_eur,
    }));
}

export function buildModelScoreSeries(
  comparison: ModelComparisonResponse,
): TimeSeriesPoint[] {
  return (comparison.ranking ?? []).map((row) => ({
    label: row.model.toUpperCase(),
    value: row.score,
  }));
}

export function buildDecisionDeltaComparison(decisions: RiskDecisionResponse[]) {
  const rows = decisions.slice(0, 8).reverse();
  return {
    labels: rows.map((decision, index) => decision.symbol || String(index + 1)),
    requested: rows.map((decision) => Math.abs(decision.requested_exposure_change)),
    approved: rows.map((decision) => Math.abs(decision.approved_exposure_change)),
  };
}

export function buildDecisionImpactSeries(
  decisions: RiskDecisionResponse[],
): TimeSeriesPoint[] {
  return decisions
    .slice(0, 8)
    .reverse()
    .map((decision, index) => ({
      label: decision.symbol || String(index + 1),
      value: decision.post_trade.var - decision.pre_trade.var,
    }));
}

export function buildDecisionVerdictCounts(decisions: RiskDecisionResponse[]) {
  return decisions.reduce<Record<string, number>>((acc, decision) => {
    const key = decision.decision.toUpperCase();
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});
}

export function averageDecisionFillRatio(decisions: RiskDecisionResponse[]) {
  const ratios = decisions
    .filter((decision) => Math.abs(decision.requested_exposure_change) > 0)
    .map((decision) =>
      Math.abs(decision.approved_exposure_change / decision.requested_exposure_change),
    );

  if (ratios.length === 0) {
    return null;
  }

  return ratios.reduce((sum, ratio) => sum + ratio, 0) / ratios.length;
}

export function buildAssetClassExposureSeries(
  holdings: HoldingSnapshotResponse[],
): TimeSeriesPoint[] {
  const totals = holdings.reduce<Record<string, number>>((acc, holding) => {
    const key = holding.asset_class || "unknown";
    acc[key] = (acc[key] ?? 0) + Math.abs(Number(holding.signed_exposure_base_ccy ?? 0));
    return acc;
  }, {});

  return Object.entries(totals)
    .map(([label, value]) => ({ label: humanizeIdentifier(label), value }))
    .sort((left, right) => right.value - left.value);
}

export function buildInstrumentClassCounts(
  instruments: InstrumentDefinitionResponse[],
): TimeSeriesPoint[] {
  const totals = instruments.reduce<Record<string, number>>((acc, instrument) => {
    const key = instrument.asset_class || "unknown";
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});

  return Object.entries(totals)
    .map(([label, value]) => ({ label: humanizeIdentifier(label), value }))
    .sort((left, right) => right.value - left.value);
}

export function buildReconciliationDriftSeries(
  summary: ReconciliationSummaryResponse,
): TimeSeriesPoint[] {
  return (summary.mismatches ?? []).map((item) => ({
    label: item.symbol,
    value: Math.abs(item.difference_eur),
  }));
}

export function countManualMt5Events(
  orders: OrderHistoryEntryResponse[],
  deals: DealHistoryEntryResponse[],
) {
  return {
    orders: orders.filter((item) => item.is_manual).length,
    deals: deals.filter((item) => item.is_manual).length,
  };
}

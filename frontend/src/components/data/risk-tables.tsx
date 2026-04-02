"use client";

import type { ColumnDef } from "@tanstack/react-table";
import type {
  AuditEventResponse, DealHistoryEntryResponse, ExecutionFillResponse, ExecutionResultResponse,
  HoldingSnapshotResponse, InstrumentDefinitionResponse, ModelComparisonRow, MT5PendingOrderResponse,
  MT5PositionResponse, OrderHistoryEntryResponse, ReconciliationMismatchResponse, RiskDecisionResponse,
} from "@/lib/api/types";
import type { FlatAttributionRow, FlatCapitalRow } from "@/lib/view-models";
import { DataGrid } from "@/components/data/data-grid";
import { StatusBadge } from "@/components/ui/primitives";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

function tone(v: string) {
  const s = v.toLowerCase();
  if (s.includes("reject") || s.includes("breach") || s.includes("critical") || s.includes("failed")) return "danger" as const;
  if (s.includes("reduce") || s.includes("warn") || s.includes("partial") || s.includes("manual") || s.includes("drift") || s.includes("hold")) return "warning" as const;
  if (s.includes("accept") || s.includes("ok") || s.includes("stable") || s.includes("executed") || s.includes("connected")) return "success" as const;
  if (s.includes("champion")) return "accent" as const;
  return "neutral" as const;
}

export function ModelRankingTable({ rows }: { rows: ModelComparisonRow[] }) {
  const cols: ColumnDef<ModelComparisonRow>[] = [
    { accessorKey: "rank", header: "#" },
    { accessorKey: "model", header: "Model", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.model.toUpperCase()}</span> },
    { accessorKey: "score", header: "Score", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{row.original.score.toFixed(1)}</span> },
    { accessorKey: "actual_rate", header: "Rate", cell: ({ row }) => formatPercent(row.original.actual_rate) },
    { accessorKey: "exceptions", header: "Exc." },
    { accessorKey: "traffic_light", header: "Signal", cell: ({ row }) => row.original.traffic_light ? <StatusBadge label={row.original.traffic_light} tone={tone(row.original.traffic_light)} /> : "—" },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="28rem" />;
}

export function AttributionTable({ rows }: { rows: FlatAttributionRow[] }) {
  const cols: ColumnDef<FlatAttributionRow>[] = [
      { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
      { accessorKey: "assetClass", header: "Asset class", cell: ({ row }) => <span className="text-[var(--color-text-muted)]">{row.original.assetClass}</span> },
      { accessorKey: "position", header: "Position", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.position)}</span> },
      { accessorKey: "componentVar", header: "cVaR", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{formatCurrency(row.original.componentVar)}</span> },
      { accessorKey: "incrementalVar", header: "iVaR", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.incrementalVar)}</span> },
    { accessorKey: "contributionPctVar", header: "Contrib", cell: ({ row }) => formatPercent(row.original.contributionPctVar) },
    { accessorKey: "status", header: "Status", cell: ({ row }) => <StatusBadge label={row.original.status} tone={tone(row.original.status)} /> },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function CapitalAllocationTable({ rows }: { rows: FlatCapitalRow[] }) {
  const cols: ColumnDef<FlatCapitalRow>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "targetCapital", header: "Target", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.targetCapital)}</span> },
    { accessorKey: "consumedCapital", header: "Consumed", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{formatCurrency(row.original.consumedCapital)}</span> },
    { accessorKey: "utilization", header: "Util", cell: ({ row }) => formatPercent(row.original.utilization) },
    { accessorKey: "status", header: "Status", cell: ({ row }) => <StatusBadge label={row.original.status} tone={tone(row.original.status)} /> },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function DecisionHistoryTable({ rows }: { rows: RiskDecisionResponse[] }) {
  const cols: ColumnDef<RiskDecisionResponse>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "decision", header: "Decision", cell: ({ row }) => <StatusBadge label={row.original.decision} tone={tone(row.original.decision)} /> },
    { accessorKey: "requested_exposure_change", header: "Requested", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.requested_exposure_change)}</span> },
    { accessorKey: "approved_exposure_change", header: "Approved", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{formatCurrency(row.original.approved_exposure_change)}</span> },
    { accessorKey: "model_used", header: "Model", cell: ({ row }) => <span className="mono">{row.original.model_used.toUpperCase()}</span> },
    { accessorKey: "created_at", header: "Time", cell: ({ row }) => formatTimestamp(row.original.created_at ?? row.original.time_utc) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function AuditTrailTable({ rows }: { rows: AuditEventResponse[] }) {
  const cols: ColumnDef<AuditEventResponse>[] = [
    { accessorKey: "actor", header: "Actor", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.actor}</span> },
    { accessorKey: "action_type", header: "Action", cell: ({ row }) => <span className="mono">{row.original.action_type}</span> },
    { accessorKey: "object_type", header: "Object", cell: ({ row }) => row.original.object_type ?? "—" },
    { accessorKey: "created_at", header: "Time", cell: ({ row }) => formatTimestamp(row.original.created_at) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="28rem" />;
}

export function MT5PositionsTable({ rows }: { rows: MT5PositionResponse[] }) {
  const cols: ColumnDef<MT5PositionResponse>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "side", header: "Side", cell: ({ row }) => <StatusBadge label={row.original.side} tone={tone(row.original.side)} /> },
    { accessorKey: "volume_lots", header: "Lots", cell: ({ row }) => <span className="mono">{row.original.volume_lots.toFixed(2)}</span> },
    { accessorKey: "signed_exposure_base_ccy", header: "Exposure", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{formatCurrency(row.original.signed_exposure_base_ccy)}</span> },
    { accessorKey: "profit", header: "P&L", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.profit, 2)}</span> },
    { accessorKey: "time_utc", header: "Updated", cell: ({ row }) => formatTimestamp(row.original.time_utc) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function MT5OrdersTable({ rows }: { rows: MT5PendingOrderResponse[] }) {
  const cols: ColumnDef<MT5PendingOrderResponse>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "side", header: "Side", cell: ({ row }) => <StatusBadge label={row.original.side} tone={tone(row.original.side)} /> },
    { accessorKey: "volume_current", header: "Vol", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{row.original.volume_current.toFixed(2)}</span> },
    { accessorKey: "price_open", header: "Price", cell: ({ row }) => <span className="mono">{row.original.price_open.toFixed(4)}</span> },
    { accessorKey: "time_setup_utc", header: "Setup", cell: ({ row }) => formatTimestamp(row.original.time_setup_utc) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="24rem" />;
}

export function HoldingsTable({ rows }: { rows: HoldingSnapshotResponse[] }) {
  const cols: ColumnDef<HoldingSnapshotResponse>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "asset_class", header: "Asset", cell: ({ row }) => humanize(row.original.asset_class) },
    { accessorKey: "side", header: "Side", cell: ({ row }) => <StatusBadge label={row.original.side} tone={tone(row.original.side)} /> },
    { accessorKey: "volume_lots", header: "Lots", cell: ({ row }) => <span className="mono">{row.original.volume_lots.toFixed(2)}</span> },
    { accessorKey: "signed_exposure_base_ccy", header: "Exposure", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{formatCurrency(row.original.signed_exposure_base_ccy)}</span> },
    { accessorKey: "profit", header: "P&L", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.profit, 2)}</span> },
    { accessorKey: "time_utc", header: "Updated", cell: ({ row }) => formatTimestamp(row.original.time_utc) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function OrderHistoryTable({ rows }: { rows: OrderHistoryEntryResponse[] }) {
  const cols: ColumnDef<OrderHistoryEntryResponse>[] = [
    { accessorKey: "ticket", header: "Ticket", cell: ({ row }) => <span className="mono">{row.original.ticket ?? "—"}</span> },
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "side", header: "Side", cell: ({ row }) => row.original.side ? <StatusBadge label={row.original.side} tone={tone(row.original.side)} /> : "—" },
    { accessorKey: "state", header: "State", cell: ({ row }) => row.original.state ? <StatusBadge label={row.original.state} tone={tone(row.original.state)} /> : "—" },
    { accessorKey: "is_manual", header: "Origin", cell: ({ row }) => <StatusBadge label={row.original.is_manual ? "Manual" : "Desk"} tone={row.original.is_manual ? "warning" : "success"} /> },
    { accessorKey: "time_setup_utc", header: "Time", cell: ({ row }) => formatTimestamp(row.original.time_setup_utc) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function DealHistoryTable({ rows }: { rows: DealHistoryEntryResponse[] }) {
  const cols: ColumnDef<DealHistoryEntryResponse>[] = [
    { accessorKey: "ticket", header: "Deal", cell: ({ row }) => <span className="mono">{row.original.ticket ?? "—"}</span> },
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "side", header: "Side", cell: ({ row }) => row.original.side ? <StatusBadge label={row.original.side} tone={tone(row.original.side)} /> : "—" },
    { accessorKey: "volume", header: "Vol", cell: ({ row }) => row.original.volume == null ? "—" : row.original.volume.toFixed(2) },
    { accessorKey: "profit", header: "Profit", cell: ({ row }) => formatCurrency(row.original.profit, 2) },
    { accessorKey: "is_manual", header: "Origin", cell: ({ row }) => <StatusBadge label={row.original.is_manual ? "Manual" : "Desk"} tone={row.original.is_manual ? "warning" : "success"} /> },
    { accessorKey: "time_utc", header: "Time", cell: ({ row }) => formatTimestamp(row.original.time_utc) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function ReconciliationTable({
  rows,
  onManage,
}: {
  rows: ReconciliationMismatchResponse[];
  onManage?: (symbol: string) => void;
}) {
  const cols: ColumnDef<ReconciliationMismatchResponse>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "desk_exposure_eur", header: "Desk", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.desk_exposure_eur)}</span> },
    { accessorKey: "live_exposure_eur", header: "Live", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{formatCurrency(row.original.live_exposure_eur)}</span> },
    { accessorKey: "difference_eur", header: "Drift", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.difference_eur)}</span> },
    { accessorKey: "status", header: "Status", cell: ({ row }) => <StatusBadge label={row.original.status} tone={tone(row.original.status)} /> },
    { accessorKey: "incident_status", header: "Incident", cell: ({ row }) => row.original.incident_status ? <StatusBadge label={row.original.incident_status} tone={tone(row.original.incident_status)} /> : <StatusBadge label="new" tone="neutral" /> },
    ...(onManage ? [{
      id: "action", header: "Manage",
      cell: ({ row }: { row: { original: ReconciliationMismatchResponse } }) =>
        row.original.status !== "match" ? (
          <button type="button" className="rounded-[var(--radius-sm)] border border-[var(--color-border)] px-2 py-0.5 text-[10px] font-medium text-[var(--color-text)] transition hover:bg-[var(--color-surface-hover)]"
            onClick={() => onManage(row.original.symbol)}>Manage</button>
        ) : row.original.incident_status ? <span className="text-[10px] text-[var(--color-text-muted)]">{row.original.incident_status}</span> : null,
    } satisfies ColumnDef<ReconciliationMismatchResponse>] : []),
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="24rem" />;
}

export function InstrumentUniverseTable({ rows }: { rows: InstrumentDefinitionResponse[] }) {
  const cols: ColumnDef<InstrumentDefinitionResponse>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "asset_class", header: "Asset", cell: ({ row }) => humanize(row.original.asset_class) },
    { accessorKey: "contract_size", header: "Contract", cell: ({ row }) => row.original.contract_size?.toLocaleString() ?? "—" },
    { accessorKey: "base_currency", header: "Base", cell: ({ row }) => row.original.base_currency ?? "—" },
    { accessorKey: "volume_step", header: "Step", cell: ({ row }) => row.original.volume_step?.toFixed(2) ?? "—" },
    { accessorKey: "trading_mode", header: "Mode", cell: ({ row }) => row.original.trading_mode ? <StatusBadge label={row.original.trading_mode} tone={tone(row.original.trading_mode)} /> : "—" },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function ExecutionHistoryTable({ rows }: { rows: ExecutionResultResponse[] }) {
  const cols: ColumnDef<ExecutionResultResponse>[] = [
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "status", header: "Status", cell: ({ row }) => <StatusBadge label={row.original.status} tone={tone(row.original.status)} /> },
    { accessorKey: "approved_exposure_change", header: "Approved", cell: ({ row }) => <span className="mono">{formatCurrency(row.original.approved_exposure_change)}</span> },
    { accessorKey: "executed_exposure_change", header: "Executed", cell: ({ row }) => <span className="mono text-[var(--color-text)]">{formatCurrency(row.original.executed_exposure_change)}</span> },
    { accessorKey: "fill_ratio", header: "Fill", cell: ({ row }) => row.original.fill_ratio == null ? "—" : formatPercent(row.original.fill_ratio, 0) },
    { accessorKey: "reconciliation_status", header: "Broker", cell: ({ row }) => <StatusBadge label={row.original.reconciliation_status ?? row.original.guard.decision} tone={tone(row.original.reconciliation_status ?? row.original.guard.decision)} /> },
    { accessorKey: "created_at", header: "Time", cell: ({ row }) => formatTimestamp(row.original.created_at ?? row.original.time_utc) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="30rem" />;
}

export function ExecutionFillsTable({ rows }: { rows: ExecutionFillResponse[] }) {
  const cols: ColumnDef<ExecutionFillResponse>[] = [
    { accessorKey: "deal_ticket", header: "Deal", cell: ({ row }) => <span className="mono">{row.original.deal_ticket ?? "—"}</span> },
    { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-semibold text-[var(--color-text)]">{row.original.symbol}</span> },
    { accessorKey: "side", header: "Side", cell: ({ row }) => row.original.side ? <StatusBadge label={row.original.side} tone={tone(row.original.side)} /> : "—" },
    { accessorKey: "volume_lots", header: "Lots", cell: ({ row }) => <span className="mono">{row.original.volume_lots.toFixed(2)}</span> },
    { accessorKey: "price", header: "Price", cell: ({ row }) => row.original.price == null ? "—" : row.original.price.toFixed(4) },
    { accessorKey: "is_manual", header: "Origin", cell: ({ row }) => <StatusBadge label={row.original.is_manual ? "Manual" : "Desk"} tone={row.original.is_manual ? "warning" : "success"} /> },
    { accessorKey: "created_at", header: "Time", cell: ({ row }) => formatTimestamp(row.original.created_at ?? row.original.time_utc) },
  ];
  return <DataGrid data={rows} columns={cols} maxHeight="24rem" />;
}

function humanize(v: string | null | undefined) {
  if (!v) return "Unknown";
  return v.split("_").map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(" ");
}

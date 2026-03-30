"use client";

import type { ColumnDef } from "@tanstack/react-table";
import type {
  AuditEventResponse,
  DealHistoryEntryResponse,
  ExecutionFillResponse,
  ExecutionResultResponse,
  HoldingSnapshotResponse,
  InstrumentDefinitionResponse,
  ModelComparisonRow,
  MT5PendingOrderResponse,
  MT5PositionResponse,
  OrderHistoryEntryResponse,
  ReconciliationMismatchResponse,
  RiskDecisionResponse,
} from "@/lib/api/types";
import type { FlatAttributionRow, FlatCapitalRow } from "@/lib/view-models";
import { DataGrid } from "@/components/data/data-grid";
import { StatusBadge } from "@/components/ui/primitives";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";

function toneFromKeyword(value: string) {
  const normalized = value.toLowerCase();
  if (
    normalized.includes("reject") ||
    normalized.includes("breach") ||
    normalized.includes("critical") ||
    normalized.includes("danger") ||
    normalized.includes("failed")
  ) {
    return "danger" as const;
  }
  if (
    normalized.includes("reduce") ||
    normalized.includes("warn") ||
    normalized.includes("amber") ||
    normalized.includes("partial") ||
    normalized.includes("manual") ||
    normalized.includes("drift") ||
    normalized.includes("watch") ||
    normalized.includes("hold")
  ) {
    return "warning" as const;
  }
  if (
    normalized.includes("accept") ||
    normalized.includes("ok") ||
    normalized.includes("stable") ||
    normalized.includes("connected") ||
    normalized.includes("executed")
  ) {
    return "success" as const;
  }
  if (normalized.includes("champion")) {
    return "accent" as const;
  }
  return "neutral" as const;
}

export function ModelRankingTable({ rows }: { rows: ModelComparisonRow[] }) {
  const columns: ColumnDef<ModelComparisonRow>[] = [
    { accessorKey: "rank", header: "Rank" },
    {
      accessorKey: "model",
      header: "Model",
      cell: ({ row }) => (
        <span className="font-semibold text-white">{row.original.model.toUpperCase()}</span>
      ),
    },
    {
      accessorKey: "score",
      header: "Score",
      cell: ({ row }) => (
        <span className="mono text-white">{row.original.score.toFixed(1)}</span>
      ),
    },
    {
      accessorKey: "actual_rate",
      header: "Actual Rate",
      cell: ({ row }) => formatPercent(row.original.actual_rate),
    },
    {
      accessorKey: "exceptions",
      header: "Exceptions",
    },
    {
      accessorKey: "traffic_light",
      header: "Traffic Light",
      cell: ({ row }) =>
        row.original.traffic_light ? (
          <StatusBadge
            label={row.original.traffic_light}
            tone={toneFromKeyword(row.original.traffic_light)}
          />
        ) : (
          "n/a"
        ),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="30rem" />;
}

export function AttributionTable({ rows }: { rows: FlatAttributionRow[] }) {
  const columns: ColumnDef<FlatAttributionRow>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => (
        <span className="font-semibold text-white">{row.original.symbol}</span>
      ),
    },
    {
      accessorKey: "position",
      header: "Position",
      cell: ({ row }) => <span className="mono">{formatCurrency(row.original.position)}</span>,
    },
    {
      accessorKey: "componentVar",
      header: "Component VaR",
      cell: ({ row }) => (
        <span className="mono text-white">{formatCurrency(row.original.componentVar)}</span>
      ),
    },
    {
      accessorKey: "incrementalVar",
      header: "Incremental VaR",
      cell: ({ row }) => (
        <span className="mono">{formatCurrency(row.original.incrementalVar)}</span>
      ),
    },
    {
      accessorKey: "contributionPctVar",
      header: "Contribution",
      cell: ({ row }) => formatPercent(row.original.contributionPctVar),
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }) => (
        <StatusBadge label={row.original.status} tone={toneFromKeyword(row.original.status)} />
      ),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function CapitalAllocationTable({ rows }: { rows: FlatCapitalRow[] }) {
  const columns: ColumnDef<FlatCapitalRow>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "targetCapital",
      header: "Target",
      cell: ({ row }) => <span className="mono">{formatCurrency(row.original.targetCapital)}</span>,
    },
    {
      accessorKey: "consumedCapital",
      header: "Consumed",
      cell: ({ row }) => (
        <span className="mono text-white">{formatCurrency(row.original.consumedCapital)}</span>
      ),
    },
    {
      accessorKey: "remainingCapital",
      header: "Remaining",
      cell: ({ row }) => (
        <span className="mono">{formatCurrency(row.original.remainingCapital)}</span>
      ),
    },
    {
      accessorKey: "utilization",
      header: "Utilization",
      cell: ({ row }) => formatPercent(row.original.utilization),
    },
    {
      accessorKey: "action",
      header: "Action",
      cell: ({ row }) => (
        <StatusBadge label={row.original.action} tone={toneFromKeyword(row.original.action)} />
      ),
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }) => (
        <StatusBadge label={row.original.status} tone={toneFromKeyword(row.original.status)} />
      ),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function DecisionHistoryTable({
  rows,
}: {
  rows: RiskDecisionResponse[];
}) {
  const columns: ColumnDef<RiskDecisionResponse>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "decision",
      header: "Decision",
      cell: ({ row }) => (
        <StatusBadge
          label={row.original.decision}
          tone={toneFromKeyword(row.original.decision)}
        />
      ),
    },
    {
      accessorKey: "requested_exposure_change",
      header: "Requested exposure",
      cell: ({ row }) => (
        <span className="mono">{formatCurrency(row.original.requested_exposure_change)}</span>
      ),
    },
    {
      accessorKey: "approved_exposure_change",
      header: "Approved exposure",
      cell: ({ row }) => (
        <span className="mono text-white">{formatCurrency(row.original.approved_exposure_change)}</span>
      ),
    },
    {
      id: "fill_ratio",
      header: "Fill",
      cell: ({ row }) => {
        const requested = Math.abs(row.original.requested_exposure_change);
        const approved = Math.abs(row.original.approved_exposure_change);
        return requested === 0 ? "n/a" : formatPercent(approved / requested, 0);
      },
    },
    {
      accessorKey: "model_used",
      header: "Model",
      cell: ({ row }) => <span className="mono">{row.original.model_used.toUpperCase()}</span>,
    },
    {
      accessorKey: "created_at",
      header: "Time",
      cell: ({ row }) => formatTimestamp(row.original.created_at ?? row.original.time_utc),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function AuditTrailTable({ rows }: { rows: AuditEventResponse[] }) {
  const columns: ColumnDef<AuditEventResponse>[] = [
    {
      accessorKey: "actor",
      header: "Actor",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.actor}</span>,
    },
    {
      accessorKey: "action_type",
      header: "Action",
      cell: ({ row }) => <span className="mono">{row.original.action_type}</span>,
    },
    {
      accessorKey: "object_type",
      header: "Object",
      cell: ({ row }) => row.original.object_type ?? "n/a",
    },
    {
      accessorKey: "created_at",
      header: "Time",
      cell: ({ row }) => formatTimestamp(row.original.created_at),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="30rem" />;
}

export function MT5PositionsTable({ rows }: { rows: MT5PositionResponse[] }) {
  const columns: ColumnDef<MT5PositionResponse>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "side",
      header: "Side",
      cell: ({ row }) => (
        <StatusBadge label={row.original.side} tone={toneFromKeyword(row.original.side)} />
      ),
    },
    {
      accessorKey: "volume_lots",
      header: "Lots",
      cell: ({ row }) => <span className="mono">{row.original.volume_lots.toFixed(2)}</span>,
    },
    {
      accessorKey: "signed_exposure_base_ccy",
      header: "Exposure",
      cell: ({ row }) => <span className="mono text-white">{formatCurrency(row.original.signed_exposure_base_ccy)}</span>,
    },
    {
      accessorKey: "profit",
      header: "P&L",
      cell: ({ row }) => <span className="mono">{formatCurrency(row.original.profit, 2)}</span>,
    },
    {
      accessorKey: "time_utc",
      header: "Updated",
      cell: ({ row }) => formatTimestamp(row.original.time_utc),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function MT5OrdersTable({ rows }: { rows: MT5PendingOrderResponse[] }) {
  const columns: ColumnDef<MT5PendingOrderResponse>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "side",
      header: "Side",
      cell: ({ row }) => (
        <StatusBadge label={row.original.side} tone={toneFromKeyword(row.original.side)} />
      ),
    },
    {
      accessorKey: "volume_initial",
      header: "Initial",
      cell: ({ row }) => <span className="mono">{row.original.volume_initial.toFixed(2)}</span>,
    },
    {
      accessorKey: "volume_current",
      header: "Current",
      cell: ({ row }) => <span className="mono text-white">{row.original.volume_current.toFixed(2)}</span>,
    },
    {
      accessorKey: "price_open",
      header: "Price",
      cell: ({ row }) => <span className="mono">{row.original.price_open.toFixed(4)}</span>,
    },
    {
      accessorKey: "time_setup_utc",
      header: "Setup",
      cell: ({ row }) => formatTimestamp(row.original.time_setup_utc),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="28rem" />;
}

export function HoldingsTable({ rows }: { rows: HoldingSnapshotResponse[] }) {
  const columns: ColumnDef<HoldingSnapshotResponse>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "asset_class",
      header: "Asset",
      cell: ({ row }) => humanizeAssetClass(row.original.asset_class),
    },
    {
      accessorKey: "side",
      header: "Side",
      cell: ({ row }) => (
        <StatusBadge label={row.original.side} tone={toneFromKeyword(row.original.side)} />
      ),
    },
    {
      accessorKey: "volume_lots",
      header: "Lots",
      cell: ({ row }) => <span className="mono">{row.original.volume_lots.toFixed(2)}</span>,
    },
    {
      accessorKey: "signed_exposure_base_ccy",
      header: "Exposure",
      cell: ({ row }) => (
        <span className="mono text-white">{formatCurrency(row.original.signed_exposure_base_ccy)}</span>
      ),
    },
    {
      accessorKey: "profit",
      header: "P&L",
      cell: ({ row }) => <span className="mono">{formatCurrency(row.original.profit, 2)}</span>,
    },
    {
      accessorKey: "time_utc",
      header: "Updated",
      cell: ({ row }) => formatTimestamp(row.original.time_utc),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function OrderHistoryTable({ rows }: { rows: OrderHistoryEntryResponse[] }) {
  const columns: ColumnDef<OrderHistoryEntryResponse>[] = [
    {
      accessorKey: "ticket",
      header: "Ticket",
      cell: ({ row }) => <span className="mono">{row.original.ticket ?? "n/a"}</span>,
    },
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "side",
      header: "Side",
      cell: ({ row }) =>
        row.original.side ? (
          <StatusBadge label={row.original.side} tone={toneFromKeyword(row.original.side)} />
        ) : (
          "n/a"
        ),
    },
    {
      accessorKey: "state",
      header: "State",
      cell: ({ row }) =>
        row.original.state ? (
          <StatusBadge label={row.original.state} tone={toneFromKeyword(row.original.state)} />
        ) : (
          "n/a"
        ),
    },
    {
      accessorKey: "volume_initial",
      header: "Initial",
      cell: ({ row }) => (row.original.volume_initial == null ? "n/a" : row.original.volume_initial.toFixed(2)),
    },
    {
      accessorKey: "price_open",
      header: "Price",
      cell: ({ row }) => (row.original.price_open == null ? "n/a" : row.original.price_open.toFixed(4)),
    },
    {
      accessorKey: "is_manual",
      header: "Origin",
      cell: ({ row }) => (
        <StatusBadge
          label={row.original.is_manual ? "Manual" : "Desk"}
          tone={row.original.is_manual ? "warning" : "success"}
        />
      ),
    },
    {
      accessorKey: "time_setup_utc",
      header: "Setup",
      cell: ({ row }) => formatTimestamp(row.original.time_setup_utc),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function DealHistoryTable({ rows }: { rows: DealHistoryEntryResponse[] }) {
  const columns: ColumnDef<DealHistoryEntryResponse>[] = [
    {
      accessorKey: "ticket",
      header: "Deal",
      cell: ({ row }) => <span className="mono">{row.original.ticket ?? "n/a"}</span>,
    },
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "side",
      header: "Side",
      cell: ({ row }) =>
        row.original.side ? (
          <StatusBadge label={row.original.side} tone={toneFromKeyword(row.original.side)} />
        ) : (
          "n/a"
        ),
    },
    {
      accessorKey: "volume",
      header: "Volume",
      cell: ({ row }) => (row.original.volume == null ? "n/a" : row.original.volume.toFixed(2)),
    },
    {
      accessorKey: "price",
      header: "Price",
      cell: ({ row }) => (row.original.price == null ? "n/a" : row.original.price.toFixed(4)),
    },
    {
      accessorKey: "profit",
      header: "Profit",
      cell: ({ row }) => formatCurrency(row.original.profit, 2),
    },
    {
      accessorKey: "is_manual",
      header: "Origin",
      cell: ({ row }) => (
        <StatusBadge
          label={row.original.is_manual ? "Manual" : "Desk"}
          tone={row.original.is_manual ? "warning" : "success"}
        />
      ),
    },
    {
      accessorKey: "time_utc",
      header: "Time",
      cell: ({ row }) => formatTimestamp(row.original.time_utc),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function ReconciliationTable({
  rows,
  onAcknowledge,
}: {
  rows: ReconciliationMismatchResponse[];
  onAcknowledge?: (symbol: string) => void;
}) {
  const columns: ColumnDef<ReconciliationMismatchResponse>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "desk_exposure_eur",
      header: "Desk",
      cell: ({ row }) => <span className="mono">{formatCurrency(row.original.desk_exposure_eur)}</span>,
    },
    {
      accessorKey: "live_exposure_eur",
      header: "Live",
      cell: ({ row }) => <span className="mono text-white">{formatCurrency(row.original.live_exposure_eur)}</span>,
    },
    {
      accessorKey: "difference_eur",
      header: "Drift",
      cell: ({ row }) => <span className="mono">{formatCurrency(row.original.difference_eur)}</span>,
    },
    {
      accessorKey: "reason",
      header: "Reason",
      cell: ({ row }) => row.original.reason ?? "n/a",
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }) => (
        <StatusBadge label={row.original.status} tone={toneFromKeyword(row.original.status)} />
      ),
    },
    {
      id: "acknowledgement",
      header: "Ack",
      cell: ({ row }) =>
        row.original.acknowledged ? (
          <div className="space-y-1">
            <StatusBadge label="Acknowledged" tone="neutral" />
            <div className="text-xs text-[var(--color-text-muted)]">
              {formatTimestamp(row.original.acknowledged_at)}
            </div>
          </div>
        ) : (
          <span className="text-xs text-[var(--color-text-muted)]">Pending</span>
        ),
    },
    ...(onAcknowledge
      ? [
          {
            id: "action",
            header: "Action",
            cell: ({ row }: { row: { original: ReconciliationMismatchResponse } }) =>
              row.original.status !== "match" && !row.original.acknowledged ? (
                <button
                  type="button"
                  className="rounded-full border border-white/12 bg-white/5 px-3 py-1 text-xs font-semibold text-[var(--color-text)] transition hover:bg-white/8"
                  onClick={() => onAcknowledge(row.original.symbol)}
                >
                  Acknowledge
                </button>
              ) : row.original.acknowledged ? (
                <span className="text-xs text-[var(--color-text-muted)]">Logged</span>
              ) : null,
          } satisfies ColumnDef<ReconciliationMismatchResponse>,
        ]
      : []),
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="28rem" />;
}

export function InstrumentUniverseTable({
  rows,
}: {
  rows: InstrumentDefinitionResponse[];
}) {
  const columns: ColumnDef<InstrumentDefinitionResponse>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "asset_class",
      header: "Asset",
      cell: ({ row }) => humanizeAssetClass(row.original.asset_class),
    },
    {
      accessorKey: "contract_size",
      header: "Contract",
      cell: ({ row }) => (row.original.contract_size == null ? "n/a" : row.original.contract_size.toLocaleString()),
    },
    {
      accessorKey: "base_currency",
      header: "Base",
      cell: ({ row }) => row.original.base_currency ?? "n/a",
    },
    {
      accessorKey: "profit_currency",
      header: "Profit",
      cell: ({ row }) => row.original.profit_currency ?? "n/a",
    },
    {
      accessorKey: "volume_step",
      header: "Step",
      cell: ({ row }) => (row.original.volume_step == null ? "n/a" : row.original.volume_step.toFixed(2)),
    },
    {
      accessorKey: "trading_mode",
      header: "Mode",
      cell: ({ row }) =>
        row.original.trading_mode ? (
          <StatusBadge label={row.original.trading_mode} tone={toneFromKeyword(row.original.trading_mode)} />
        ) : (
          "n/a"
        ),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function ExecutionHistoryTable({ rows }: { rows: ExecutionResultResponse[] }) {
  const columns: ColumnDef<ExecutionResultResponse>[] = [
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }) => (
        <StatusBadge label={row.original.status} tone={toneFromKeyword(row.original.status)} />
      ),
    },
    {
      accessorKey: "approved_exposure_change",
      header: "Approved exposure",
      cell: ({ row }) => <span className="mono">{formatCurrency(row.original.approved_exposure_change)}</span>,
    },
    {
      accessorKey: "executed_exposure_change",
      header: "Executed exposure",
      cell: ({ row }) => <span className="mono text-white">{formatCurrency(row.original.executed_exposure_change)}</span>,
    },
    {
      accessorKey: "volume_lots",
      header: "Lots",
      cell: ({ row }) => (
        <span className="mono">
          {(row.original.filled_volume_lots ?? row.original.submitted_volume_lots ?? row.original.guard.volume_lots).toFixed(2)}
        </span>
      ),
    },
    {
      accessorKey: "fill_ratio",
      header: "Fill",
      cell: ({ row }) =>
        row.original.fill_ratio == null ? "n/a" : formatPercent(row.original.fill_ratio, 0),
    },
    {
      accessorKey: "reconciliation_status",
      header: "Broker",
      cell: ({ row }) => (
        <StatusBadge
          label={row.original.reconciliation_status ?? row.original.guard.decision}
          tone={toneFromKeyword(row.original.reconciliation_status ?? row.original.guard.decision)}
        />
      ),
    },
    {
      accessorKey: "created_at",
      header: "Time",
      cell: ({ row }) => formatTimestamp(row.original.created_at ?? row.original.time_utc),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="34rem" />;
}

export function ExecutionFillsTable({ rows }: { rows: ExecutionFillResponse[] }) {
  const columns: ColumnDef<ExecutionFillResponse>[] = [
    {
      accessorKey: "deal_ticket",
      header: "Deal",
      cell: ({ row }) => <span className="mono">{row.original.deal_ticket ?? "n/a"}</span>,
    },
    {
      accessorKey: "symbol",
      header: "Symbol",
      cell: ({ row }) => <span className="font-semibold text-white">{row.original.symbol}</span>,
    },
    {
      accessorKey: "side",
      header: "Side",
      cell: ({ row }) =>
        row.original.side ? (
          <StatusBadge label={row.original.side} tone={toneFromKeyword(row.original.side)} />
        ) : (
          "n/a"
        ),
    },
    {
      accessorKey: "volume_lots",
      header: "Lots",
      cell: ({ row }) => <span className="mono">{row.original.volume_lots.toFixed(2)}</span>,
    },
    {
      accessorKey: "price",
      header: "Price",
      cell: ({ row }) => (row.original.price == null ? "n/a" : row.original.price.toFixed(4)),
    },
    {
      accessorKey: "slippage_points",
      header: "Slippage",
      cell: ({ row }) => (row.original.slippage_points == null ? "n/a" : row.original.slippage_points.toFixed(1)),
    },
    {
      accessorKey: "is_manual",
      header: "Origin",
      cell: ({ row }) => (
        <StatusBadge
          label={row.original.is_manual ? "Manual" : "Desk"}
          tone={row.original.is_manual ? "warning" : "success"}
        />
      ),
    },
    {
      accessorKey: "created_at",
      header: "Time",
      cell: ({ row }) => formatTimestamp(row.original.created_at ?? row.original.time_utc),
    },
  ];

  return <DataGrid data={rows} columns={columns} maxHeight="28rem" />;
}

function humanizeAssetClass(value: string | null | undefined) {
  if (!value) {
    return "Unknown";
  }
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

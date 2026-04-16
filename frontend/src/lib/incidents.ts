import type {
  AuditEventResponse,
  DealHistoryEntryResponse,
  OrderHistoryEntryResponse,
  ReconciliationAcknowledgementResponse,
  ReconciliationSummaryResponse,
} from "@/lib/api/types";

export type IncidentWorkbenchRow = {
  symbol: string;
  mismatch_status: string;
  incident_status: string | null;
  reason: string;
  operator_note: string;
  resolution_note: string;
  desk_exposure_eur: number;
  live_exposure_eur: number;
  difference_eur: number;
  desk_volume_lots: number | null;
  live_volume_lots: number | null;
  acknowledged_at: string | null;
  resolved_at: string | null;
  updated_at: string | null;
  incident_id: number | null;
  active: boolean;
  requires_action: boolean;
};

export type IncidentTimelineEntry = {
  id: string;
  time: string | null;
  kind: "incident" | "audit" | "order" | "deal";
  title: string;
  detail: string;
  tone: "neutral" | "accent" | "success" | "warning" | "danger";
};

function toMillis(value: string | null | undefined): number {
  if (!value) {
    return 0;
  }
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function defaultMismatchStatus(incident: ReconciliationAcknowledgementResponse): string {
  return incident.mismatch_status || (incident.incident_status === "resolved" ? "match" : "desk_vs_broker_drift");
}

export function buildIncidentWorkbenchRows(
  reconciliation: ReconciliationSummaryResponse | null | undefined,
  fallbackIncidents: ReconciliationAcknowledgementResponse[] = [],
): IncidentWorkbenchRow[] {
  const mismatches = reconciliation?.mismatches ?? [];
  const incidentRecords = reconciliation?.incidents?.length
    ? reconciliation.incidents
    : fallbackIncidents;

  const incidentMap = new Map(
    incidentRecords.map((item) => [item.symbol, item] as const),
  );

  const rows = new Map<string, IncidentWorkbenchRow>();

  for (const mismatch of mismatches) {
    const incident = incidentMap.get(mismatch.symbol);
    const mismatchStatus = mismatch.status ?? defaultMismatchStatus(incident ?? { symbol: mismatch.symbol } as ReconciliationAcknowledgementResponse);
    const incidentStatus = mismatch.incident_status ?? incident?.incident_status ?? null;
    const resolved = incidentStatus === "resolved" && mismatchStatus === "match";
    rows.set(mismatch.symbol, {
      symbol: mismatch.symbol,
      mismatch_status: mismatchStatus,
      incident_status: incidentStatus,
      reason: mismatch.incident_reason ?? incident?.reason ?? mismatch.reason ?? "",
      operator_note: mismatch.incident_note ?? incident?.operator_note ?? "",
      resolution_note: mismatch.resolution_note ?? incident?.resolution_note ?? "",
      desk_exposure_eur: Number(mismatch.desk_exposure_eur ?? 0),
      live_exposure_eur: Number(mismatch.live_exposure_eur ?? 0),
      difference_eur: Number(mismatch.difference_eur ?? 0),
      desk_volume_lots: mismatch.desk_volume_lots ?? null,
      live_volume_lots: mismatch.live_volume_lots ?? null,
      acknowledged_at: mismatch.acknowledged_at ?? incident?.acknowledged_at ?? null,
      resolved_at: mismatch.resolved_at ?? incident?.resolved_at ?? null,
      updated_at: mismatch.incident_updated_at ?? incident?.updated_at ?? null,
      incident_id: mismatch.incident_id ?? incident?.id ?? null,
      active: mismatchStatus !== "match",
      requires_action: mismatchStatus !== "match" && !resolved,
    });
  }

  for (const incident of incidentRecords) {
    if (rows.has(incident.symbol)) {
      continue;
    }
    const mismatchStatus = defaultMismatchStatus(incident as ReconciliationAcknowledgementResponse);
    rows.set(incident.symbol, {
      symbol: incident.symbol,
      mismatch_status: mismatchStatus,
      incident_status: incident.incident_status ?? null,
      reason: incident.reason ?? "",
      operator_note: incident.operator_note ?? "",
      resolution_note: incident.resolution_note ?? "",
      desk_exposure_eur: Number((incident as { desk_exposure_eur?: number }).desk_exposure_eur ?? 0),
      live_exposure_eur: Number((incident as { live_exposure_eur?: number }).live_exposure_eur ?? 0),
      difference_eur: Number((incident as { difference_eur?: number }).difference_eur ?? 0),
      desk_volume_lots: null,
      live_volume_lots: null,
      acknowledged_at: incident.acknowledged_at ?? null,
      resolved_at: incident.resolved_at ?? null,
      updated_at: incident.updated_at ?? null,
      incident_id: incident.id ?? null,
      active: mismatchStatus !== "match",
      requires_action: incident.incident_status !== "resolved",
    });
  }

  return Array.from(rows.values()).sort((left, right) => {
    if (left.requires_action !== right.requires_action) {
      return left.requires_action ? -1 : 1;
    }
    if (left.active !== right.active) {
      return left.active ? -1 : 1;
    }
    return toMillis(right.updated_at || right.acknowledged_at) - toMillis(left.updated_at || left.acknowledged_at);
  });
}

export function buildIncidentTimeline(
  symbol: string | null | undefined,
  audit: AuditEventResponse[],
  orders: OrderHistoryEntryResponse[],
  deals: DealHistoryEntryResponse[],
): IncidentTimelineEntry[] {
  const normalized = String(symbol || "").toUpperCase();
  if (!normalized) {
    return [];
  }

  const entries: IncidentTimelineEntry[] = [];

  for (const event of audit) {
    const eventSymbol = String((event.payload ?? {}).symbol || "").toUpperCase();
    if (eventSymbol !== normalized || !String(event.action_type || "").startsWith("reconciliation.")) {
      continue;
    }
    entries.push({
      id: `audit:${event.id}`,
      time: event.created_at ?? null,
      kind: "audit",
      title: String(event.action_type || "reconciliation.update"),
      detail: String(
        (event.payload ?? {}).resolution_note
          || (event.payload ?? {}).operator_note
          || (event.payload ?? {}).reason
          || "Operator reconciliation update."
      ),
      tone: String((event.payload ?? {}).incident_status || "").toLowerCase() === "resolved" ? "success" : "accent",
    });
  }

  for (const order of orders) {
    if (String(order.symbol || "").toUpperCase() !== normalized) {
      continue;
    }
    entries.push({
      id: [
        "order",
        order.ticket ?? "ticket",
        order.time_setup_utc ?? "time",
        order.symbol ?? normalized,
        order.side ?? "side",
        order.state ?? "state",
      ].join(":"),
      time: order.time_setup_utc ?? null,
      kind: "order",
      title: `${order.is_manual ? "Manual" : "Desk"} order ${order.state ?? "seen"}`,
      detail: `${order.side ?? "n/a"} ${order.volume_initial ?? order.volume_current ?? 0} lots`,
      tone: order.is_manual ? "warning" : "neutral",
    });
  }

  for (const deal of deals) {
    if (String(deal.symbol || "").toUpperCase() !== normalized) {
      continue;
    }
    entries.push({
      id: [
        "deal",
        deal.ticket ?? "ticket",
        deal.time_utc ?? "time",
        deal.symbol ?? normalized,
        deal.side ?? "side",
        deal.entry ?? "entry",
      ].join(":"),
      time: deal.time_utc ?? null,
      kind: "deal",
      title: `${deal.is_manual ? "Manual" : "Desk"} deal ${deal.side ?? "seen"}`,
      detail: `${deal.volume ?? 0} lots @ ${deal.price ?? "n/a"}`,
      tone: deal.is_manual ? "warning" : "success",
    });
  }

  return entries.sort((left, right) => toMillis(right.time) - toMillis(left.time));
}

export function incidentMatchesQuery(row: IncidentWorkbenchRow, query: string): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return true;
  }
  return [row.symbol, row.reason, row.operator_note, row.resolution_note, row.mismatch_status, row.incident_status]
    .filter(Boolean)
    .some((value) => String(value).toLowerCase().includes(normalized));
}

export function incidentMatchesWindow(
  row: IncidentWorkbenchRow,
  window: "all" | "24h" | "7d",
  referenceTimeMs = 0,
): boolean {
  if (window === "all") {
    return true;
  }
  const anchor = toMillis(row.updated_at || row.resolved_at || row.acknowledged_at);
  if (anchor <= 0) {
    return false;
  }
  if (referenceTimeMs <= 0) {
    return true;
  }
  const maxAge = window === "24h" ? 24 * 60 * 60 * 1000 : 7 * 24 * 60 * 60 * 1000;
  return referenceTimeMs - anchor <= maxAge;
}

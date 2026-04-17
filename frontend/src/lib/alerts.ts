import type { AlertSummary, OperatorAlertResponse } from "@/lib/api/types";

function stableJson(value: unknown): string {
  if (value == null || typeof value !== "object") {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map((item) => stableJson(item)).join(",")}]`;
  }

  const entries = Object.entries(value as Record<string, unknown>).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
}

function persistedAlertSignature(alert: AlertSummary) {
  return [
    alert.code,
    alert.severity,
    alert.message,
    stableJson(alert.context ?? {}),
  ].join("|");
}

function operatorAlertSignature(alert: OperatorAlertResponse) {
  return [
    alert.code,
    alert.severity,
    alert.message,
    stableJson(alert.context ?? {}),
  ].join("|");
}

function comparePersistedAlerts(left: AlertSummary, right: AlertSummary) {
  const leftTime = Date.parse(left.created_at ?? "");
  const rightTime = Date.parse(right.created_at ?? "");
  if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) {
    return rightTime - leftTime;
  }
  return (right.id ?? 0) - (left.id ?? 0);
}

export function dedupePersistedAlerts(alerts: AlertSummary[]) {
  const latestBySignature = new Map<string, AlertSummary>();
  for (const alert of alerts) {
    const signature = persistedAlertSignature(alert);
    const current = latestBySignature.get(signature);
    if (current == null || comparePersistedAlerts(alert, current) < 0) {
      latestBySignature.set(signature, alert);
    }
  }
  return [...latestBySignature.values()].sort(comparePersistedAlerts);
}

export function dedupeOperatorAlerts(alerts: OperatorAlertResponse[]) {
  const seen = new Set<string>();
  const deduped: OperatorAlertResponse[] = [];
  for (const alert of alerts) {
    const signature = operatorAlertSignature(alert);
    if (seen.has(signature)) {
      continue;
    }
    seen.add(signature);
    deduped.push(alert);
  }
  return deduped;
}

export function alertPriorityCode(code: string | null | undefined): number {
  const normalized = String(code || "").toUpperCase();
  if (
    normalized.includes("VALIDATION_GOVERNANCE_FAIL")
    || normalized.includes("VALIDATION_SURFACE_COVERAGE_FAIL")
    || normalized.includes("VALIDATION_SURFACE_CONDITIONAL_FAIL")
    || normalized.includes("VALIDATION_HORIZON_FAIL")
    || normalized.includes("VALIDATION_ES_SHORTFALL_BREACH")
    || normalized.includes("VALIDATION_ES_BREACH_RATE_BREACH")
    || normalized.includes("BROKER_REJECTION")
    || normalized.includes("DESK_BROKER_DRIFT")
    || normalized.includes("ORPHAN_LIVE_POSITION")
    || normalized.includes("OVERFILL_OR_VOLUME_DRIFT")
  ) {
    return 0;
  }
  if (normalized.includes("RECONCILIATION_INCOMPLETE")) return 1;
  if (
    normalized.includes("VALIDATION_GOVERNANCE_WARN")
    || normalized.includes("VALIDATION_SURFACE_INDEPENDENCE_FAIL")
    || normalized.includes("VALIDATION_HORIZON_WARN")
    || normalized.includes("VALIDATION_SURFACE_SAMPLE_THIN")
    || normalized.includes("VALIDATION_HORIZON_SAMPLE_THIN")
    || normalized.includes("VALIDATION_ES_SHORTFALL_WARN")
    || normalized.includes("VALIDATION_ES_BREACH_RATE_WARN")
    || normalized.includes("WINDOW_EXPIRED")
  ) {
    return 2;
  }
  if (normalized.includes("PARTIAL_FILL")) return 3;
  if (normalized.includes("PENDING_BROKER")) return 4;
  if (normalized.includes("MANUAL_TRADE") || normalized.includes("MANUAL_EVENTS")) return 5;
  if (normalized.includes("UNMATCHED")) return 6;
  return 7;
}

export function isValidationGovernanceAlertCode(code: string | null | undefined): boolean {
  return String(code || "").toUpperCase().startsWith("VALIDATION_");
}

import type { OperatorAlertResponse } from "@/lib/api/types";
import { StatusBadge } from "@/components/ui/primitives";
import { dedupeOperatorAlerts } from "@/lib/alerts";
import { humanizeIdentifier } from "@/lib/utils";

function toneFromSeverity(severity: string) {
  const s = severity.toLowerCase();
  if (s.includes("breach") || s.includes("critical") || s.includes("danger")) {
    return "danger" as const;
  }
  if (s.includes("warn")) {
    return "warning" as const;
  }
  return "neutral" as const;
}

function priorityFromAlert(alert: OperatorAlertResponse): number {
  const code = String(alert.code || "").toUpperCase();
  if (code.includes("BROKER_REJECTION")) return 0;
  if (code.includes("RECONCILIATION_INCOMPLETE")) return 1;
  if (code.includes("WINDOW_EXPIRED")) return 2;
  if (code.includes("PARTIAL_FILL")) return 3;
  if (code.includes("MANUAL_TRADE") || code.includes("MANUAL_EVENTS")) return 4;
  if (code.includes("DRIFT") || code.includes("ORPHAN")) return 5;
  return 6;
}

function contextLabel(alert: OperatorAlertResponse): string | null {
  const context = alert.context ?? {};
  if (context.evidence_state === "empty_live_book") {
    return "broker empty";
  }
  if (typeof context.history_window_expired_execution_count === "number" && context.history_window_expired_execution_count > 0) {
    return `${context.history_window_expired_execution_count} stale`;
  }
  if (typeof context.count === "number") {
    return `${context.count} active`;
  }
  if (typeof context.manual_event_count === "number") {
    return `${context.manual_event_count} manual`;
  }
  if (typeof context.unmatched_execution_count === "number") {
    return `${context.unmatched_execution_count} unmatched`;
  }
  return null;
}

export function LiveOperatorAlerts({
  alerts,
  title = "Alerts",
}: {
  alerts: OperatorAlertResponse[];
  title?: string;
}) {
  const dedupedAlerts = dedupeOperatorAlerts(alerts);
  if (dedupedAlerts.length === 0) {
    return null;
  }

  const sortedAlerts = [...dedupedAlerts].sort((left, right) => {
    const severityDelta = priorityFromAlert(left) - priorityFromAlert(right);
    if (severityDelta !== 0) {
      return severityDelta;
    }
    return left.code.localeCompare(right.code);
  });
  const hasDanger = dedupedAlerts.some((alert) => toneFromSeverity(alert.severity) === "danger");
  const hasWarning = dedupedAlerts.some((alert) => toneFromSeverity(alert.severity) === "warning");
  const aggregateTone = hasDanger ? "danger" : hasWarning ? "warning" : "neutral";

  return (
    <section className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
          {title}
        </span>
        <StatusBadge
          label={`${dedupedAlerts.length} active`}
          tone={aggregateTone}
        />
      </div>
      <div className="grid gap-2 xl:grid-cols-2">
        {sortedAlerts.map((alert) => {
          const context = alert.context ?? {};
          const highlight = priorityFromAlert(alert) <= 4;
          const contextText = contextLabel(alert);
          return (
            <div
              key={`${alert.code}:${JSON.stringify(context)}`}
              className={`flex items-start justify-between gap-2 rounded-[var(--radius-md)] border px-3 py-2 ${
                highlight
                  ? "border-[var(--color-border-strong)] bg-[var(--color-surface-hover)]"
                  : "border-[var(--color-border)] bg-[var(--color-bg)]"
              }`}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-[var(--color-text)]">
                    {humanizeIdentifier(alert.code)}
                  </span>
                  <StatusBadge label={alert.severity} tone={toneFromSeverity(alert.severity)} />
                  {contextText ? <StatusBadge label={contextText} tone="neutral" /> : null}
                </div>
                <p className="mt-0.5 text-[11px] leading-relaxed text-[var(--color-text-muted)]">
                  {alert.message}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

import type { OperatorAlertResponse } from "@/lib/api/types";
import { StatusBadge } from "@/components/ui/primitives";
import { formatTimestamp, humanizeIdentifier } from "@/lib/utils";

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

export function LiveOperatorAlerts({
  alerts,
  title = "Alerts",
  copy,
}: {
  alerts: OperatorAlertResponse[];
  title?: string;
  copy?: string;
}) {
  if (alerts.length === 0) {
    return null;
  }

  return (
    <section className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
          {title}
        </span>
        <StatusBadge
          label={`${alerts.length} active`}
          tone={alerts.some((a) => toneFromSeverity(a.severity) === "danger") ? "danger" : "warning"}
        />
      </div>
      <div className="grid gap-2 xl:grid-cols-2">
        {alerts.map((alert) => {
          const context = alert.context ?? {};
          return (
            <div
              key={`${alert.code}:${JSON.stringify(context)}`}
              className="flex items-start justify-between gap-2 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-[var(--color-text)]">
                    {humanizeIdentifier(alert.code)}
                  </span>
                  <StatusBadge label={alert.severity} tone={toneFromSeverity(alert.severity)} />
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

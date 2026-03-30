import type { OperatorAlertResponse } from "@/lib/api/types";
import { StatusBadge } from "@/components/ui/primitives";
import { formatTimestamp, humanizeIdentifier } from "@/lib/utils";

function toneFromSeverity(severity: string) {
  const normalized = severity.toLowerCase();
  if (normalized.includes("breach") || normalized.includes("critical") || normalized.includes("danger")) {
    return "danger" as const;
  }
  if (normalized.includes("warn")) {
    return "warning" as const;
  }
  return "neutral" as const;
}

export function LiveOperatorAlerts({
  alerts,
  title = "Operator alerts",
  copy = "Live incidents derived from the MT5 bridge, reconciliation and execution feed.",
}: {
  alerts: OperatorAlertResponse[];
  title?: string;
  copy?: string;
}) {
  if (alerts.length === 0) {
    return (
      <section className="surface rounded-[1.7rem] p-6">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          {title}
        </div>
        <div className="mt-3 text-sm leading-7 text-[var(--color-text-soft)]">
          {copy} No live incident requires operator follow-up right now.
        </div>
      </section>
    );
  }

  return (
    <section className="surface rounded-[1.7rem] p-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            {title}
          </div>
          <div className="mt-2 text-sm leading-7 text-[var(--color-text-soft)]">{copy}</div>
        </div>
        <StatusBadge
          label={`${alerts.length} live`}
          tone={alerts.some((alert) => toneFromSeverity(alert.severity) === "danger") ? "danger" : "warning"}
        />
      </div>
      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        {alerts.map((alert) => {
          const context = alert.context ?? {};
          return (
          <div
            key={`${alert.code}:${JSON.stringify(context)}`}
            className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-base font-semibold text-white">
                  {humanizeIdentifier(alert.code)}
                </div>
                <div className="mt-2 text-sm leading-7 text-[var(--color-text-soft)]">
                  {alert.message}
                </div>
              </div>
              <StatusBadge label={alert.severity} tone={toneFromSeverity(alert.severity)} />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.2em] text-[var(--color-text-muted)]">
              <span>{humanizeIdentifier(alert.source)}</span>
              {context.generated_at ? <span>{formatTimestamp(String(context.generated_at))}</span> : null}
              {context.count != null ? <span>{String(context.count)} active</span> : null}
              {context.manual_event_count != null ? (
                <span>{String(context.manual_event_count)} manual</span>
              ) : null}
            </div>
          </div>
          );
        })}
      </div>
    </section>
  );
}

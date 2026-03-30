"use client";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  BriefcaseBusiness,
  FileText,
  Flame,
  Gauge,
  Landmark,
  Languages,
  Orbit,
  PanelRightClose,
  PanelRightOpen,
  Radar,
  ShieldCheck,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";
import type {
  AlertSummary,
  AuditEventResponse,
  HealthResponse,
  MT5LiveStateResponse,
  PortfolioSummary,
  WorkerStatusResponse,
} from "@/lib/api/types";
import { OperatorActions } from "@/components/app-shell/operator-actions";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { cn, formatTimestamp } from "@/lib/utils";

const navItems = [
  { href: "/desk", label: "Overview", icon: Gauge },
  { href: "/desk/live", label: "MT5 Ops", icon: Activity },
  { href: "/desk/universe", label: "Universe", icon: Orbit },
  { href: "/desk/models", label: "Models", icon: Radar },
  { href: "/desk/attribution", label: "Attribution", icon: BarChart3 },
  { href: "/desk/capital", label: "Capital", icon: Landmark },
  { href: "/desk/decisions", label: "Decisions", icon: ShieldCheck },
  { href: "/desk/execution", label: "Dry Run", icon: BriefcaseBusiness },
  { href: "/desk/stress", label: "Stress", icon: Flame },
  { href: "/desk/blotter", label: "Blotter", icon: Orbit },
  { href: "/desk/reports", label: "Reports", icon: FileText },
] as const;

function severityTone(severity: string) {
  const normalized = severity.toLowerCase();
  if (normalized.includes("breach") || normalized.includes("high")) {
    return "text-[var(--color-red)]";
  }
  if (normalized.includes("warn")) {
    return "text-[var(--color-amber)]";
  }
  return "text-[var(--color-green)]";
}

function getRouteBehavior(pathname: string) {
  if (pathname === "/desk" || pathname === "/desk/reports") {
    return {
      dockedRailClass: "xl:flex",
      gridClass: "xl:grid-cols-[92px_minmax(0,1fr)_330px]",
      showInspectorToggle: false,
    };
  }

  if (pathname === "/desk/capital") {
    return {
      dockedRailClass: "2xl:flex",
      gridClass: "xl:grid-cols-[92px_minmax(0,1fr)] 2xl:grid-cols-[92px_minmax(0,1fr)_330px]",
      showInspectorToggle: true,
    };
  }

  return {
    dockedRailClass: "hidden",
    gridClass: "xl:grid-cols-[92px_minmax(0,1fr)]",
    showInspectorToggle: true,
  };
}

export function DeskChrome({
  children,
  health,
  portfolios,
  alerts,
  audit,
  jobsStatus,
}: {
  children: React.ReactNode;
  health: HealthResponse;
  portfolios: PortfolioSummary[];
  alerts: AlertSummary[];
  audit: AuditEventResponse[];
  jobsStatus: WorkerStatusResponse | null;
}) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const portfolioSlug = searchParams.get("portfolio") ?? health.portfolio_slug;
  const latestReady =
    health.latest_artifacts.daily_report || health.latest_artifacts.backtest_compare;
  const routeBehavior = useMemo(() => getRouteBehavior(pathname), [pathname]);
  const contextKey = `${pathname}:${portfolioSlug}`;
  const [openContextKey, setOpenContextKey] = useState<string | null>(null);
  const contextOpen = openContextKey === contextKey;
  const { liveState, transport } = useMt5LiveState(portfolioSlug ?? undefined);
  const liveTone =
    liveState?.status === "ok"
      ? "border-emerald-400/18 bg-emerald-400/10 text-[var(--color-green)]"
      : liveState?.degraded
        ? "border-amber-400/18 bg-amber-400/10 text-[var(--color-amber)]"
        : "border-white/8 text-[var(--color-text-soft)]";

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_right,rgba(216,155,73,0.08),transparent_18%),linear-gradient(180deg,rgba(255,255,255,0.02),transparent_24%)]">
      <div className={cn("grid min-h-screen grid-cols-1", routeBehavior.gridClass)}>
        <aside
          data-desk-rail
          className="sticky top-0 hidden h-screen border-r border-white/8 bg-[rgba(8,9,11,0.82)] px-5 py-7 backdrop-blur-2xl xl:flex xl:flex-col xl:gap-8"
        >
          <Link href="/en" className="flex items-center justify-center">
            <div className="flex size-14 items-center justify-center rounded-[1.4rem] border border-[var(--color-border-strong)] bg-[var(--color-accent-soft)] text-sm font-semibold tracking-[0.24em] text-[var(--color-accent)]">
              VR
            </div>
          </Link>
          <nav className="flex flex-1 flex-col gap-3">
            {navItems.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={`${href}?portfolio=${portfolioSlug}`}
                  className={cn(
                    "group flex flex-col items-center gap-3 rounded-[1.4rem] border px-2 py-3 text-center transition duration-300 motion-safe:hover:-translate-y-[1px]",
                    active
                      ? "border-[var(--color-border-strong)] bg-[linear-gradient(180deg,rgba(216,155,73,0.16),rgba(216,155,73,0.06))] text-white shadow-[0_16px_36px_rgba(216,155,73,0.12)]"
                      : "border-transparent text-[var(--color-text-muted)] hover:border-white/8 hover:bg-white/[0.03] hover:text-[var(--color-text)]",
                  )}
                >
                  <Icon className="size-5" />
                  <span className="text-[10px] font-medium uppercase tracking-[0.22em]">
                    {label}
                  </span>
                </Link>
              );
            })}
          </nav>
          <Link
            href="/fr"
            className="flex items-center justify-center gap-2 rounded-full border border-white/8 px-3 py-2 text-xs uppercase tracking-[0.24em] text-[var(--color-text-soft)] transition hover:border-[var(--color-border-strong)] hover:text-white"
          >
            <Languages className="size-4" />
            FR
          </Link>
        </aside>

        <div className="min-w-0">
          <header
            data-desk-topbar
            className="sticky top-0 z-20 border-b border-white/8 bg-[rgba(8,9,11,0.82)] px-5 py-4 backdrop-blur-2xl md:px-8"
          >
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-accent)]">
                    MT5 Risk Desk Platform
                  </div>
                  <div className="mt-2 text-sm text-[var(--color-text-soft)]">
                    Operator surface for market sync, blotter, reconciliation, guarded execution and reports.
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <OperatorActions portfolioSlug={portfolioSlug} />
                  {routeBehavior.showInspectorToggle ? (
                    <button
                      type="button"
                      onClick={() =>
                        setOpenContextKey((value) => (value === contextKey ? null : contextKey))
                      }
                    className="inline-flex h-10 items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 text-xs uppercase tracking-[0.22em] text-[var(--color-text-soft)] transition duration-300 motion-safe:hover:-translate-y-[1px] hover:border-[var(--color-border-strong)] hover:text-white"
                    >
                      {contextOpen ? (
                        <PanelRightClose className="size-4" />
                      ) : (
                        <PanelRightOpen className="size-4" />
                      )}
                      Operator panel
                    </button>
                  ) : null}
                  <div className="rounded-full border border-white/8 px-3 py-2 text-xs uppercase tracking-[0.22em] text-[var(--color-text-soft)]">
                    {health.desk_slug ?? "desk"}
                  </div>
                  <div className="rounded-full border border-emerald-400/18 bg-emerald-400/10 px-3 py-2 text-xs uppercase tracking-[0.22em] text-[var(--color-green)]">
                    API {health.status}
                  </div>
                  <div className={cn("rounded-full border px-3 py-2 text-xs uppercase tracking-[0.22em]", liveTone)}>
                    {liveState
                      ? `MT5 ${liveState.status} · ${transport === "stream" ? "stream" : transport === "polling" ? "poll" : "connect"}`
                      : "MT5 live pending"}
                  </div>
                  {(liveState?.operator_alerts ?? []).length > 0 ? (
                    <div className="rounded-full border border-amber-400/18 bg-amber-400/10 px-3 py-2 text-xs uppercase tracking-[0.22em] text-[var(--color-amber)]">
                      {(liveState?.operator_alerts ?? []).length} live alert
                      {(liveState?.operator_alerts ?? []).length > 1 ? "s" : ""}
                    </div>
                  ) : null}
                  <div className="rounded-full border border-white/8 px-3 py-2 text-xs uppercase tracking-[0.22em] text-[var(--color-text-soft)]">
                    {latestReady ? "report ready" : "report pending"}
                  </div>
                </div>
              </div>

              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex flex-wrap items-center gap-3 text-sm text-[var(--color-text-soft)]">
                  <div className="inline-flex items-center gap-2 rounded-full border border-white/8 px-3 py-2">
                    <BriefcaseBusiness className="size-4 text-white" />
                    {health.desk_slug ?? "desk"}
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-full border border-white/8 px-3 py-2">
                    <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-accent)]">
                      default portfolio
                    </div>
                    <span className="mono text-white">{portfolioSlug}</span>
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-full border border-white/8 px-3 py-2">
                    <Activity className="size-4 text-[var(--color-green)]" />
                    mt5-first workflow
                  </div>
                  {liveState?.generated_at ? (
                    <div className="inline-flex items-center gap-2 rounded-full border border-white/8 px-3 py-2">
                      <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-accent)]">
                        live feed
                      </div>
                      <span className="mono text-white">
                        {formatTimestamp(liveState.generated_at)}
                      </span>
                    </div>
                  ) : null}
                </div>
                <div className="flex flex-wrap gap-2">
                  {portfolios.map((portfolio) => (
                    <Link
                      key={portfolio.slug}
                      href={`${pathname}?portfolio=${portfolio.slug}`}
                      className={cn(
                        "rounded-full border px-3 py-2 text-xs uppercase tracking-[0.22em] transition duration-300 motion-safe:hover:-translate-y-[1px]",
                        portfolio.slug === portfolioSlug
                          ? "border-[var(--color-border-strong)] bg-[linear-gradient(180deg,rgba(216,155,73,0.16),rgba(216,155,73,0.06))] text-white"
                          : "border-white/8 text-[var(--color-text-soft)] hover:border-white/16 hover:text-white",
                      )}
                    >
                      {portfolio.slug}
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </header>

          <main
            data-desk-main
            className={cn(
              "px-5 py-8 md:px-8",
              routeBehavior.showInspectorToggle ? "2xl:px-10" : "",
            )}
          >
            {children}
          </main>
        </div>

        <aside
          data-desk-side
          className={cn(
            "sticky top-0 hidden h-screen overflow-y-auto border-l border-white/8 bg-[rgba(8,9,11,0.72)] px-5 py-6 backdrop-blur-2xl",
            routeBehavior.dockedRailClass,
          )}
        >
          <DeskInspectorContent health={health} alerts={alerts} audit={audit} jobsStatus={jobsStatus} liveState={liveState} transport={transport} />
        </aside>
      </div>

      {routeBehavior.showInspectorToggle ? (
        <>
          <div
            className={cn(
              "fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px] transition",
              contextOpen ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0",
            )}
            onClick={() => setOpenContextKey(null)}
          />
          <aside
            className={cn(
              "fixed inset-y-0 right-0 z-50 flex w-[min(92vw,360px)] flex-col border-l border-white/10 bg-[rgba(8,9,11,0.94)] px-5 py-5 shadow-[0_24px_80px_rgba(0,0,0,0.45)] backdrop-blur-2xl transition duration-300",
              contextOpen ? "translate-x-0" : "translate-x-full",
            )}
          >
            <div className="mb-4 flex items-center justify-between">
              <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-accent)]">
                Operator panel
              </div>
              <button
                type="button"
                onClick={() => setOpenContextKey(null)}
                className="inline-flex size-9 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-[var(--color-text-soft)] transition hover:border-[var(--color-border-strong)] hover:text-white"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto pr-1">
              <DeskInspectorContent health={health} alerts={alerts} audit={audit} jobsStatus={jobsStatus} liveState={liveState} transport={transport} />
            </div>
          </aside>
        </>
      ) : null}
    </div>
  );
}

function DeskInspectorContent({
  health,
  alerts,
  audit,
  jobsStatus,
  liveState,
  transport,
}: {
  health: HealthResponse;
  alerts: AlertSummary[];
  audit: AuditEventResponse[];
  jobsStatus: WorkerStatusResponse | null;
  liveState: MT5LiveStateResponse | null;
  transport: "stream" | "polling" | "connecting";
}) {
  const dbDependency = health.dependencies?.database as
    | { reachable?: boolean; schema_ready?: boolean }
    | undefined;
  const mt5Dependency = health.dependencies?.mt5 as
    | { reachable?: boolean | null; mode?: string }
    | undefined;
  const mt5LiveDependency = health.dependencies?.mt5_live as
    | { reachable?: boolean | null; mode?: string; stale?: boolean; detail?: string; generated_at?: string }
    | undefined;

  return (
    <div className="flex flex-col gap-6">
      <section className="surface rounded-[1.6rem] p-5">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          System
        </div>
        <div className="mt-4 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">Portfolios</span>
            <span className="mono text-lg text-white">{health.portfolio_count}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">Latest report</span>
            <span className="mono text-xs text-[var(--color-text-muted)]">
              {health.latest_artifacts.daily_report ? "ready" : "pending"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">Database</span>
            <span className="mono text-xs text-[var(--color-text-muted)]">
              {dbDependency?.reachable ? (dbDependency.schema_ready ? "ready" : "pending") : "offline"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">MT5 bridge</span>
            <span className="mono text-xs text-[var(--color-text-muted)]">
              {mt5Dependency?.reachable === true
                ? "reachable"
                : mt5Dependency?.reachable === false
                  ? "offline"
                  : mt5Dependency?.mode ?? "n/a"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">Live feed</span>
            <span className="mono text-xs text-[var(--color-text-muted)]">
              {liveState
                ? `${liveState.status}/${transport}`
                : mt5LiveDependency?.reachable === false
                  ? "offline"
                  : "pending"}
            </span>
          </div>
        </div>
      </section>

      <section className="surface rounded-[1.6rem] p-5">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Live Bridge
        </div>
        <div className="mt-4 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">Status</span>
            <span className="mono text-xs text-[var(--color-text-muted)]">
              {liveState?.status ?? mt5LiveDependency?.detail ?? "unknown"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">Transport</span>
            <span className="mono text-xs text-[var(--color-text-muted)]">{transport}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">Freshness</span>
            <span className="mono text-xs text-[var(--color-text-muted)]">
              {liveState?.generated_at
                ? formatTimestamp(liveState.generated_at)
                : mt5LiveDependency?.generated_at
                  ? formatTimestamp(mt5LiveDependency.generated_at)
                  : "n/a"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-[var(--color-text-soft)]">Sequence</span>
            <span className="mono text-xs text-[var(--color-text-muted)]">
              {liveState?.sequence ?? (health.dependencies?.mt5_live as { sequence?: number } | undefined)?.sequence ?? "n/a"}
            </span>
          </div>
          <div className="text-sm text-[var(--color-text-soft)]">
            {liveState?.last_error
              ? liveState.last_error
              : liveState?.stale
                ? "Live bridge connected but stale. The desk is in degraded read mode until MT5 recovers."
                : "The desk shell is following the MT5 live bridge and can fall back to polling if streaming drops."}
          </div>
        </div>
      </section>

      <section className="surface rounded-[1.6rem] p-5">
        <div className="flex items-center justify-between">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Worker
          </div>
          <div className="rounded-full border border-white/8 px-2 py-1 text-xs text-[var(--color-text-soft)]">
            {jobsStatus?.database_ready ? "db ready" : "db pending"}
          </div>
        </div>
        <div className="mt-4 space-y-4">
          {jobsStatus ? (
            Object.entries(jobsStatus.jobs).map(([jobName, job]) => (
              <div key={jobName} className="border-t border-white/8 pt-4 first:border-t-0 first:pt-0">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold text-white">{jobName}</div>
                  <div className="mono text-xs text-[var(--color-text-muted)]">{job.state}</div>
                </div>
                <div className="mt-2 text-sm text-[var(--color-text-soft)]">
                  Interval {job.interval_seconds}s
                  {job.last_run_at ? `, last run ${formatTimestamp(job.last_run_at)}` : ", never ran"}
                </div>
              </div>
            ))
          ) : (
            <div className="text-sm text-[var(--color-text-muted)]">
              Worker status unavailable.
            </div>
          )}
        </div>
      </section>

      <section className="surface rounded-[1.6rem] p-5">
        <div className="flex items-center justify-between">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Live Operator Alerts
          </div>
          <div className="rounded-full border border-white/8 px-2 py-1 text-xs text-[var(--color-text-soft)]">
            {(liveState?.operator_alerts ?? []).length}
          </div>
        </div>
        <div className="mt-4 space-y-4">
          {(liveState?.operator_alerts ?? []).length === 0 ? (
            <div className="text-sm text-[var(--color-text-muted)]">
              No live bridge incident is currently flagged.
            </div>
          ) : (
            (liveState?.operator_alerts ?? []).slice(0, 4).map((alert) => (
              <div key={`${alert.code}:${JSON.stringify(alert.context)}`} className="border-t border-white/8 pt-4 first:border-t-0 first:pt-0">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-white">{alert.code}</div>
                    <div className="mt-1 text-sm leading-6 text-[var(--color-text-soft)]">
                      {alert.message}
                    </div>
                  </div>
                  <AlertTriangle
                    className={cn("mt-1 size-4 shrink-0", severityTone(alert.severity))}
                  />
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="surface rounded-[1.6rem] p-5">
        <div className="flex items-center justify-between">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Active Alerts
          </div>
          <div className="rounded-full border border-white/8 px-2 py-1 text-xs text-[var(--color-text-soft)]">
            {alerts.length}
          </div>
        </div>
        <div className="mt-4 space-y-4">
          {alerts.length === 0 ? (
            <div className="text-sm text-[var(--color-text-muted)]">
              No active alerts yet.
            </div>
          ) : (
            alerts.slice(0, 6).map((alert) => (
              <div key={alert.id} className="border-t border-white/8 pt-4 first:border-t-0 first:pt-0">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-white">{alert.code}</div>
                    <div className="mt-1 text-sm leading-6 text-[var(--color-text-soft)]">
                      {alert.message}
                    </div>
                  </div>
                  <AlertTriangle
                    className={cn("mt-1 size-4 shrink-0", severityTone(alert.severity))}
                  />
                </div>
                <div className="mt-2 text-xs text-[var(--color-text-muted)]">
                  {formatTimestamp(alert.created_at)}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="surface rounded-[1.6rem] p-5">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Audit Trail
        </div>
        <div className="mt-4 space-y-4">
          {audit.slice(0, 6).map((event) => (
            <div key={event.id} className="border-t border-white/8 pt-4 first:border-t-0 first:pt-0">
              <div className="text-sm font-semibold text-white">{event.action_type}</div>
              <div className="mt-1 text-sm text-[var(--color-text-soft)]">
                {event.actor} on {event.object_type ?? "object"}
              </div>
              <div className="mt-2 text-xs text-[var(--color-text-muted)]">
                {formatTimestamp(event.created_at)}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

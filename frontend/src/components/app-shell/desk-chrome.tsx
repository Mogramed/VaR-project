"use client";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  BriefcaseBusiness,
  ChevronDown,
  FileText,
  Flame,
  Gauge,
  Landmark,
  Orbit,
  Radar,
  ShieldCheck,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useState } from "react";
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
  { href: "/desk/incidents", label: "Incidents", icon: AlertTriangle },
  { href: "/desk/universe", label: "Universe", icon: Orbit },
  { href: "/desk/models", label: "Models", icon: Radar },
  { href: "/desk/attribution", label: "Attribution", icon: BarChart3 },
  { href: "/desk/capital", label: "Capital", icon: Landmark },
  { href: "/desk/decisions", label: "Decisions", icon: ShieldCheck },
  { href: "/desk/execution", label: "Execution", icon: BriefcaseBusiness },
  { href: "/desk/stress", label: "Stress", icon: Flame },
  { href: "/desk/blotter", label: "Blotter", icon: Orbit },
  { href: "/desk/reports", label: "Reports", icon: FileText },
] as const;

function StatusDot({ status }: { status: "ok" | "warn" | "off" }) {
  const color =
    status === "ok"
      ? "bg-[var(--color-green)]"
      : status === "warn"
        ? "bg-[var(--color-amber)]"
        : "bg-[var(--color-text-muted)]";
  return <span className={cn("inline-block size-1.5 rounded-full", color)} />;
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
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [portfolioDropdownOpen, setPortfolioDropdownOpen] = useState(false);
  const { liveState, transport } = useMt5LiveState(portfolioSlug ?? undefined);

  const apiStatus = health.status === "ok" ? "ok" : "off";
  const mt5Status = liveState?.status === "ok" ? "ok" : liveState?.degraded ? "warn" : "off";
  const alertCount = (liveState?.operator_alerts ?? []).length + alerts.length;

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--color-bg)]">
      {/* ── Sidebar ── */}
      <aside
        data-desk-rail
        className="hidden w-[52px] shrink-0 flex-col border-r border-[var(--color-border)] bg-[var(--color-bg)] xl:flex"
      >
        {/* Logo */}
        <Link
          href="/en"
          className="flex h-11 items-center justify-center border-b border-[var(--color-border)] text-xs font-bold tracking-widest text-[var(--color-accent)]"
        >
          VR
        </Link>

        {/* Nav items */}
        <nav className="flex flex-1 flex-col gap-0.5 overflow-y-auto px-1.5 py-2">
          {navItems.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={`${href}?portfolio=${portfolioSlug}`}
                title={label}
                className={cn(
                  "group relative flex h-9 w-full items-center justify-center rounded-[var(--radius-md)] transition-colors",
                  active
                    ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                    : "text-[var(--color-text-muted)] hover:bg-[var(--color-surface)] hover:text-[var(--color-text-soft)]",
                )}
              >
                <Icon className="size-4" strokeWidth={active ? 2 : 1.5} />
                {/* Tooltip */}
                <span className="pointer-events-none absolute left-full ml-2 whitespace-nowrap rounded-[var(--radius-sm)] bg-[var(--color-surface-strong)] px-2 py-1 text-[11px] text-[var(--color-text)] opacity-0 shadow-lg transition-opacity group-hover:opacity-100">
                  {label}
                </span>
                {/* Active indicator */}
                {active ? (
                  <span className="absolute left-0 top-1/2 h-4 w-[2px] -translate-y-1/2 rounded-r-full bg-[var(--color-accent)]" />
                ) : null}
              </Link>
            );
          })}
        </nav>

        {/* Bottom */}
        <div className="border-t border-[var(--color-border)] px-1.5 py-2">
          <Link
            href="/fr"
            className="flex h-9 w-full items-center justify-center rounded-[var(--radius-md)] text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface)] hover:text-[var(--color-text-soft)]"
          >
            FR
          </Link>
        </div>
      </aside>

      {/* ── Main area ── */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* ── Topbar ── */}
        <header
          data-desk-topbar
          className="flex h-11 shrink-0 items-center gap-3 border-b border-[var(--color-border)] bg-[var(--color-bg)] px-4"
        >
          {/* Left: Portfolio + status */}
          <div className="flex items-center gap-3">
            {/* Portfolio dropdown */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setPortfolioDropdownOpen(!portfolioDropdownOpen)}
                className="flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 text-xs font-medium text-[var(--color-text)] transition-colors hover:border-[var(--color-border-strong)]"
              >
                <BriefcaseBusiness className="size-3 text-[var(--color-text-muted)]" />
                {portfolioSlug}
                <ChevronDown className="size-3 text-[var(--color-text-muted)]" />
              </button>
              {portfolioDropdownOpen ? (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setPortfolioDropdownOpen(false)}
                  />
                  <div className="absolute left-0 top-full z-50 mt-1 min-w-[160px] rounded-[var(--radius-md)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] py-1 shadow-xl">
                    {portfolios.map((p) => (
                      <Link
                        key={p.slug}
                        href={`${pathname}?portfolio=${p.slug}`}
                        onClick={() => setPortfolioDropdownOpen(false)}
                        className={cn(
                          "flex items-center gap-2 px-3 py-1.5 text-xs transition-colors",
                          p.slug === portfolioSlug
                            ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                            : "text-[var(--color-text-soft)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]",
                        )}
                      >
                        {p.slug}
                      </Link>
                    ))}
                  </div>
                </>
              ) : null}
            </div>

            {/* Status indicators */}
            <div className="hidden items-center gap-3 text-[11px] text-[var(--color-text-muted)] md:flex">
              <span className="flex items-center gap-1.5">
                <StatusDot status={apiStatus as "ok" | "warn" | "off"} />
                API
              </span>
              <span className="flex items-center gap-1.5">
                <StatusDot status={mt5Status as "ok" | "warn" | "off"} />
                MT5
                {transport === "stream" ? " · SSE" : transport === "polling" ? " · Poll" : ""}
              </span>
              {liveState?.generated_at ? (
                <span className="mono text-[10px]">
                  {formatTimestamp(liveState.generated_at)}
                </span>
              ) : null}
            </div>
          </div>

          {/* Center spacer */}
          <div className="flex-1" />

          {/* Right: Actions + alerts */}
          <div className="flex items-center gap-2">
            <OperatorActions portfolioSlug={portfolioSlug} />

            {alertCount > 0 ? (
              <button
                type="button"
                onClick={() => setInspectorOpen(true)}
                className="flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--color-red)]/20 bg-[var(--color-red-soft)] px-2 text-[11px] font-medium text-[var(--color-red)] transition-colors hover:bg-[var(--color-red)]/20"
              >
                <AlertTriangle className="size-3" />
                {alertCount}
              </button>
            ) : null}

            <button
              type="button"
              onClick={() => setInspectorOpen(true)}
              className="flex h-7 items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--color-border)] px-2 text-[11px] text-[var(--color-text-muted)] transition-colors hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-soft)]"
            >
              Panel
            </button>
          </div>
        </header>

        {/* ── Page content ── */}
        <main
          data-desk-main
          className="flex-1 overflow-y-auto px-4 py-4 lg:px-6 lg:py-5"
        >
          {children}
        </main>
      </div>

      {/* ── Inspector drawer ── */}
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/40 transition-opacity",
          inspectorOpen ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={() => setInspectorOpen(false)}
      />
      <aside
        className={cn(
          "fixed inset-y-0 right-0 z-50 flex w-[320px] flex-col border-l border-[var(--color-border)] bg-[var(--color-bg)] transition-transform duration-200",
          inspectorOpen ? "translate-x-0" : "translate-x-full",
        )}
      >
        <div className="flex h-11 items-center justify-between border-b border-[var(--color-border)] px-4">
          <span className="text-xs font-medium text-[var(--color-text-soft)]">
            Operator Panel
          </span>
          <button
            type="button"
            onClick={() => setInspectorOpen(false)}
            className="flex size-6 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface)] hover:text-[var(--color-text)]"
          >
            <X className="size-3.5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <InspectorContent
            health={health}
            alerts={alerts}
            audit={audit}
            jobsStatus={jobsStatus}
            liveState={liveState}
            transport={transport}
          />
        </div>
      </aside>
    </div>
  );
}

/* ── Inspector content ── */

function InspectorContent({
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
  const dbDep = health.dependencies?.database as { reachable?: boolean; schema_ready?: boolean } | undefined;
  const mt5Dep = health.dependencies?.mt5 as { reachable?: boolean | null; mode?: string } | undefined;

  return (
    <div className="space-y-4">
      {/* System */}
      <InspectorSection title="System">
        <InspectorRow label="API" value={health.status} />
        <InspectorRow label="Database" value={dbDep?.reachable ? "ready" : "offline"} />
        <InspectorRow label="MT5 Bridge" value={mt5Dep?.reachable ? "connected" : mt5Dep?.mode ?? "n/a"} />
        <InspectorRow label="Transport" value={transport} />
        <InspectorRow label="Portfolios" value={String(health.portfolio_count)} />
      </InspectorSection>

      {/* Live bridge */}
      <InspectorSection title="Live Bridge">
        <InspectorRow label="Status" value={liveState?.status ?? "pending"} />
        <InspectorRow label="Sequence" value={String(liveState?.sequence ?? "n/a")} />
        <InspectorRow label="Updated" value={liveState?.generated_at ? formatTimestamp(liveState.generated_at) : "n/a"} />
        {liveState?.last_error ? (
          <p className="mt-1 text-[11px] leading-relaxed text-[var(--color-red)]">
            {liveState.last_error}
          </p>
        ) : null}
      </InspectorSection>

      {/* Worker */}
      {jobsStatus ? (
        <InspectorSection title="Worker">
          {Object.entries(jobsStatus.jobs).map(([name, job]) => (
            <InspectorRow
              key={name}
              label={name}
              value={`${job.state} · ${job.interval_seconds}s`}
            />
          ))}
        </InspectorSection>
      ) : null}

      {/* Live alerts */}
      {(liveState?.operator_alerts ?? []).length > 0 ? (
        <InspectorSection title={`Live Alerts (${(liveState?.operator_alerts ?? []).length})`}>
          {(liveState?.operator_alerts ?? []).slice(0, 5).map((alert) => (
            <div key={`${alert.code}:${JSON.stringify(alert.context)}`} className="border-b border-[var(--color-border)] pb-2 last:border-b-0 last:pb-0">
              <div className="flex items-start justify-between gap-2">
                <span className="text-[11px] font-medium text-[var(--color-text)]">{alert.code}</span>
                <AlertSeverityBadge severity={alert.severity} />
              </div>
              <p className="mt-0.5 text-[11px] leading-relaxed text-[var(--color-text-muted)]">
                {alert.message}
              </p>
            </div>
          ))}
        </InspectorSection>
      ) : null}

      {/* Persisted alerts */}
      {alerts.length > 0 ? (
        <InspectorSection title={`Alerts (${alerts.length})`}>
          {alerts.slice(0, 5).map((alert) => (
            <div key={alert.id} className="border-b border-[var(--color-border)] pb-2 last:border-b-0 last:pb-0">
              <div className="flex items-start justify-between gap-2">
                <span className="text-[11px] font-medium text-[var(--color-text)]">{alert.code}</span>
                <AlertSeverityBadge severity={alert.severity} />
              </div>
              <p className="mt-0.5 text-[11px] text-[var(--color-text-muted)]">
                {alert.message}
              </p>
            </div>
          ))}
        </InspectorSection>
      ) : null}

      {/* Audit */}
      <InspectorSection title="Audit Trail">
        {audit.slice(0, 6).map((event) => (
          <div key={event.id} className="border-b border-[var(--color-border)] pb-2 last:border-b-0 last:pb-0">
            <div className="flex items-center justify-between gap-2 text-[11px]">
              <span className="font-medium text-[var(--color-text)]">{event.action_type}</span>
              <span className="mono text-[var(--color-text-muted)]">{formatTimestamp(event.created_at)}</span>
            </div>
            <p className="text-[11px] text-[var(--color-text-muted)]">{event.actor}</p>
          </div>
        ))}
      </InspectorSection>
    </div>
  );
}

function InspectorSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
      <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
        {title}
      </h4>
      <div className="space-y-1.5">{children}</div>
    </section>
  );
}

function InspectorRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-[11px]">
      <span className="text-[var(--color-text-muted)]">{label}</span>
      <span className="mono text-[var(--color-text-soft)]">{value}</span>
    </div>
  );
}

function AlertSeverityBadge({ severity }: { severity: string }) {
  const s = severity.toLowerCase();
  const color = s.includes("breach") || s.includes("critical")
    ? "text-[var(--color-red)] bg-[var(--color-red-soft)]"
    : s.includes("warn")
      ? "text-[var(--color-amber)] bg-[var(--color-amber-soft)]"
      : "text-[var(--color-green)] bg-[var(--color-green-soft)]";
  return (
    <span className={cn("rounded-[2px] px-1 py-0.5 text-[9px] font-semibold uppercase", color)}>
      {severity}
    </span>
  );
}

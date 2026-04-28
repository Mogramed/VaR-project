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
  Settings2,
  ShieldCheck,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import type {
  AlertSummary,
  AuditEventResponse,
  HealthResponse,
  MT5LiveStateResponse,
  PortfolioSummary,
  WorkerStatusResponse,
  MT5AccountsResponse,
} from "@/lib/api/types";
import { OperatorActions } from "@/components/app-shell/operator-actions";
import { DeskLiveProvider, useDeskLive } from "@/components/app-shell/desk-live-provider";
import { deriveLiveRuntimeDiagnostics } from "@/components/app-shell/live-runtime-phase";
import { api } from "@/lib/api/client";
import { alertPriorityCode, dedupeOperatorAlerts, dedupePersistedAlerts } from "@/lib/alerts";
import { cn, formatTimestamp } from "@/lib/utils";
import { useRelativeTime } from "@/lib/use-relative-time";
import { useQuery } from "@tanstack/react-query";
import { DashboardConfigPanel } from "@/components/app-shell/dashboard-config-panel";
import { DashboardPreferencesProvider } from "@/lib/dashboard-preferences-context";
import { useDashboardPreferences, type PageId } from "@/lib/dashboard-preferences";

const navItems: ReadonlyArray<{ href: string; label: string; icon: typeof Gauge; pageId: PageId }> = [
  { href: "/desk", label: "Overview", icon: Gauge, pageId: "overview" },
  { href: "/desk/live", label: "MT5 Ops", icon: Activity, pageId: "live" },
  { href: "/desk/incidents", label: "Incidents", icon: AlertTriangle, pageId: "incidents" },
  { href: "/desk/universe", label: "Universe", icon: Orbit, pageId: "universe" },
  { href: "/desk/models", label: "Models", icon: Radar, pageId: "models" },
  { href: "/desk/attribution", label: "Attribution", icon: BarChart3, pageId: "attribution" },
  { href: "/desk/capital", label: "Capital", icon: Landmark, pageId: "capital" },
  { href: "/desk/decisions", label: "Decisions", icon: ShieldCheck, pageId: "decisions" },
  { href: "/desk/alpha/features", label: "Alpha Features", icon: Radar, pageId: "alpha-features" },
  { href: "/desk/alpha/performance", label: "Alpha Replay", icon: BarChart3, pageId: "alpha-performance" },
  { href: "/desk/execution", label: "Execution", icon: BriefcaseBusiness, pageId: "execution" },
  { href: "/desk/stress", label: "Stress", icon: Flame, pageId: "stress" },
  { href: "/desk/blotter", label: "Blotter", icon: Orbit, pageId: "blotter" },
  { href: "/desk/reports", label: "Reports", icon: FileText, pageId: "reports" },
];

const mobileNavItems: ReadonlyArray<{ href: string; label: string; icon: typeof Gauge; pageId: PageId }> = [
  { href: "/desk", label: "Overview", icon: Gauge, pageId: "overview" },
  { href: "/desk/live", label: "MT5 Ops", icon: Activity, pageId: "live" },
  { href: "/desk/capital", label: "Capital", icon: Landmark, pageId: "capital" },
  { href: "/desk/execution", label: "Execute", icon: BriefcaseBusiness, pageId: "execution" },
  { href: "/desk/blotter", label: "Blotter", icon: Orbit, pageId: "blotter" },
];

const MT5_ACCOUNT_STORAGE_KEY = "desk:active-mt5-account";
const EMPTY_MT5_ACCOUNTS: NonNullable<MT5AccountsResponse["accounts"]> = [];

function StatusDot({ status }: { status: "ok" | "warn" | "off" }) {
  const color =
    status === "ok"
      ? "bg-[var(--color-green)]"
      : status === "warn"
        ? "bg-[var(--color-amber)]"
        : "bg-[var(--color-text-muted)]";
  return <span className={cn("inline-block size-1.5 rounded-full", color)} />;
}

function topbarButtonClassName(extra?: string) {
  return cn(
    "flex h-8 items-center gap-2 rounded-[12px] border border-[var(--color-border)] bg-[linear-gradient(180deg,rgba(24,28,36,0.96),rgba(11,14,20,0.98))] px-3 text-[11px] font-semibold tracking-[0.02em] text-[var(--color-text-soft)] transition-all duration-150 hover:border-[var(--color-border-strong)] hover:text-[var(--color-text)] hover:shadow-[0_10px_30px_rgba(0,0,0,0.18)]",
    extra,
  );
}

function topbarPanelClassName(extra?: string) {
  return cn(
    "rounded-[18px] border border-[var(--color-border)] bg-[linear-gradient(180deg,rgba(18,22,30,0.96),rgba(9,11,17,0.99))] px-3 py-2 shadow-[0_18px_40px_rgba(0,0,0,0.16),inset_0_1px_0_rgba(255,255,255,0.03)]",
    extra,
  );
}

function RuntimeChip({
  label,
  status,
  detail,
  emphasis,
}: {
  label: string;
  status: "ok" | "warn" | "off";
  detail?: string | null;
  emphasis?: string | null;
}) {
  return (
    <div className="flex h-8 items-center gap-2 rounded-[12px] border border-[var(--color-border)] bg-[linear-gradient(180deg,rgba(20,23,31,0.96),rgba(10,12,18,0.98))] px-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
      <StatusDot status={status} />
      <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--color-text-muted)]">
        {label}
      </span>
      {detail ? (
        <span className="text-[11px] text-[var(--color-text-soft)]">{detail}</span>
      ) : null}
      {emphasis ? (
        <span className="mono text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-text-muted)]">
          {emphasis}
        </span>
      ) : null}
    </div>
  );
}

function isAlertScopedToPortfolio(
  alert: AlertSummary,
  activePortfolioId: number | null,
  activePortfolioSlug: string | null,
) {
  if (activePortfolioId == null) {
    return true;
  }
  const alertPortfolioId =
    typeof alert.portfolio_id === "number" ? alert.portfolio_id : null;
  if (alertPortfolioId == null) {
    const contextPortfolioSlug = String(
      (alert.context as { portfolio_slug?: string } | undefined)?.portfolio_slug ?? "",
    )
      .trim()
      .toLowerCase();
    if (!contextPortfolioSlug || !activePortfolioSlug) {
      return true;
    }
    return contextPortfolioSlug === activePortfolioSlug.toLowerCase();
  }
  return alertPortfolioId === activePortfolioId;
}

function canonicalPortfolioSlug(
  requestedSlug: string | null,
  healthSlug: string | null | undefined,
  portfolios: PortfolioSummary[],
): string {
  const fallbackSlug = String(healthSlug ?? portfolios[0]?.slug ?? requestedSlug ?? "default").trim();
  if (!requestedSlug || !requestedSlug.trim()) {
    return fallbackSlug;
  }

  const trimmed = requestedSlug.trim();
  const normalized = trimmed.toLowerCase();
  if (["default", "main", "primary"].includes(normalized)) {
    return fallbackSlug;
  }

  const exact = portfolios.find((portfolio) => String(portfolio.slug) === trimmed);
  if (exact) {
    return String(exact.slug);
  }

  const caseInsensitive = portfolios.find(
    (portfolio) => String(portfolio.slug).toLowerCase() === normalized,
  );
  if (caseInsensitive) {
    return String(caseInsensitive.slug);
  }

  if (normalized.replace(/-/g, "_") === "fx_eur_20k") {
    return fallbackSlug;
  }

  return fallbackSlug;
}

function canonicalAccountId(
  requestedAccountId: string | null,
  rememberedAccountId: string | null,
  accountsPayload: MT5AccountsResponse | null | undefined,
): string | null {
  const accounts = Array.isArray(accountsPayload?.accounts) ? accountsPayload.accounts : [];
  if (accounts.length === 0) {
    const fallback = requestedAccountId ?? rememberedAccountId ?? accountsPayload?.active_account_id ?? null;
    return fallback ? fallback.trim() || null : null;
  }
  const normalizedByLower = new Map<string, string>();
  for (const account of accounts) {
    const id = String(account.account_id ?? "").trim();
    if (!id) continue;
    normalizedByLower.set(id.toLowerCase(), id);
  }
  const defaultIdRaw =
    accountsPayload?.active_account_id
    ?? accounts.find((account) => account.is_default)?.account_id
    ?? accounts[0]?.account_id
    ?? null;
  const defaultId = String(defaultIdRaw ?? "").trim() || null;
  const resolveCandidate = (candidate: string | null) => {
    const normalized = String(candidate ?? "").trim();
    if (!normalized) return null;
    const lowered = normalized.toLowerCase();
    if (["default", "main", "primary"].includes(lowered) && defaultId) {
      return defaultId;
    }
    return normalizedByLower.get(lowered) ?? null;
  };
  return (
    resolveCandidate(requestedAccountId)
    ?? resolveCandidate(rememberedAccountId)
    ?? resolveCandidate(defaultId)
    ?? accounts[0].account_id
  );
}

export function DeskChrome({
  children,
  health,
  portfolios,
  activeAlerts,
  recentAlerts,
  audit,
  jobsStatus,
  initialMt5Accounts,
}: {
  children: React.ReactNode;
  health: HealthResponse;
  portfolios: PortfolioSummary[];
  activeAlerts: AlertSummary[];
  recentAlerts: AlertSummary[];
  audit: AuditEventResponse[];
  jobsStatus: WorkerStatusResponse | null;
  initialMt5Accounts: MT5AccountsResponse | null;
}) {
  const router = useRouter();
  const pathname = usePathname() ?? "/desk";
  const searchParams = useSearchParams();
  const searchParamsText = searchParams?.toString() ?? "";
  const requestedPortfolioSlug = searchParams?.get("portfolio") ?? null;
  const requestedAccountId = searchParams?.get("account") ?? null;
  const [rememberedAccountId] = useState<string | null>(() => {
    if (typeof window === "undefined") {
      return null;
    }
    const stored = window.localStorage.getItem(MT5_ACCOUNT_STORAGE_KEY);
    const normalized = stored && stored.trim() ? stored.trim() : null;
    return normalized;
  });
  const mt5AccountsQuery = useQuery({
    queryKey: ["mt5-accounts"],
    queryFn: () => api.mt5Accounts(),
    initialData: initialMt5Accounts ?? undefined,
    staleTime: 30_000,
    gcTime: 5 * 60_000,
  });
  const mt5Accounts = mt5AccountsQuery.data ?? null;
  const portfolioSlug = useMemo(
    () =>
      canonicalPortfolioSlug(
        requestedPortfolioSlug,
        health.portfolio_slug,
        portfolios,
      ),
    [requestedPortfolioSlug, health.portfolio_slug, portfolios],
  );
  const activeAccountId = useMemo(
    () => canonicalAccountId(requestedAccountId, rememberedAccountId, mt5Accounts),
    [requestedAccountId, rememberedAccountId, mt5Accounts],
  );

  useEffect(() => {
    if (typeof window === "undefined" || !activeAccountId) {
      return;
    }
    window.localStorage.setItem(MT5_ACCOUNT_STORAGE_KEY, activeAccountId);
  }, [activeAccountId]);

  useEffect(() => {
    const normalizedPortfolio = String(requestedPortfolioSlug ?? "").trim();
    const normalizedAccount = String(requestedAccountId ?? "").trim();
    const needsPortfolioUpdate = Boolean(portfolioSlug) && normalizedPortfolio !== portfolioSlug;
    const needsAccountUpdate = activeAccountId == null
      ? normalizedAccount.length > 0
      : normalizedAccount !== activeAccountId;
    if (!needsPortfolioUpdate && !needsAccountUpdate) {
      return;
    }
    const params = new URLSearchParams(searchParamsText);
    if (portfolioSlug) {
      params.set("portfolio", portfolioSlug);
    }
    if (activeAccountId) {
      params.set("account", activeAccountId);
    } else {
      params.delete("account");
    }
    const query = params.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }, [
    activeAccountId,
    pathname,
    portfolioSlug,
    requestedAccountId,
    requestedPortfolioSlug,
    router,
    searchParamsText,
  ]);

  return (
    <DeskLiveProvider
      portfolioSlug={portfolioSlug ?? undefined}
      accountId={activeAccountId ?? undefined}
    >
      <DeskChromeFrame
        pathname={pathname}
        portfolioSlug={portfolioSlug}
        accountId={activeAccountId}
        mt5Accounts={mt5Accounts}
        health={health}
        portfolios={portfolios}
        activeAlerts={activeAlerts}
        recentAlerts={recentAlerts}
        audit={audit}
        jobsStatus={jobsStatus}
      >
        {children}
      </DeskChromeFrame>
    </DeskLiveProvider>
  );
}

function DeskChromeFrame({
  children,
  pathname,
  portfolioSlug,
  accountId,
  mt5Accounts,
  health,
  portfolios,
  activeAlerts,
  recentAlerts,
  audit,
  jobsStatus,
}: {
  children: React.ReactNode;
  pathname: string;
  portfolioSlug: string;
  accountId: string | null;
  mt5Accounts: MT5AccountsResponse | null;
  health: HealthResponse;
  portfolios: PortfolioSummary[];
  activeAlerts: AlertSummary[];
  recentAlerts: AlertSummary[];
  audit: AuditEventResponse[];
  jobsStatus: WorkerStatusResponse | null;
}) {
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);
  const [portfolioDropdownOpen, setPortfolioDropdownOpen] = useState(false);
  const [accountDropdownOpen, setAccountDropdownOpen] = useState(false);
  const { liveState, heartbeatAt, transport } = useDeskLive();
  const dashboardPrefs = useDashboardPreferences();
  const visibleNavItems = useMemo(
    () => navItems.filter((item) => dashboardPrefs.isPageVisible(item.pageId)),
    [dashboardPrefs],
  );
  const visibleMobileNavItems = useMemo(
    () => mobileNavItems.filter((item) => dashboardPrefs.isPageVisible(item.pageId)),
    [dashboardPrefs],
  );
  const mt5AccountsList = mt5Accounts?.accounts;
  const accountOptions = useMemo(
    () => (Array.isArray(mt5AccountsList) ? mt5AccountsList : EMPTY_MT5_ACCOUNTS),
    [mt5AccountsList],
  );
  const activeAccountLabel = useMemo(() => {
    if (!accountId) return "account";
    const matched = accountOptions.find((item) => item.account_id === accountId);
    return matched?.label ?? accountId;
  }, [accountId, accountOptions]);
  const buildDeskHref = (href: string, nextPortfolioSlug = portfolioSlug, nextAccountId = accountId) => {
    const params = new URLSearchParams();
    if (nextPortfolioSlug) {
      params.set("portfolio", nextPortfolioSlug);
    }
    if (nextAccountId) {
      params.set("account", nextAccountId);
    }
    const query = params.toString();
    return query ? `${href}?${query}` : href;
  };
  const liveAlerts = [...dedupeOperatorAlerts(liveState?.operator_alerts ?? [])].sort((left, right) => {
    const rankDelta = alertPriorityCode(left.code) - alertPriorityCode(right.code);
    if (rankDelta !== 0) {
      return rankDelta;
    }
    return left.code.localeCompare(right.code);
  });
  const activePortfolioSlug = portfolioSlug ?? null;
  const activePortfolioId =
    portfolios.find((portfolio) => portfolio.slug === portfolioSlug)?.id ?? null;
  const persistedActiveAlerts = dedupePersistedAlerts(activeAlerts).filter((alert) =>
    isAlertScopedToPortfolio(alert, activePortfolioId, activePortfolioSlug),
  );
  const persistedRecentAlerts = dedupePersistedAlerts(recentAlerts).filter((alert) =>
    isAlertScopedToPortfolio(alert, activePortfolioId, activePortfolioSlug),
  );

  const apiStatus = health.status === "ok" ? "ok" : "off";
  const runtimeDiagnostics = deriveLiveRuntimeDiagnostics(liveState, transport);
  const runtimePhase = runtimeDiagnostics.phase;
  const mt5Status = runtimePhase === "live"
    ? "ok"
    : runtimePhase === "recovering" || runtimePhase === "degraded"
      ? "warn"
      : "off";
  const transportLabel = transport === "stream" ? "SSE" : transport === "polling" ? "Poll" : "";
  const phaseLabel = runtimePhase === "recovering"
    ? runtimeDiagnostics.isRetrying
      ? "retrying"
      : "recovering"
    : runtimePhase === "degraded"
      ? "delayed"
      : runtimePhase === "offline"
        ? "offline"
        : "";
  const retryDelayLabel = runtimeDiagnostics.retryInSeconds == null
    ? ""
    : runtimeDiagnostics.retryInSeconds >= 10
      ? `${Math.round(runtimeDiagnostics.retryInSeconds)}s`
      : `${runtimeDiagnostics.retryInSeconds.toFixed(1)}s`;
  const alertCount = liveAlerts.length;
  const freshestUpdateAt = heartbeatAt ?? liveState?.generated_at ?? null;
  const liveRelativeTime = useRelativeTime(freshestUpdateAt);
  const liveProfit = liveState?.account?.profit ?? null;

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--color-bg)]">
      {/* Sidebar */}
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
          {visibleNavItems.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={buildDeskHref(href)}
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

      {/* Main area */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Topbar */}
        <header
          data-desk-topbar
          className="grid shrink-0 gap-2.5 border-b border-[var(--color-border)] bg-[linear-gradient(180deg,rgba(13,15,20,0.98),rgba(8,10,14,0.98))] px-3 py-2.5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:px-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]"
        >
          {/* Left: portfolio + runtime */}
          <div className="flex min-w-0 flex-col gap-2">
            <section className={topbarPanelClassName()}>
              <div className="flex flex-wrap items-start justify-between gap-2.5">
                <div className="min-w-0">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--color-text-muted)]">
                    Desk Context
                  </div>
                </div>
                {freshestUpdateAt ? (
                  <div className="rounded-full border border-[var(--color-border)] bg-[rgba(255,255,255,0.02)] px-2.5 py-1 text-[10px] uppercase tracking-[0.16em] text-[var(--color-text-muted)]">
                    Updated {liveRelativeTime}
                  </div>
                ) : null}
              </div>

              <div className="mt-2.5 flex flex-wrap items-center gap-2">
                {/* Portfolio dropdown */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => {
                      setPortfolioDropdownOpen(!portfolioDropdownOpen);
                      setAccountDropdownOpen(false);
                    }}
                    className={topbarButtonClassName("max-w-[240px] text-[var(--color-text)]")}
                  >
                    <BriefcaseBusiness className="size-3 text-[var(--color-text-muted)]" />
                    <span className="max-w-[180px] truncate">{portfolioSlug}</span>
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
                            href={buildDeskHref(pathname, p.slug)}
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

                {/* MT5 account dropdown */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => {
                      setAccountDropdownOpen(!accountDropdownOpen);
                      setPortfolioDropdownOpen(false);
                    }}
                    className={topbarButtonClassName("max-w-[260px] text-[var(--color-text)]")}
                  >
                    <Activity className="size-3 text-[var(--color-text-muted)]" />
                    <span className="max-w-[170px] truncate">{activeAccountLabel}</span>
                    <ChevronDown className="size-3 text-[var(--color-text-muted)]" />
                  </button>
                  {accountDropdownOpen ? (
                    <>
                      <div
                        className="fixed inset-0 z-40"
                        onClick={() => setAccountDropdownOpen(false)}
                      />
                      <div className="absolute left-0 top-full z-50 mt-1 min-w-[220px] rounded-[var(--radius-md)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] py-1 shadow-xl">
                        {accountOptions.length === 0 ? (
                          <div className="px-3 py-2 text-xs text-[var(--color-text-muted)]">
                            No MT5 account available.
                          </div>
                        ) : (
                          accountOptions.map((account) => (
                            <Link
                              key={account.account_id}
                              href={buildDeskHref(pathname, portfolioSlug, account.account_id)}
                              onClick={() => setAccountDropdownOpen(false)}
                              className={cn(
                                "flex items-center justify-between gap-2 px-3 py-1.5 text-xs transition-colors",
                                account.account_id === accountId
                                  ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                                  : "text-[var(--color-text-soft)] hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]",
                              )}
                            >
                              <span className="truncate">{account.label}</span>
                              {account.is_default ? (
                                <span className="text-[10px] uppercase tracking-wider text-[var(--color-text-muted)]">
                                  default
                                </span>
                              ) : null}
                            </Link>
                          ))
                        )}
                      </div>
                    </>
                  ) : null}
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <RuntimeChip
                  label="API"
                  status={apiStatus as "ok" | "warn" | "off"}
                  detail={health.status === "ok" ? "healthy" : String(health.status || "offline")}
                />
                <div className="flex h-8 items-center gap-2 rounded-[12px] border border-[var(--color-border)] bg-[linear-gradient(180deg,rgba(20,23,31,0.96),rgba(10,12,18,0.98))] px-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                  {mt5Status === "ok" ? (
                    <span className="relative inline-block size-1.5">
                      <span className="absolute inset-0 animate-ping rounded-full bg-[var(--color-green)] opacity-40" />
                      <span className="relative inline-block size-1.5 rounded-full bg-[var(--color-green)]" />
                    </span>
                  ) : (
                    <StatusDot status={mt5Status as "ok" | "warn" | "off"} />
                  )}
                  <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--color-text-muted)]">
                    MT5
                  </span>
                  <span className="text-[11px] text-[var(--color-text-soft)]">
                    {[transportLabel, phaseLabel].filter(Boolean).join(" | ") || "linked"}
                  </span>
                  {runtimeDiagnostics.isRetrying && retryDelayLabel ? (
                    <span className="mono text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-amber)]">
                      retry {retryDelayLabel}
                    </span>
                  ) : null}
                  {runtimeDiagnostics.failureCount > 0 ? (
                    <span className="mono text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-red)]">
                      fail {runtimeDiagnostics.failureCount}
                    </span>
                  ) : null}
                </div>
                <RuntimeChip
                  label="Account"
                  status="off"
                  detail={activeAccountLabel}
                  emphasis={accountId ?? "n/a"}
                />
                {liveProfit != null ? (
                  <RuntimeChip
                    label="PnL"
                    status={liveProfit >= 0 ? "ok" : "warn"}
                    detail={`${liveProfit >= 0 ? "+" : ""}${liveProfit.toFixed(2)}`}
                  />
                ) : null}
                {freshestUpdateAt ? (
                  <RuntimeChip
                    label="Freshness"
                    status={mt5Status as "ok" | "warn" | "off"}
                    detail={liveRelativeTime}
                    emphasis={formatTimestamp(freshestUpdateAt)}
                  />
                ) : null}
              </div>
            </section>
          </div>

          {/* Right: operator controls */}
          <div className="flex min-w-0 flex-col gap-2">
            <section className={topbarPanelClassName("flex flex-wrap items-start justify-between gap-2.5")}>
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--color-text-muted)]">
                  Operator Controls
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {alertCount > 0 ? (
                  <button
                    type="button"
                    onClick={() => setInspectorOpen(true)}
                    className={topbarButtonClassName("border-[var(--color-red)]/22 bg-[linear-gradient(180deg,rgba(84,24,24,0.26),rgba(27,16,18,0.98))] text-[var(--color-red)]")}
                  >
                    <AlertTriangle className="size-3.5" />
                    <span>Alerts</span>
                    <span className="rounded-full border border-[currentColor]/20 bg-[currentColor]/10 px-1.5 py-0.5 text-[10px] leading-none">
                      {alertCount}
                    </span>
                  </button>
                ) : null}

                <button
                  type="button"
                  onClick={() => {
                    setConfigOpen(true);
                    setInspectorOpen(false);
                  }}
                  title="Configure my view"
                  className={topbarButtonClassName(
                    cn(
                      "hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]",
                      configOpen && "border-[var(--color-accent)]/24 bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
                    ),
                  )}
                >
                  <Settings2 className="size-3.5" />
                  <span>Configure</span>
                </button>

                <button
                  type="button"
                  onClick={() => setInspectorOpen(true)}
                  className={topbarButtonClassName(
                    inspectorOpen
                      ? "border-[var(--color-border-strong)] bg-[linear-gradient(180deg,rgba(31,35,46,0.98),rgba(15,18,24,0.98))] text-[var(--color-text)]"
                      : undefined,
                  )}
                >
                  <BarChart3 className="size-3.5" />
                  <span>Panel</span>
                </button>
              </div>
            </section>

            <OperatorActions portfolioSlug={portfolioSlug} accountId={accountId ?? undefined} />
          </div>
        </header>

        {/* Page content */}
        <main
          data-desk-main
          className="flex-1 overflow-y-auto px-4 py-4 pb-20 lg:px-6 lg:py-5 xl:pb-5"
        >
          <DashboardPreferencesProvider value={dashboardPrefs}>
            {children}
          </DashboardPreferencesProvider>
        </main>

        {/* Mobile bottom nav */}
        <nav className="fixed inset-x-0 bottom-0 z-30 flex h-14 items-center justify-around border-t border-[var(--color-border)] bg-[var(--color-bg)]/95 backdrop-blur-md xl:hidden">
          {visibleMobileNavItems.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={buildDeskHref(href)}
                className={cn(
                  "flex flex-col items-center gap-0.5 px-2 py-1 text-[9px] font-medium transition-colors",
                  active
                    ? "text-[var(--color-accent)]"
                    : "text-[var(--color-text-muted)]",
                )}
              >
                <Icon className="size-4" strokeWidth={active ? 2 : 1.5} />
                {label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Inspector drawer */}
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
            activeAlerts={persistedActiveAlerts}
            recentAlerts={persistedRecentAlerts}
            audit={audit}
            jobsStatus={jobsStatus}
            liveState={liveState}
            liveAlerts={liveAlerts}
            transport={transport}
          />
        </div>
      </aside>

      {/* Dashboard config drawer */}
      <DashboardConfigPanel
        open={configOpen}
        onClose={() => setConfigOpen(false)}
        api={dashboardPrefs}
      />
    </div>
  );
}

/* Inspector content */

function InspectorContent({
  health,
  activeAlerts,
  recentAlerts,
  audit,
  jobsStatus,
  liveState,
  liveAlerts,
  transport,
}: {
  health: HealthResponse;
  activeAlerts: AlertSummary[];
  recentAlerts: AlertSummary[];
  audit: AuditEventResponse[];
  jobsStatus: WorkerStatusResponse | null;
  liveState: MT5LiveStateResponse | null;
  liveAlerts: NonNullable<MT5LiveStateResponse["operator_alerts"]>;
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
              value={`${job.state} | ${job.interval_seconds}s`}
            />
          ))}
        </InspectorSection>
      ) : null}

      {/* Live alerts */}
      {liveAlerts.length > 0 ? (
        <InspectorSection title={`Live Alerts (${liveAlerts.length})`}>
          {liveAlerts.slice(0, 5).map((alert) => (
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

      {/* Active alerts (API) */}
      {activeAlerts.length > 0 ? (
        <InspectorSection title={`Active Alerts (${activeAlerts.length})`}>
          {activeAlerts.slice(0, 5).map((alert) => (
            <div key={alert.id} className="border-b border-[var(--color-border)] pb-2 last:border-b-0 last:pb-0">
              <div className="flex items-start justify-between gap-2">
                <span className="text-[11px] font-medium text-[var(--color-text)]">{alert.code}</span>
                <span className="rounded-[2px] bg-[var(--color-accent-soft)] px-1 py-0.5 text-[9px] font-semibold uppercase text-[var(--color-accent)]">
                  ACTIVE
                </span>
                <AlertSeverityBadge severity={alert.severity} />
              </div>
              <p className="mt-0.5 text-[11px] text-[var(--color-text-muted)]">
                {alert.message}
              </p>
            </div>
          ))}
        </InspectorSection>
      ) : null}

      {/* Historical events (API) */}
      {recentAlerts.length > 0 ? (
        <InspectorSection title={`Recent Events (${recentAlerts.length})`}>
          {recentAlerts.slice(0, 5).map((alert) => (
            <div key={`history-${alert.id}`} className="border-b border-[var(--color-border)] pb-2 last:border-b-0 last:pb-0">
              <div className="flex items-start justify-between gap-2">
                <span className="text-[11px] font-medium text-[var(--color-text)]">{alert.code}</span>
                <span className="rounded-[2px] bg-[var(--color-surface-hover)] px-1 py-0.5 text-[9px] font-semibold uppercase text-[var(--color-text-muted)]">
                  HISTORY
                </span>
                <AlertSeverityBadge severity={alert.severity} />
              </div>
              <p className="mt-0.5 text-[11px] text-[var(--color-text-muted)]">
                {alert.message}
              </p>
              <p className="mt-0.5 text-[10px] text-[var(--color-text-muted)]">
                {formatTimestamp(alert.created_at)}
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

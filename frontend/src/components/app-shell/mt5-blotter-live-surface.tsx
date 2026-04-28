"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
import { DashboardActiveFilters } from "@/components/app-shell/dashboard-active-filters";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { LivePostureBanner } from "@/components/app-shell/live-posture-banner";
import { LiveRuntimeBadgeGroup } from "@/components/app-shell/live-runtime-badge-group";
import { PageHeader } from "@/components/app-shell/page-header";
import { ExecutionFillsTable, ExecutionHistoryTable, MT5TransactionHistoryTable, ReconciliationTable } from "@/components/data/risk-tables";
import { FieldInput, FieldLabel, FieldSelect, FieldTextarea, FormError, FormSection, SubmitButton } from "@/components/forms/shared";
import { MetricBlock } from "@/components/ui/metric-block";
import { ButtonLink, StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type {
  AuditEventResponse,
  ExecutionFillResponse,
  ExecutionResultResponse,
  MT5TransactionHistoryResponse,
  ReconciliationHistoryEntryResponse,
  ReconciliationSummaryResponse,
} from "@/lib/api/types";
import { buildIncidentTimeline } from "@/lib/incidents";
import { useRecentExecutionActivity } from "@/lib/use-recent-execution-activity";
import { formatCurrency, formatTimestamp, humanizeIdentifier } from "@/lib/utils";
import { countManualMt5Events } from "@/lib/view-models";
import { useDashboardPrefs } from "@/lib/dashboard-preferences-context";
import { symbolFilterTokens } from "@/lib/dashboard-preferences";

type IncidentStatus = "acknowledged" | "investigating" | "resolved";
type TransactionHistoryType = "all" | "order" | "deal" | "manual" | "desk";

type TransactionHistoryFilters = {
  dateFrom: string;
  dateTo: string;
  symbol: string;
  type: TransactionHistoryType;
};

function selectedMismatch(
  mismatches: ReconciliationSummaryResponse["mismatches"] | null | undefined,
  symbol: string | null,
) {
  if (!mismatches || !symbol) {
    return null;
  }
  return (
    (mismatches ?? []).find((item) => item.symbol === symbol) ?? null
  );
}

export function Mt5BlotterLiveSurface({
  portfolioSlug, initialExecutions, initialFills, initialAudit,
}: {
  portfolioSlug: string;
  initialExecutions: ExecutionResultResponse[];
  initialFills: ExecutionFillResponse[];
  initialAudit: AuditEventResponse[];
}) {
  const { liveState, transport, accountId } = useDeskLive();
  const { matchesSymbol, prefs } = useDashboardPrefs();
  const { executions, fills } = useRecentExecutionActivity({
    portfolioSlug,
    accountId,
    initialExecutions,
    initialFills,
    liveSequence: liveState?.sequence,
    executionLimit: 20,
    fillLimit: 20,
  });
  const filteredExecutions = useMemo(
    () => executions.filter((row) => matchesSymbol(row.symbol)),
    [executions, matchesSymbol],
  );
  const filteredFills = useMemo(
    () => fills.filter((row) => matchesSymbol(row.symbol)),
    [fills, matchesSymbol],
  );
  const orders = useMemo(
    () => (liveState?.order_history ?? []).filter((row) => matchesSymbol(row.symbol)),
    [liveState?.order_history, matchesSymbol],
  );
  const deals = useMemo(
    () => (liveState?.deal_history ?? []).filter((row) => matchesSymbol(row.symbol)),
    [liveState?.deal_history, matchesSymbol],
  );
  const manual = countManualMt5Events(orders, deals);

  const [reconciliationState, setReconciliationState] = useState<ReconciliationSummaryResponse | null>(
    null,
  );
  const [auditState, setAuditState] = useState<AuditEventResponse[]>(initialAudit);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(
    null,
  );
  const [incidentStatus, setIncidentStatus] = useState<IncidentStatus>("acknowledged");
  const [reason, setReason] = useState("operator_reviewed");
  const [operatorNote, setOperatorNote] = useState("");
  const [resolutionNote, setResolutionNote] = useState("");
  const [submitPending, setSubmitPending] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [historyDraftFilters, setHistoryDraftFilters] = useState<TransactionHistoryFilters>({
    dateFrom: "",
    dateTo: "",
    symbol: "",
    type: "all",
  });
  const [historyFilters, setHistoryFilters] = useState<TransactionHistoryFilters>({
    dateFrom: "",
    dateTo: "",
    symbol: "",
    type: "all",
  });
  const [historyPage, setHistoryPage] = useState(1);
  const [historyPageSize, setHistoryPageSize] = useState(50);
  const [historyState, setHistoryState] = useState<MT5TransactionHistoryResponse | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [reconciliationHistory, setReconciliationHistory] = useState<ReconciliationHistoryEntryResponse[]>([]);
  const [reconciliationError, setReconciliationError] = useState<string | null>(null);
  const reconciliationRequestSequence = useRef(0);
  const historyRequestSequence = useRef(0);
  const globalSymbolTokens = useMemo(
    () => symbolFilterTokens(prefs.symbolFilter),
    [prefs.symbolFilter],
  );
  const inferredGlobalHistorySymbol = globalSymbolTokens.length === 1 ? globalSymbolTokens[0] : "";

  useEffect(() => {
    setReconciliationState(liveState?.reconciliation ?? null);
  }, [liveState?.reconciliation, liveState?.sequence]);

  const reconciliation = reconciliationState;
  const mismatchRows = useMemo(
    () => (reconciliation?.mismatches ?? []).filter((row) => matchesSymbol(row.symbol)),
    [matchesSymbol, reconciliation],
  );
  const incidents = useMemo(
    () => (reconciliation?.incidents ?? []).filter((item) => matchesSymbol(item.symbol)),
    [matchesSymbol, reconciliation],
  );

  const activeSymbol = useMemo(() => {
    const active = mismatchRows.find((item) => item.status !== "match")?.symbol;
    return active ?? mismatchRows[0]?.symbol ?? null;
  }, [mismatchRows]);

  useEffect(() => {
    if (!selectedSymbol || !selectedMismatch(mismatchRows, selectedSymbol)) {
      setSelectedSymbol(activeSymbol);
    }
  }, [activeSymbol, mismatchRows, selectedSymbol]);

  const selectedRow = useMemo(
    () => selectedMismatch(mismatchRows, selectedSymbol),
    [mismatchRows, selectedSymbol],
  );

  useEffect(() => {
    if (!selectedRow) {
      setIncidentStatus("acknowledged");
      setReason("operator_reviewed");
      setOperatorNote("");
      setResolutionNote("");
      return;
    }
    setIncidentStatus((selectedRow.incident_status as IncidentStatus | null) ?? "acknowledged");
    setReason(selectedRow.incident_reason ?? selectedRow.acknowledged_reason ?? "operator_reviewed");
    setOperatorNote(selectedRow.incident_note ?? selectedRow.acknowledged_note ?? "");
    setResolutionNote(selectedRow.resolution_note ?? "");
  }, [selectedRow]);

  const mismatchCount = mismatchRows.filter((item) => item.status !== "match").length;
  const openIncidentCount =
    incidents.filter((item) => item.incident_status !== "resolved").length;
  const resolvedIncidentCount =
    incidents.filter((item) => item.incident_status === "resolved").length;
  const autoResolvedCount = reconciliation?.autoresolved_count ?? 0;
  const incidentTimeline = useMemo(
    () => buildIncidentTimeline(selectedSymbol, auditState, orders, deals),
    [auditState, deals, orders, selectedSymbol],
  );

  const refreshReconciliation = useCallback(async () => {
    const requestId = reconciliationRequestSequence.current + 1;
    reconciliationRequestSequence.current = requestId;
    setReconciliationError(null);
    try {
      const [summary, audit, history] = await Promise.all([
        api.reconciliationSummary(portfolioSlug),
        api.recentAudit(portfolioSlug, 120),
        api.reconciliationHistory({ portfolioSlug, limit: 20 }),
      ]);
      if (requestId !== reconciliationRequestSequence.current) {
        return;
      }
      setReconciliationState(summary);
      setAuditState(audit);
      setReconciliationHistory(history);
      setReconciliationError(null);
    } catch (error) {
      if (requestId !== reconciliationRequestSequence.current) {
        return;
      }
      setReconciliationError(
        error instanceof Error ? error.message : "Unable to refresh reconciliation diagnostics.",
      );
    }
  }, [portfolioSlug]);

  useEffect(() => {
    void refreshReconciliation();
  }, [refreshReconciliation]);

  const handleManage = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
  }, []);

  const handleApplyHistoryFilters = useCallback(() => {
    setHistoryFilters({
      ...historyDraftFilters,
      symbol: historyDraftFilters.symbol.trim().toUpperCase(),
    });
    setHistoryPage(1);
  }, [historyDraftFilters]);

  const handleResetHistoryFilters = useCallback(() => {
    const cleared: TransactionHistoryFilters = { dateFrom: "", dateTo: "", symbol: "", type: "all" };
    setHistoryDraftFilters(cleared);
    setHistoryFilters(cleared);
    setHistoryPage(1);
  }, []);
  const effectiveHistorySymbol = historyFilters.symbol || inferredGlobalHistorySymbol;

  useEffect(() => {
    let cancelled = false;
    const requestId = historyRequestSequence.current + 1;
    historyRequestSequence.current = requestId;

    async function refreshHistory() {
      setHistoryLoading(true);
      setHistoryError(null);
      try {
        const payload = await api.mt5TransactionHistory({
          portfolioSlug,
          accountId,
          dateFrom: historyFilters.dateFrom ? `${historyFilters.dateFrom}T00:00:00Z` : undefined,
          dateTo: historyFilters.dateTo ? `${historyFilters.dateTo}T23:59:59Z` : undefined,
          symbol: effectiveHistorySymbol || undefined,
          type: historyFilters.type,
          sort: "time_desc",
          page: historyPage,
          pageSize: historyPageSize,
        });
        if (cancelled || requestId !== historyRequestSequence.current) {
          return;
        }
        setHistoryState(payload);
        setHistoryError(null);
      } catch (error) {
        if (cancelled || requestId !== historyRequestSequence.current) {
          return;
        }
        setHistoryState(null);
        setHistoryError(error instanceof Error ? error.message : "Unable to load MT5 transaction history.");
      } finally {
        if (!cancelled && requestId === historyRequestSequence.current) {
          setHistoryLoading(false);
        }
      }
    }

    void refreshHistory();
    return () => {
      cancelled = true;
    };
  }, [accountId, effectiveHistorySymbol, historyFilters.dateFrom, historyFilters.dateTo, historyFilters.type, historyPage, historyPageSize, portfolioSlug]);

  const rawHistoryRows = historyState?.items ?? [];
  const historyRows = rawHistoryRows.filter((row) => matchesSymbol(row.symbol));
  const historyRowsFiltered = historyRows.length !== rawHistoryRows.length;
  const historyTotal = historyState?.total ?? 0;
  const historyPageCount = Math.max(1, Math.ceil(historyTotal / historyPageSize));
  const historyFrom = historyTotal === 0 ? 0 : (historyPage - 1) * historyPageSize + 1;
  const historyTo = historyTotal === 0 ? 0 : Math.min(historyPage * historyPageSize, historyTotal);
  const historyUsesGlobalSingleSymbol = historyFilters.symbol.length === 0 && inferredGlobalHistorySymbol.length > 0;
  const historyNeedsExplicitSymbol = historyFilters.symbol.length === 0 && globalSymbolTokens.length > 1;
  const showReconciliationError =
    Boolean(reconciliationError)
    && reconciliation == null
    && reconciliationHistory.length === 0
    && auditState.length === 0;
  const showHistoryError = Boolean(historyError) && historyRows.length === 0;
  const historyExportUrl = useMemo(
    () =>
      api.mt5TransactionHistoryExportUrl({
        portfolioSlug,
        accountId,
        dateFrom: historyFilters.dateFrom ? `${historyFilters.dateFrom}T00:00:00Z` : undefined,
        dateTo: historyFilters.dateTo ? `${historyFilters.dateTo}T23:59:59Z` : undefined,
        symbol: effectiveHistorySymbol || undefined,
        type: historyFilters.type,
        sort: "time_desc",
        maxRows: 5000,
      }),
    [accountId, effectiveHistorySymbol, historyFilters.dateFrom, historyFilters.dateTo, historyFilters.type, portfolioSlug],
  );

  const handleSaveIncident = useCallback(async () => {
    if (!selectedSymbol) {
      return;
    }
    setSubmitPending(true);
    setSubmitError(null);
    try {
      await api.updateReconciliationIncident({
        portfolio_slug: portfolioSlug,
        symbol: selectedSymbol,
        reason,
        operator_note: operatorNote,
        incident_status: incidentStatus,
        resolution_note: resolutionNote,
      });
      await refreshReconciliation();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Unable to update the incident.");
    } finally {
      setSubmitPending(false);
    }
  }, [incidentStatus, operatorNote, portfolioSlug, reason, refreshReconciliation, resolutionNote, selectedSymbol]);

  return (
    <div className="desk-page space-y-4">
      <PageHeader eyebrow="Blotter" title="MT5 orders, deals, fills and reconciliation"
        aside={<>
          <LiveRuntimeBadgeGroup liveState={liveState} transport={transport} />
          <ButtonLink
            href={
              accountId
                ? `/desk/incidents?portfolio=${encodeURIComponent(portfolioSlug)}&account=${encodeURIComponent(accountId)}`
                : `/desk/incidents?portfolio=${encodeURIComponent(portfolioSlug)}`
            }
            variant="secondary"
          >
            Incidents
          </ButtonLink>
        </>}
      />
      <DashboardActiveFilters showHorizon={false} showModel={false} />
      <LivePostureBanner liveState={liveState} transport={transport} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <MetricBlock label="Orders" value={String(orders.length)} hint={`${manual.orders} manual`} tone="accent" />
        <MetricBlock label="Deals" value={String(deals.length)} hint={`${manual.deals} manual`} tone="warning" />
        <MetricBlock label="Desk attempts" value={String(filteredExecutions.length)} tone="success" />
        <MetricBlock label="Fills" value={String(filteredFills.length)} hint={liveState?.generated_at ? formatTimestamp(liveState.generated_at) : undefined} />
        <MetricBlock label="Mismatches" value={String(mismatchCount)} tone={mismatchCount > 0 ? "warning" : "success"} />
        <MetricBlock label="Open incidents" value={String(openIncidentCount)} tone={openIncidentCount > 0 ? "warning" : "neutral"} />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />

      {reconciliation ? (
        <section className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-[var(--color-text-muted)]">
            <span>
              Live window: {reconciliation.live_window_minutes ?? 180}m
            </span>
            <span>
              Historical reconciliation: {reconciliation.heal_window_days ?? 30}d
            </span>
            <span>
              Evidence history window: {reconciliation.history_window_minutes ?? reconciliation.live_window_minutes ?? 180}m
            </span>
            {reconciliation.history_backfill_applied ? (
              <span>
                Historical backfill active
              </span>
            ) : null}
            <span>
              Active incidents: {openIncidentCount}
            </span>
            <span>
              Resolved incidents: {resolvedIncidentCount}
            </span>
            <span>
              Severity: {reconciliation.summary_severity ?? "ok"}
            </span>
            <span>
              Critical: {reconciliation.critical_mismatch_count ?? 0}
            </span>
            <span>
              Warning: {reconciliation.warning_mismatch_count ?? 0}
            </span>
            {autoResolvedCount > 0 ? (
              <span>
                Auto-resolved this cycle: {autoResolvedCount}
              </span>
            ) : null}
          </div>
        </section>
      ) : null}

      <div className="space-y-2">
        {showReconciliationError ? (
          <FormError message={reconciliationError} />
        ) : null}
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
              Reconciliation history
            </h4>
            <StatusBadge
              label={reconciliation?.summary_severity ?? "unknown"}
              tone={
                (reconciliation?.summary_severity ?? "ok") === "critical"
                  ? "danger"
                  : (reconciliation?.summary_severity ?? "ok") === "warn"
                    ? "warning"
                    : "success"
              }
            />
          </div>
          <div className="rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 p-3 shadow-[var(--shadow-soft)]">
            {reconciliationHistory.length > 0 ? (
              <div className="space-y-2">
                {reconciliationHistory.slice(0, 6).map((entry) => (
                  <div key={`${entry.id ?? "history"}:${entry.created_at ?? "ts"}`} className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <StatusBadge
                        label={entry.summary_severity}
                        tone={entry.summary_severity === "critical" ? "danger" : entry.summary_severity === "warn" ? "warning" : "success"}
                      />
                      <span className="text-[12px] font-medium text-[var(--color-text)]">
                        {entry.critical_mismatch_count ?? 0} critical | {entry.warning_mismatch_count ?? 0} warning
                      </span>
                    </div>
                    <div className="mt-1 text-[11px] text-[var(--color-text-muted)]">
                      {formatTimestamp(entry.created_at)} | unmatched executions: {entry.unmatched_execution_count ?? 0}
                    </div>
                    {entry.top_symbols?.length ? (
                      <div className="mt-1 text-[11px] text-[var(--color-text-soft)]">
                        Top symbols: {entry.top_symbols.join(", ")}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-[12px] text-[var(--color-text-soft)]">
                Reconciliation history will appear after the first divergence snapshots are recorded.
              </p>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
            MT5 transaction history (read-only)
          </h4>
          <ButtonLink href={historyExportUrl} variant="secondary">Export CSV</ButtonLink>
        </div>
        <div className="rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 p-3 shadow-[var(--shadow-soft)]">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <div>
              <FieldLabel htmlFor="history-date-from">Date from</FieldLabel>
              <FieldInput
                id="history-date-from"
                type="date"
                value={historyDraftFilters.dateFrom}
                onChange={(event) => setHistoryDraftFilters((current) => ({ ...current, dateFrom: event.target.value }))}
              />
            </div>
            <div>
              <FieldLabel htmlFor="history-date-to">Date to</FieldLabel>
              <FieldInput
                id="history-date-to"
                type="date"
                value={historyDraftFilters.dateTo}
                onChange={(event) => setHistoryDraftFilters((current) => ({ ...current, dateTo: event.target.value }))}
              />
            </div>
            <div>
              <FieldLabel htmlFor="history-symbol">Symbol</FieldLabel>
              <FieldInput
                id="history-symbol"
                placeholder="EURUSD"
                value={historyDraftFilters.symbol}
                onChange={(event) => setHistoryDraftFilters((current) => ({ ...current, symbol: event.target.value }))}
              />
            </div>
            <div>
              <FieldLabel htmlFor="history-type">Type</FieldLabel>
              <FieldSelect
                id="history-type"
                value={historyDraftFilters.type}
                onChange={(event) => setHistoryDraftFilters((current) => ({ ...current, type: event.target.value as TransactionHistoryType }))}
              >
                <option value="all">All</option>
                <option value="order">Order</option>
                <option value="deal">Deal</option>
                <option value="manual">Manual</option>
                <option value="desk">Desk</option>
              </FieldSelect>
            </div>
            <div>
              <FieldLabel htmlFor="history-page-size">Rows per page</FieldLabel>
              <FieldSelect
                id="history-page-size"
                value={String(historyPageSize)}
                onChange={(event) => {
                  setHistoryPageSize(Number(event.target.value));
                  setHistoryPage(1);
                }}
              >
                <option value="25">25</option>
                <option value="50">50</option>
                <option value="100">100</option>
              </FieldSelect>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="rounded-[var(--radius-md)] border border-[var(--color-border)] px-3 py-1 text-[12px] font-semibold text-[var(--color-text)] transition hover:bg-[var(--color-surface-hover)]"
                onClick={handleApplyHistoryFilters}
              >
                Apply filters
              </button>
              <button
                type="button"
                className="rounded-[var(--radius-md)] border border-[var(--color-border)] px-3 py-1 text-[12px] font-semibold text-[var(--color-text-muted)] transition hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]"
                onClick={handleResetHistoryFilters}
              >
                Reset
              </button>
            </div>
            <span className="text-[11px] text-[var(--color-text-muted)]">
              {historyLoading
                ? "Loading..."
                : `Showing ${historyFrom}-${historyTo} of ${historyTotal}${historyRowsFiltered ? ` (filtered view ${historyRows.length})` : ""}`}
            </span>
          </div>
          {historyUsesGlobalSingleSymbol ? (
            <p className="mt-2 text-[11px] text-[var(--color-text-muted)]">
              Using global symbol filter: {inferredGlobalHistorySymbol}.
            </p>
          ) : null}
          {historyNeedsExplicitSymbol ? (
            <p className="mt-2 text-[11px] text-[var(--color-text-muted)]">
              Multiple global symbols are active. Set a specific symbol to export a scoped CSV.
            </p>
          ) : null}

          <div className="mt-3">
            {showHistoryError ? (
              <FormError message={historyError} />
            ) : historyRows.length > 0 ? (
              <MT5TransactionHistoryTable rows={historyRows} />
            ) : (
              <p className="rounded-[var(--radius-md)] border border-dashed border-[var(--color-border)] px-3 py-2 text-[12px] text-[var(--color-text-soft)]">
                No MT5 transactions matched the selected filters.
              </p>
            )}
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--color-text-muted)]">
            <button
              type="button"
              className="rounded-[var(--radius-sm)] border border-[var(--color-border)] px-2 py-1 transition enabled:hover:bg-[var(--color-surface-hover)] disabled:cursor-not-allowed disabled:opacity-60"
              onClick={() => setHistoryPage((current) => Math.max(current - 1, 1))}
              disabled={historyPage <= 1 || historyLoading}
            >
              Previous
            </button>
            <span>
              Page {historyPage} / {historyPageCount}
            </span>
            <button
              type="button"
              className="rounded-[var(--radius-sm)] border border-[var(--color-border)] px-2 py-1 transition enabled:hover:bg-[var(--color-surface-hover)] disabled:cursor-not-allowed disabled:opacity-60"
              onClick={() => setHistoryPage((current) => current + 1)}
              disabled={historyLoading || !historyState?.has_next}
            >
              Next
            </button>
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.35fr,0.9fr]">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Execution attempts</h4>
          <ExecutionHistoryTable rows={filteredExecutions} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Incident workflow</h4>
          <div className="rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 p-4 shadow-[var(--shadow-soft)]">
            {selectedRow ? (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <StatusBadge label={selectedRow.symbol} tone="accent" />
                    <StatusBadge label={selectedRow.status} tone={selectedRow.status === "match" ? "success" : "warning"} />
                    <StatusBadge label={selectedRow.incident_status ?? "new"} tone={selectedRow.incident_status ? "accent" : "neutral"} />
                  </div>
                  <p className="text-[12px] text-[var(--color-text-soft)]">{selectedRow.reason}</p>
                  <div className="pt-1">
                    <ButtonLink
                      href={
                        accountId
                          ? `/desk/execution?portfolio=${encodeURIComponent(portfolioSlug)}&account=${encodeURIComponent(accountId)}&symbol=${encodeURIComponent(selectedRow.symbol)}&action=close`
                          : `/desk/execution?portfolio=${encodeURIComponent(portfolioSlug)}&symbol=${encodeURIComponent(selectedRow.symbol)}&action=close`
                      }
                      variant="secondary"
                    >
                      Close now on {selectedRow.symbol}
                    </ButtonLink>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <MetricBlock label="Desk exposure" value={formatCurrency(selectedRow.desk_exposure_eur)} />
                  <MetricBlock label="Live exposure" value={formatCurrency(selectedRow.live_exposure_eur)} tone="accent" />
                  <MetricBlock label="Drift" value={formatCurrency(selectedRow.difference_eur)} tone={Math.abs(selectedRow.difference_eur) > 1e-6 ? "warning" : "success"} />
                  <MetricBlock label="Volumes" value={`${selectedRow.desk_volume_lots ?? 0} / ${selectedRow.live_volume_lots ?? 0}`} hint="desk / broker lots" />
                </div>

                <FormSection title="Incident control">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <FieldLabel htmlFor="incident-status">Incident status</FieldLabel>
                      <FieldSelect id="incident-status" value={incidentStatus} onChange={(event) => setIncidentStatus(event.target.value as IncidentStatus)}>
                        <option value="acknowledged">Acknowledged</option>
                        <option value="investigating">Investigating</option>
                        <option value="resolved">Resolved</option>
                      </FieldSelect>
                    </div>
                    <div>
                      <FieldLabel htmlFor="incident-reason">Operator reason</FieldLabel>
                      <FieldSelect id="incident-reason" value={reason} onChange={(event) => setReason(event.target.value)}>
                        <option value="operator_reviewed">Operator reviewed</option>
                        <option value="manual_trade_expected">Manual trade expected</option>
                        <option value="broker_fill_in_progress">Broker fill in progress</option>
                        <option value="accepted_drift">Accepted drift</option>
                        <option value="resolved_after_reconciliation">Resolved after reconciliation</option>
                      </FieldSelect>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <FieldLabel htmlFor="incident-note">Operator note</FieldLabel>
                      <FieldTextarea id="incident-note" value={operatorNote} onChange={(event) => setOperatorNote(event.target.value)} placeholder="What happened, what you saw, what the broker showed..." />
                    </div>
                    <div>
                      <FieldLabel htmlFor="incident-resolution">Resolution note</FieldLabel>
                      <FieldTextarea id="incident-resolution" value={resolutionNote} onChange={(event) => setResolutionNote(event.target.value)} placeholder="How the incident was resolved, accepted or closed." />
                    </div>
                  </div>
                </FormSection>

                <div className="flex flex-wrap items-center gap-3 text-[11px] text-[var(--color-text-muted)]">
                  <span>Acknowledged: {formatTimestamp(selectedRow.acknowledged_at)}</span>
                  <span>Updated: {formatTimestamp(selectedRow.incident_updated_at)}</span>
                  {selectedRow.resolved_at ? <span>Resolved: {formatTimestamp(selectedRow.resolved_at)}</span> : null}
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3">
                  <FormError message={submitError} />
                  <SubmitButton
                    type="button"
                    isPending={submitPending}
                    label="Save incident"
                    pendingLabel="Saving..."
                    onClick={handleSaveIncident}
                  />
                </div>

                <FormSection title="Symbol history">
                  {incidentTimeline.length > 0 ? (
                    <div className="space-y-2">
                      {incidentTimeline.slice(0, 6).map((entry) => (
                        <div key={entry.id} className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <StatusBadge label={entry.kind} tone={entry.tone} />
                            <span className="text-[12px] font-medium text-[var(--color-text)]">
                              {humanizeIdentifier(entry.title)}
                            </span>
                          </div>
                          <p className="mt-1 text-[12px] text-[var(--color-text-soft)]">{entry.detail}</p>
                          <div className="mt-1 text-[11px] text-[var(--color-text-muted)]">{formatTimestamp(entry.time)}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[12px] text-[var(--color-text-soft)]">
                      No incident history was recorded yet for this symbol.
                    </p>
                  )}
                </FormSection>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-[13px] text-[var(--color-text-soft)]">No active mismatch is selected. Use the reconciliation table to manage a broker incident.</p>
                {incidents.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {incidents.slice(0, 6).map((incident) => (
                      <button
                        key={`${incident.symbol}-${incident.updated_at ?? incident.acknowledged_at ?? "incident"}`}
                        type="button"
                        className="rounded-[var(--radius-sm)] border border-[var(--color-border)] px-2 py-1 text-[10px] font-medium text-[var(--color-text)] transition hover:bg-[var(--color-surface-hover)]"
                        onClick={() => setSelectedSymbol(incident.symbol)}
                      >
                        {incident.symbol} | {incident.incident_status ?? "acknowledged"}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Reconciliation</h4>
          <ReconciliationTable rows={mismatchRows} onManage={handleManage} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Execution fills</h4>
          <ExecutionFillsTable rows={filteredFills} />
        </div>
      </div>
    </div>
  );
}

"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useDeskLive } from "@/components/app-shell/desk-live-provider";
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
  ReconciliationSummaryResponse,
} from "@/lib/api/types";
import { buildIncidentTimeline } from "@/lib/incidents";
import { useRecentExecutionActivity } from "@/lib/use-recent-execution-activity";
import { formatCurrency, formatTimestamp, humanizeIdentifier } from "@/lib/utils";
import { countManualMt5Events } from "@/lib/view-models";

type IncidentStatus = "acknowledged" | "investigating" | "resolved";
type TransactionHistoryType = "all" | "order" | "deal" | "manual" | "desk";

type TransactionHistoryFilters = {
  dateFrom: string;
  dateTo: string;
  symbol: string;
  type: TransactionHistoryType;
};

function selectedMismatch(
  reconciliation: ReconciliationSummaryResponse | null,
  symbol: string | null,
) {
  if (!reconciliation || !symbol) {
    return null;
  }
  return (
    (reconciliation.mismatches ?? []).find((item) => item.symbol === symbol) ?? null
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
  const { executions, fills } = useRecentExecutionActivity({
    portfolioSlug,
    accountId,
    initialExecutions,
    initialFills,
    liveSequence: liveState?.sequence,
    executionLimit: 20,
    fillLimit: 20,
  });
  const orders = useMemo(() => liveState?.order_history ?? [], [liveState?.order_history]);
  const deals = useMemo(() => liveState?.deal_history ?? [], [liveState?.deal_history]);
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

  useEffect(() => {
    setReconciliationState(liveState?.reconciliation ?? null);
  }, [liveState?.reconciliation, liveState?.sequence]);

  const activeSymbol = useMemo(() => {
    const mismatches = reconciliationState?.mismatches ?? [];
    const active = mismatches.find((item) => item.status !== "match")?.symbol;
    return active ?? mismatches[0]?.symbol ?? null;
  }, [reconciliationState]);

  useEffect(() => {
    if (!selectedSymbol || !selectedMismatch(reconciliationState, selectedSymbol)) {
      setSelectedSymbol(activeSymbol);
    }
  }, [activeSymbol, reconciliationState, selectedSymbol]);

  const selectedRow = useMemo(
    () => selectedMismatch(reconciliationState, selectedSymbol),
    [reconciliationState, selectedSymbol],
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

  const reconciliation = reconciliationState;
  const mismatchCount = (reconciliation?.mismatches ?? []).filter((i) => i.status !== "match").length;
  const incidents = reconciliation?.incidents ?? [];
  const openIncidentCount =
    reconciliation?.active_incident_count
    ?? incidents.filter((item) => item.incident_status !== "resolved").length;
  const resolvedIncidentCount =
    reconciliation?.resolved_incident_count
    ?? incidents.filter((item) => item.incident_status === "resolved").length;
  const autoResolvedCount = reconciliation?.autoresolved_count ?? 0;
  const incidentTimeline = useMemo(
    () => buildIncidentTimeline(selectedSymbol, auditState, orders, deals),
    [auditState, deals, orders, selectedSymbol],
  );

  const refreshReconciliation = useCallback(async () => {
    const [summary, audit] = await Promise.all([
      api.reconciliationSummary(portfolioSlug),
      api.recentAudit(portfolioSlug, 120),
    ]);
    setReconciliationState(summary);
    setAuditState(audit);
  }, [portfolioSlug]);

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

  useEffect(() => {
    let cancelled = false;

    async function refreshHistory() {
      setHistoryLoading(true);
      setHistoryError(null);
      try {
        const payload = await api.mt5TransactionHistory({
          portfolioSlug,
          accountId,
          dateFrom: historyFilters.dateFrom ? `${historyFilters.dateFrom}T00:00:00Z` : undefined,
          dateTo: historyFilters.dateTo ? `${historyFilters.dateTo}T23:59:59Z` : undefined,
          symbol: historyFilters.symbol || undefined,
          type: historyFilters.type,
          sort: "time_desc",
          page: historyPage,
          pageSize: historyPageSize,
        });
        if (cancelled) {
          return;
        }
        setHistoryState(payload);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setHistoryState(null);
        setHistoryError(error instanceof Error ? error.message : "Unable to load MT5 transaction history.");
      } finally {
        if (!cancelled) {
          setHistoryLoading(false);
        }
      }
    }

    void refreshHistory();
    return () => {
      cancelled = true;
    };
  }, [accountId, historyFilters.dateFrom, historyFilters.dateTo, historyFilters.symbol, historyFilters.type, historyPage, historyPageSize, portfolioSlug]);

  const historyRows = historyState?.items ?? [];
  const historyTotal = historyState?.total ?? 0;
  const historyPageCount = Math.max(1, Math.ceil(historyTotal / historyPageSize));
  const historyFrom = historyTotal === 0 ? 0 : (historyPage - 1) * historyPageSize + 1;
  const historyTo = historyTotal === 0 ? 0 : Math.min(historyPage * historyPageSize, historyTotal);
  const historyExportUrl = useMemo(
    () =>
      api.mt5TransactionHistoryExportUrl({
        portfolioSlug,
        accountId,
        dateFrom: historyFilters.dateFrom ? `${historyFilters.dateFrom}T00:00:00Z` : undefined,
        dateTo: historyFilters.dateTo ? `${historyFilters.dateTo}T23:59:59Z` : undefined,
        symbol: historyFilters.symbol || undefined,
        type: historyFilters.type,
        sort: "time_desc",
        maxRows: 5000,
      }),
    [accountId, historyFilters.dateFrom, historyFilters.dateTo, historyFilters.symbol, historyFilters.type, portfolioSlug],
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
      <LivePostureBanner liveState={liveState} transport={transport} />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <MetricBlock label="Orders" value={String(orders.length)} hint={`${manual.orders} manual`} tone="accent" />
        <MetricBlock label="Deals" value={String(deals.length)} hint={`${manual.deals} manual`} tone="warning" />
        <MetricBlock label="Desk attempts" value={String(executions.length)} tone="success" />
        <MetricBlock label="Fills" value={String(fills.length)} hint={liveState?.generated_at ? formatTimestamp(liveState.generated_at) : undefined} />
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
            {autoResolvedCount > 0 ? (
              <span>
                Auto-resolved this cycle: {autoResolvedCount}
              </span>
            ) : null}
          </div>
        </section>
      ) : null}

      <div className="space-y-2">
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
              {historyLoading ? "Loading..." : `Showing ${historyFrom}-${historyTo} of ${historyTotal}`}
            </span>
          </div>

          <div className="mt-3">
            {historyError ? (
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
          <ExecutionHistoryTable rows={executions} />
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
          <ReconciliationTable rows={reconciliation?.mismatches ?? []} onManage={handleManage} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Execution fills</h4>
          <ExecutionFillsTable rows={fills} />
        </div>
      </div>
    </div>
  );
}

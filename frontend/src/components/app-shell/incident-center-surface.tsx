"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import {
  FieldInput,
  FieldLabel,
  FieldSelect,
  FieldTextarea,
  FormError,
  FormSection,
  SubmitButton,
} from "@/components/forms/shared";
import { MetricBlock } from "@/components/ui/metric-block";
import { ButtonLink, StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type {
  AuditEventResponse,
  MT5LiveStateResponse,
  ReconciliationAcknowledgementResponse,
  ReconciliationSummaryResponse,
} from "@/lib/api/types";
import {
  buildIncidentTimeline,
  buildIncidentWorkbenchRows,
  incidentMatchesQuery,
  incidentMatchesWindow,
  type IncidentWorkbenchRow,
} from "@/lib/incidents";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatTimestamp, humanizeIdentifier } from "@/lib/utils";

type IncidentStatus = "acknowledged" | "investigating" | "resolved";
type IncidentFilter = "all" | "open" | "new" | IncidentStatus | "critical";
type IncidentWindow = "all" | "24h" | "7d";

function mismatchTone(status: string | null | undefined) {
  const normalized = String(status || "").toLowerCase();
  if (
    normalized.includes("reject")
    || normalized.includes("drift")
    || normalized.includes("orphan_live_position")
    || normalized.includes("overfill")
  ) {
    return "danger" as const;
  }
  if (normalized.includes("partial") || normalized.includes("pending") || normalized.includes("manual") || normalized.includes("investigating")) {
    return "warning" as const;
  }
  if (normalized.includes("match") || normalized.includes("resolved")) {
    return "success" as const;
  }
  if (normalized.includes("acknowledged")) {
    return "accent" as const;
  }
  return "neutral" as const;
}

function matchesStatusFilter(row: IncidentWorkbenchRow, filter: IncidentFilter): boolean {
  if (filter === "all") {
    return true;
  }
  if (filter === "open") {
    return row.requires_action;
  }
  if (filter === "new") {
    return row.active && !row.incident_status;
  }
  if (filter === "critical") {
    return ["desk_vs_broker_drift", "rejected_by_broker", "orphan_live_position", "overfill_or_volume_drift"].includes(
      row.mismatch_status,
    );
  }
  return row.incident_status === filter;
}

function alertPriorityCode(code: string | null | undefined): number {
  const normalized = String(code || "").toUpperCase();
  if (normalized.includes("BROKER_REJECTION")) return 0;
  if (normalized.includes("PARTIAL_FILL")) return 1;
  if (normalized.includes("MANUAL_TRADE") || normalized.includes("MANUAL_EVENTS")) return 2;
  if (normalized.includes("DRIFT") || normalized.includes("ORPHAN")) return 3;
  return 4;
}

export function IncidentCenterSurface({
  portfolioSlug,
  initialLiveState,
  initialIncidents,
  initialAudit,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialIncidents: ReconciliationAcknowledgementResponse[];
  initialAudit: AuditEventResponse[];
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [reconciliationState, setReconciliationState] = useState<ReconciliationSummaryResponse | null>(
    initialLiveState?.reconciliation ?? null,
  );
  const [auditState, setAuditState] = useState<AuditEventResponse[]>(initialAudit);
  const [incidentFallback, setIncidentFallback] = useState<ReconciliationAcknowledgementResponse[]>(initialIncidents);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<IncidentFilter>("open");
  const [windowFilter, setWindowFilter] = useState<IncidentWindow>("7d");
  const [query, setQuery] = useState("");
  const [incidentStatus, setIncidentStatus] = useState<IncidentStatus>("acknowledged");
  const [reason, setReason] = useState("operator_reviewed");
  const [operatorNote, setOperatorNote] = useState("");
  const [resolutionNote, setResolutionNote] = useState("");
  const [submitPending, setSubmitPending] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    setReconciliationState(liveState?.reconciliation ?? null);
  }, [liveState?.reconciliation, liveState?.sequence]);

  const incidentRows = useMemo(
    () => buildIncidentWorkbenchRows(reconciliationState, incidentFallback),
    [incidentFallback, reconciliationState],
  );

  const filteredRows = useMemo(
    () =>
      incidentRows.filter(
        (row) =>
          incidentMatchesQuery(row, query)
          && incidentMatchesWindow(row, windowFilter)
          && matchesStatusFilter(row, statusFilter),
      ),
    [incidentRows, query, statusFilter, windowFilter],
  );

  useEffect(() => {
    const active = filteredRows.find((row) => row.requires_action)?.symbol ?? filteredRows[0]?.symbol ?? null;
    if (!selectedSymbol || !filteredRows.some((row) => row.symbol === selectedSymbol)) {
      setSelectedSymbol(active);
    }
  }, [filteredRows, selectedSymbol]);

  const selectedRow = useMemo(
    () => filteredRows.find((row) => row.symbol === selectedSymbol) ?? incidentRows.find((row) => row.symbol === selectedSymbol) ?? null,
    [filteredRows, incidentRows, selectedSymbol],
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
    setReason(selectedRow.reason || "operator_reviewed");
    setOperatorNote(selectedRow.operator_note || "");
    setResolutionNote(selectedRow.resolution_note || "");
  }, [selectedRow]);

  const timeline = useMemo(
    () =>
      buildIncidentTimeline(
        selectedSymbol,
        auditState,
        liveState?.order_history ?? [],
        liveState?.deal_history ?? [],
      ),
    [auditState, liveState?.deal_history, liveState?.order_history, selectedSymbol],
  );

  const refreshIncidentData = useCallback(async () => {
    const [summary, incidents, audit] = await Promise.all([
      api.reconciliationSummary(portfolioSlug),
      api.reconciliationIncidents({ portfolioSlug, includeResolved: true, limit: 200 }),
      api.recentAudit(portfolioSlug, 150),
    ]);
    setReconciliationState(summary);
    setIncidentFallback(incidents);
    setAuditState(audit);
  }, [portfolioSlug]);

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
      await refreshIncidentData();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Unable to update the incident.");
    } finally {
      setSubmitPending(false);
    }
  }, [incidentStatus, operatorNote, portfolioSlug, reason, refreshIncidentData, resolutionNote, selectedSymbol]);

  const openCount = incidentRows.filter((row) => row.requires_action).length;
  const investigatingCount = incidentRows.filter((row) => row.incident_status === "investigating").length;
  const resolvedCount = incidentRows.filter((row) => row.incident_status === "resolved").length;
  const criticalCount = incidentRows.filter((row) => matchesStatusFilter(row, "critical")).length;
  const alertHighlights = [...(liveState?.operator_alerts ?? [])].sort(
    (left, right) => alertPriorityCode(left.code) - alertPriorityCode(right.code),
  );

  return (
    <div className="desk-page space-y-4">
      <PageHeader
        eyebrow="Incident Center"
        title="Broker incidents, drift follow-up and operator history"
        aside={(
          <>
            <StatusBadge label={liveState?.status ?? "unknown"} tone={liveState?.status === "ok" ? "success" : "warning"} />
            <StatusBadge label={transport} tone={transport === "stream" ? "success" : "warning"} />
            <ButtonLink href={`/desk/blotter?portfolio=${portfolioSlug}`} variant="secondary">
              Open blotter
            </ButtonLink>
          </>
        )}
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <MetricBlock label="Open incidents" value={String(openCount)} tone={openCount > 0 ? "warning" : "success"} />
        <MetricBlock label="Investigating" value={String(investigatingCount)} tone={investigatingCount > 0 ? "accent" : "neutral"} />
        <MetricBlock label="Critical drift" value={String(criticalCount)} tone={criticalCount > 0 ? "danger" : "success"} />
        <MetricBlock label="Resolved" value={String(resolvedCount)} />
        <MetricBlock label="Manual alerts" value={String(alertHighlights.filter((item) => /MANUAL/i.test(item.code)).length)} tone={alertHighlights.some((item) => /MANUAL/i.test(item.code)) ? "warning" : "neutral"} />
        <MetricBlock label="Broker rejects" value={String(alertHighlights.filter((item) => /BROKER_REJECTION/i.test(item.code)).length)} tone={alertHighlights.some((item) => /BROKER_REJECTION/i.test(item.code)) ? "danger" : "success"} />
      </section>

      <LiveOperatorAlerts alerts={alertHighlights.slice(0, 6)} title="Priority operator alerts" />

      <div className="grid gap-4 xl:grid-cols-[0.92fr,1.2fr]">
        <div className="space-y-3">
          <div className="rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 p-4 shadow-[var(--shadow-soft)]">
            <FormSection title="Filters">
              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <FieldLabel htmlFor="incident-search">Search symbol or note</FieldLabel>
                  <FieldInput
                    id="incident-search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="USDJPY, manual trade, broker fill..."
                  />
                </div>
                <div>
                  <FieldLabel htmlFor="incident-filter-status">Status filter</FieldLabel>
                  <FieldSelect
                    id="incident-filter-status"
                    value={statusFilter}
                    onChange={(event) => setStatusFilter(event.target.value as IncidentFilter)}
                  >
                    <option value="all">All incidents</option>
                    <option value="open">Open only</option>
                    <option value="new">New mismatches</option>
                    <option value="acknowledged">Acknowledged</option>
                    <option value="investigating">Investigating</option>
                    <option value="resolved">Resolved</option>
                    <option value="critical">Critical drift</option>
                  </FieldSelect>
                </div>
              </div>
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <div>
                  <FieldLabel htmlFor="incident-filter-window">Time window</FieldLabel>
                  <FieldSelect
                    id="incident-filter-window"
                    value={windowFilter}
                    onChange={(event) => setWindowFilter(event.target.value as IncidentWindow)}
                  >
                    <option value="all">All history</option>
                    <option value="24h">Last 24h</option>
                    <option value="7d">Last 7 days</option>
                  </FieldSelect>
                </div>
                <div className="flex items-end">
                  <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-[12px] text-[var(--color-text-muted)]">
                    {filteredRows.length} row(s) visible
                  </div>
                </div>
              </div>
            </FormSection>
          </div>

          <div className="space-y-2">
            {filteredRows.length === 0 ? (
              <div className="rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 p-4 text-[13px] text-[var(--color-text-soft)] shadow-[var(--shadow-soft)]">
                No incidents match the current filters.
              </div>
            ) : (
              filteredRows.map((row) => (
                <button
                  key={`${row.symbol}:${row.incident_id ?? row.updated_at ?? row.acknowledged_at ?? "incident"}`}
                  type="button"
                  onClick={() => setSelectedSymbol(row.symbol)}
                  className={`w-full rounded-[var(--radius-xl)] border px-4 py-3 text-left shadow-[var(--shadow-soft)] transition ${
                    selectedSymbol === row.symbol
                      ? "border-[var(--color-accent)]/40 bg-[var(--color-accent-soft)]/30"
                      : "border-[var(--color-border)] bg-[var(--color-surface)]/95 hover:bg-[var(--color-surface-hover)]"
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-[var(--color-text)]">{row.symbol}</span>
                    <StatusBadge label={row.mismatch_status} tone={mismatchTone(row.mismatch_status)} />
                    <StatusBadge label={row.incident_status ?? "new"} tone={mismatchTone(row.incident_status ?? row.mismatch_status)} />
                  </div>
                  <p className="mt-1 text-[12px] leading-relaxed text-[var(--color-text-soft)]">
                    {row.reason || "No operator note recorded yet."}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-[var(--color-text-muted)]">
                    <span>Drift: {formatCurrency(row.difference_eur)}</span>
                    <span>Desk: {formatCurrency(row.desk_exposure_eur)}</span>
                    <span>Live: {formatCurrency(row.live_exposure_eur)}</span>
                    <span>Updated: {formatTimestamp(row.updated_at || row.acknowledged_at)}</span>
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 p-4 shadow-[var(--shadow-soft)]">
            {selectedRow ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge label={selectedRow.symbol} tone="accent" />
                  <StatusBadge label={selectedRow.mismatch_status} tone={mismatchTone(selectedRow.mismatch_status)} />
                  <StatusBadge label={selectedRow.incident_status ?? "new"} tone={mismatchTone(selectedRow.incident_status ?? selectedRow.mismatch_status)} />
                </div>

                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <MetricBlock label="Desk exposure" value={formatCurrency(selectedRow.desk_exposure_eur)} />
                  <MetricBlock label="Live exposure" value={formatCurrency(selectedRow.live_exposure_eur)} tone="accent" />
                  <MetricBlock label="Drift" value={formatCurrency(selectedRow.difference_eur)} tone={Math.abs(selectedRow.difference_eur) > 1e-6 ? "warning" : "success"} />
                  <MetricBlock label="Lots" value={`${selectedRow.desk_volume_lots ?? 0} / ${selectedRow.live_volume_lots ?? 0}`} hint="desk / broker" />
                </div>

                <div className="flex flex-wrap gap-2">
                  <ButtonLink href={`/desk/blotter?portfolio=${portfolioSlug}`} variant="secondary">
                    Open blotter context
                  </ButtonLink>
                  <ButtonLink
                    href={`/desk/execution?portfolio=${portfolioSlug}&symbol=${selectedRow.symbol}`}
                    variant="secondary"
                  >
                    Open execution for {selectedRow.symbol}
                  </ButtonLink>
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
                      <FieldTextarea id="incident-note" value={operatorNote} onChange={(event) => setOperatorNote(event.target.value)} placeholder="What happened, what the broker showed, what needs follow-up..." />
                    </div>
                    <div>
                      <FieldLabel htmlFor="incident-resolution">Resolution note</FieldLabel>
                      <FieldTextarea id="incident-resolution" value={resolutionNote} onChange={(event) => setResolutionNote(event.target.value)} placeholder="How the incident was resolved or why the live state was accepted." />
                    </div>
                  </div>
                </FormSection>

                <div className="flex flex-wrap gap-3 text-[11px] text-[var(--color-text-muted)]">
                  <span>Acknowledged: {formatTimestamp(selectedRow.acknowledged_at)}</span>
                  <span>Updated: {formatTimestamp(selectedRow.updated_at)}</span>
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
              </div>
            ) : (
              <p className="text-[13px] text-[var(--color-text-soft)]">
                Select an incident on the left to review the drift, operator notes and broker timeline.
              </p>
            )}
          </div>

          <div className="rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 p-4 shadow-[var(--shadow-soft)]">
            <div className="mb-3 flex items-center justify-between">
              <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
                Symbol history
              </h4>
              {selectedSymbol ? <StatusBadge label={selectedSymbol} tone="accent" /> : null}
            </div>
            {timeline.length > 0 ? (
              <div className="space-y-2">
                {timeline.slice(0, 10).map((entry) => (
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
              <p className="text-[13px] text-[var(--color-text-soft)]">
                No incident timeline yet for this symbol.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

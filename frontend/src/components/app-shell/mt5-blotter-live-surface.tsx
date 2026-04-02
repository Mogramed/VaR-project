"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { DealHistoryTable, ExecutionFillsTable, ExecutionHistoryTable, OrderHistoryTable, ReconciliationTable } from "@/components/data/risk-tables";
import { FieldLabel, FieldSelect, FieldTextarea, FormError, FormSection, SubmitButton } from "@/components/forms/shared";
import { MetricBlock } from "@/components/ui/metric-block";
import { ButtonLink, StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type {
  AuditEventResponse,
  ExecutionFillResponse,
  ExecutionResultResponse,
  MT5LiveStateResponse,
  ReconciliationSummaryResponse,
} from "@/lib/api/types";
import { buildIncidentTimeline } from "@/lib/incidents";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { useRecentExecutionActivity } from "@/lib/use-recent-execution-activity";
import { formatCurrency, formatTimestamp, humanizeIdentifier } from "@/lib/utils";
import { countManualMt5Events } from "@/lib/view-models";

type IncidentStatus = "acknowledged" | "investigating" | "resolved";

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
  portfolioSlug, initialLiveState, initialExecutions, initialFills, initialAudit,
}: {
  portfolioSlug: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialExecutions: ExecutionResultResponse[];
  initialFills: ExecutionFillResponse[];
  initialAudit: AuditEventResponse[];
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const { executions, fills } = useRecentExecutionActivity({ portfolioSlug, initialExecutions, initialFills, liveSequence: liveState?.sequence, executionLimit: 20, fillLimit: 20 });
  const orders = useMemo(() => liveState?.order_history ?? [], [liveState?.order_history]);
  const deals = useMemo(() => liveState?.deal_history ?? [], [liveState?.deal_history]);
  const manual = countManualMt5Events(orders, deals);

  const [reconciliationState, setReconciliationState] = useState<ReconciliationSummaryResponse | null>(
    initialLiveState?.reconciliation ?? null,
  );
  const [auditState, setAuditState] = useState<AuditEventResponse[]>(initialAudit);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(
    (initialLiveState?.reconciliation?.mismatches ?? []).find((item) => item.status !== "match")?.symbol ?? null,
  );
  const [incidentStatus, setIncidentStatus] = useState<IncidentStatus>("acknowledged");
  const [reason, setReason] = useState("operator_reviewed");
  const [operatorNote, setOperatorNote] = useState("");
  const [resolutionNote, setResolutionNote] = useState("");
  const [submitPending, setSubmitPending] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

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
  const openIncidentCount = incidents.filter((item) => item.incident_status !== "resolved").length;
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
          <StatusBadge label={liveState?.status ?? "unknown"} tone={liveState?.status === "ok" ? "success" : "warning"} />
          <StatusBadge label={transport} tone={transport === "stream" ? "success" : "warning"} />
          <ButtonLink href={`/desk/incidents?portfolio=${portfolioSlug}`} variant="secondary">Incidents</ButtonLink>
        </>}
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <MetricBlock label="Orders" value={String(orders.length)} hint={`${manual.orders} manual`} tone="accent" />
        <MetricBlock label="Deals" value={String(deals.length)} hint={`${manual.deals} manual`} tone="warning" />
        <MetricBlock label="Desk attempts" value={String(executions.length)} tone="success" />
        <MetricBlock label="Fills" value={String(fills.length)} hint={liveState?.generated_at ? formatTimestamp(liveState.generated_at) : undefined} />
        <MetricBlock label="Mismatches" value={String(mismatchCount)} tone={mismatchCount > 0 ? "warning" : "success"} />
        <MetricBlock label="Open incidents" value={String(openIncidentCount)} tone={openIncidentCount > 0 ? "warning" : "neutral"} />
      </section>

      <LiveOperatorAlerts alerts={liveState?.operator_alerts ?? []} />

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Order history</h4>
          <OrderHistoryTable rows={orders} />
        </div>
        <div className="space-y-2">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Deal history</h4>
          <DealHistoryTable rows={deals} />
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
                        {incident.symbol} · {incident.incident_status ?? "acknowledged"}
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

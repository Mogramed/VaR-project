# ADR-0011 - Single Source of Truth for MT5 Business Data

- Status: Accepted
- Date: 2026-04-21
- Scope: VAR-011

## Context

The platform historically persisted multiple business-facing records in the local database while MT5 already provided the authoritative trading state for live portfolios.
This created drift risk and ambiguous ownership of truth across:

- MT5 broker state (orders/deals/positions/history)
- local business tables (for example `risk_decisions`)
- local technical logs (`audit_events`, operator runs, alerts)

VAR-009 and VAR-010 moved transaction history reads to MT5-backed data.
VAR-011 formalizes the persistence policy for live mode.

## Decision

For portfolios in `live_mt5` mode:

- MT5 is the business source of truth.
- Local database stores technical audit/operations evidence only.
- Trade decision records are no longer written to `risk_decisions`.
- Decision history APIs are served from audit events (with compatibility fallback for legacy rows).

For non-live portfolios (simulation/offline):

- Existing local decision persistence remains enabled.

## Consequences

### Positive

- Reduces business-data duplication in live mode.
- Eliminates local `risk_decisions` drift versus MT5-driven workflows.
- Keeps endpoint continuity through audit-backed reads.

### Trade-offs

- Live decision history depends on audit payload quality.
- Legacy local decision rows remain readable during transition (compatibility path), so mixed history can temporarily exist.

## Persistence Rules (Live Mode)

- Business decision row (`risk_decisions`): disabled.
- Technical audit event (`audit_events`): enabled.
- Technical alerting (`alerts`): enabled.
- Technical retention policy: TTL cleanup on audit events.

## Redundant Structures and Migration Plan

### Identified redundancy

- `risk_decisions` in live mode (business duplicate).

### Progressive legacy plan

1. Stop creating new live-mode rows in `risk_decisions` (implemented).
2. Keep read compatibility for historical rows during transition.
3. After observation period, archive and remove legacy-only live rows if no consumer remains.

## API Impact

- `/decisions/recent` and report decision history remain available.
- In live mode they are sourced from audit events first, with compatibility fallback to legacy local rows.
- Account-scoped filtering is preserved.

## Operational Notes

- Technical audit retention can be configured with:
  - `storage.technical_audit_retention_days`
  - `VAR_PROJECT_TECHNICAL_AUDIT_RETENTION_DAYS` (env override)

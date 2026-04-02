# MT5 Desk V1 Acceptance Runbook

Internal acceptance recipe for the mono-account MT5 desk.

## Goal

Validate that the platform remains the canonical risk, reconciliation, and incident layer above MT5 for a normal operator session.

## Preconditions

- MT5 terminal is open on the demo account.
- `var-project db upgrade` has already been applied.
- Backend, worker, frontend, and MT5 agent are running.
- The portfolio is configured in `live_mt5` or `hybrid` mode.

## Core Runtime Checks

1. Open `/desk`.
2. Confirm the MT5 bridge is visible as `ok`, or at worst briefly `degraded` during startup.
3. Confirm the overview live strip shows holdings, live risk, and capital.
4. Confirm no manual `POST /market-data/sync` step is required before snapshot, backtest, stress, or live analytics flows.

## Demo Trading Recipe

1. Open `/desk/execution`.
2. Preview a very small `EURUSD` exposure change.
3. Confirm the pre-trade summary shows:
   - requested exposure
   - broker lots
   - margin impact
   - post-trade VaR and budget posture
4. Submit the order on the demo account.
5. Confirm:
   - the execution result is `EXECUTED` or `PLACED`
   - the blotter shows the broker order and deal quickly
   - `filled_volume_lots` and `fill_ratio` are populated
6. Flatten the position from the platform.
7. Confirm the account returns to flat and the platform returns to `match`.

## Incident Workflow Recipe

1. Create a manual broker-side drift on the demo account.
2. Open `/desk/incidents`.
3. Confirm the mismatch is detected as one of:
   - `manual_trade_detected`
   - `orphan_live_position`
   - `desk_vs_broker_drift`
4. Move the incident to `investigating`.
5. Resolve the broker-side state.
6. Mark the incident `resolved`.
7. Confirm the reconciliation baseline is refreshed and the symbol returns to `match`.

## Reporting Recipe

1. Trigger snapshot, backtest, and report flows from the platform.
2. Confirm the report uses the live MT5 holdings snapshot when available.
3. Confirm the PDF and the UI use the same exposure, holdings, and lots vocabulary.

## Soak Test

Target: at least a few hours on a demo session.

Check continuously:

- bridge remains connected or visibly `stale/degraded` when MT5 is interrupted
- reconnection works after MT5 IPC loss or terminal restart
- no silent divergence between platform holdings and MT5 account state
- live alerts remain visible in overview, blotter, execution, and incidents
- worker `live_refresh` keeps snapshots and reports moving without a browser tab open

## Acceptance For V1

The desk is ready as an internal V1 when all of the following are true:

- the account state shown by the platform matches the MT5 demo account
- holdings, exposure, and lots are the primary product vocabulary
- pre-trade guardrails are understandable to an operator
- post-trade reconciliation is fast and actionable
- incidents can be acknowledged, investigated, resolved, and audited
- reports reflect the live desk state rather than a stale manual portfolio config

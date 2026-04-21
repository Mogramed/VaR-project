# MT5 Sync Pipeline (VAR-009)

This document describes the end-to-end MT5 sync pipeline used by `POST /market-data/sync`.

## Scope

The pipeline syncs and normalizes:

- instrument definitions
- market bars (bootstrap + incremental)
- MT5 order history
- MT5 deal history
- tick archive slices used by live analytics

## Normalized Mapping

- MT5 symbol metadata -> `instruments`
- MT5 OHLC bars -> `market_bars` (`symbol`, `timeframe`, `time_utc` unique)
- MT5 history orders -> `mt5_order_history` (`ticket` unique)
- MT5 history deals -> `mt5_deal_history` (`ticket` unique)
- sync run telemetry -> `market_data_sync_runs.details`

## Sync Strategy

- Bars:
  - bootstrap when no cached bars exist
  - incremental when cached bars exist (`latest_bar_time - overlap`)
- Orders/Deals:
  - bootstrap window on first run (`now - bootstrap_days`)
  - incremental window on subsequent runs (`latest_cached_at - overlap`)

Overlap windows are deliberate to tolerate transient MT5/API/network boundaries.

## Idempotence

- Bars are upserted logically on `(symbol, timeframe, time_utc)`.
- Orders are upserted by `ticket`.
- Deals are upserted by `ticket`.

Rerunning the same sync window does not create business duplicates.

## Checkpoints and Resume

Each running sync stores a progress checkpoint in `market_data_sync_runs.details.checkpoint`:

- current stage
- completed stages
- progress cursor (history windows)
- stage metrics

Detailed stage timing is persisted in `market_data_sync_runs.details.stage_events`.

When a previous run ended in `running`/`incomplete`, the next run stores a `resume_from` hint in details so support can reconstruct the recovery path.

## Failure Recovery

- Stage errors are captured in `details.errors`.
- Final status is:
  - `ok` when no stage error occurred
  - `incomplete` when one or more stages failed
- A stale `running` sync is automatically closed as `incomplete` by TTL protection.

The pipeline is intentionally re-runnable after partial incidents.

## Support Endpoints

- `GET /market-data/status`: current health and coverage view
- `POST /market-data/sync`: trigger a sync run
- `GET /market-data/sync/runs`: list recent sync runs with checkpoints and stage events
  - supports `limit` (default `25`, max `200`)
  - supports repeatable `status` filter (`status=ok&status=incomplete`)

## MT5 Read-Only Transaction History (VAR-010)

The blotter transaction history now reads directly from MT5 for display and export.
It no longer depends on local business-history tables for the operator-facing history view.

- `GET /mt5/history/transactions`
  - query: `portfolio_slug`, `account_id`, `date_from`, `date_to`, `symbol`, `type`, `sort`, `page`, `page_size`
  - `type` supports: `all`, `order`, `deal`, `manual`, `desk`
  - `sort` supports: `time_desc`, `time_asc`
- `GET /mt5/history/transactions/export`
  - same filter query parameters
  - `max_rows` controls CSV row cap (default `5000`, max `10000`)

# Observability Runbook (VAR-019)

## 1. Start the stack

```bash
docker compose up -d
```

Windows safe helper (avoids intermittent Compose monitor panics on `v5.x`):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker_up_safe.ps1 -FollowLogs
```

Git Bash variant:

```bash
powershell -ExecutionPolicy Bypass -File ./scripts/docker_up_safe.ps1 -FollowLogs
```

Endpoints:

- API metrics: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Alertmanager: `http://localhost:9093`
- Grafana: `http://localhost:3001` (default `admin/admin`, override with env vars)
- Grafana health API: `http://localhost:3001/api/health`

Note: `GET /api/alerts` can appear as `404` in Grafana logs on Grafana 11 (legacy route). This is not a platform failure.

## 2. What to check first during an incident

1. **API health/readiness**
- `GET /health`
- `GET /health/readiness`

2. **Operator lifecycle**
- `GET /operator/runs?limit=25`
- Grafana panels:
  - `Operator Runs by Status`
  - `Operator Failure Ratio by Action`

3. **MT5 bridge state**
- `GET /mt5/live/state?detail_level=summary`
- Grafana panel: `MT5 Bridge State`

4. **Reconciliation drift**
- `GET /reconciliation/summary`
- Grafana panels:
  - `Reconciliation Mismatches`
  - `Reconciliation Drift Counters`

## 3. Log correlation workflow

Logs are emitted with:

- `request_id`
- `run_id`
- `account_id`
- `action`

Recommended sequence:

1. Retrieve run details via `GET /operator/runs/{run_id}`.
2. Copy `request_id` and `run_id`.
3. Filter backend logs with these identifiers to trace the full lifecycle.

## 4. Alert meaning and immediate action

### `VarProjectApiHigh5xxRate` (critical)
- Meaning: sustained backend instability.
- Action: inspect `/health/readiness`, recent deploy/config changes, and API logs by `request_id`.

### `VarProjectOperatorFailureBurst` (critical)
- Meaning: operator actions failing repeatedly.
- Action: inspect `/operator/runs` for shared `error_code` and stage; validate worker/celery/Redis health.

### `VarProjectMt5BridgeDisconnected` (critical)
- Meaning: live MT5 feed unavailable.
- Action: validate MT5 agent availability (`VAR_PROJECT_MT5_AGENT_BASE_URL`), auth, and bridge errors.

### `VarProjectMarketDataSyncIncomplete` (warning)
- Meaning: the latest sync run remains incomplete for at least 10 minutes.
- Action: inspect latest `/market-data/sync/runs` details and missing symbol/timeframe stages.

### `VarProjectOperatorStaleRunsDetected` (warning)
- Meaning: queued/running runs exceeded stale TTL.
- Action: review worker queue throughput and stale thresholds.

### `VarProjectReconciliationCriticalMismatch` (warning)
- Meaning: material desk-vs-broker divergence.
- Action: inspect `/reconciliation/summary`, incidents, and execution lineage.

## 5. Threshold tuning guide

- API 5xx threshold: start at `5%` over `10m`.
- Operator failure burst: `>40%` with `>=3` failed terminal runs.
- MT5 disconnection: alert after `5m` to avoid transient noise.
- Sync incomplete/stale runs: tune based on typical sync duration and queue SLA.

If alerts are noisy:

1. increase `for:` durations first
2. then raise thresholds
3. keep critical MT5 disconnection conservative for live desks

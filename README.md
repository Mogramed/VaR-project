# VaR Risk Desk Platform

FX risk desk platform built around a FastAPI backend, a Next.js frontend, a worker for recurring analytics, and an optional Windows-side MT5 agent for demo execution.

## Supported Runtime Modes

- `var-project api`: serve the FastAPI backend.
- `var-project worker`: run snapshot, backtest, and report jobs.
- `var-project operator-worker`: process queued operator actions (`sync/snapshot/backtest/report`) via Celery.
- `var-project mt5-agent`: expose a local MT5 terminal to the backend when Docker is running on Linux containers.
- `var-project db upgrade`: apply the versioned SQL schema with Alembic.
- `var-project seed-demo`: bootstrap the local platform state from the tracked 60-day fixtures.
- `var-project demo-smoke`: run a fast API/demo readiness smoke check against a running backend.
- `frontend/`: Next.js operator UI and report export surface.

Unsupported legacy analytics scripts and the old Streamlit UI have been removed from the supported runtime path.

## Repo Layout

- `src/var_project/`: backend, risk engine, storage, MT5 execution, and API.
- `frontend/`: Next.js frontend and PDF export flow.
- `config/`: runtime settings and risk limits.
- `data/processed/`: small tracked regression fixtures used for local demos.
- `data/fixtures/`: notes about the fixture policy.
- `reports/`: generated runtime output, ignored by git.

## Local Run

Backend:

```bash
var-project db upgrade
var-project api --host 127.0.0.1 --port 8000
```

Direct Alembic workflow (equivalent schema result):

```bash
alembic upgrade head
# validates critical tables/columns + Alembic head revision
python scripts/check_storage_schema.py
```

If you target a non-default database URL, export `VAR_PROJECT_DATABASE_URL` before running these commands.

Worker:

```bash
var-project worker --once
```

Operator queue worker:

```bash
var-project operator-worker
```

## Operator Run Contract (VAR-003)

Operator actions are enqueue-first and status-driven:

- `POST /operator/actions/sync|snapshot|backtest|report` returns quickly with a run payload (`id`, `request_id`, `status`, `stage`).
- `GET /operator/runs/{run_id}` is the source of truth for lifecycle tracking (`queued`, `running`, `succeeded`, `failed`).
- `GET /operator/runs?...` lists recent/active runs for recovery after UI refresh or transient network errors.
  - supports `status_reason=timeout|abandoned|interrupted|...` for targeted diagnostics.
- `POST /operator/runs/{run_id}/interrupt` interrupts an active run and closes it as `failed` with `error_code=operator_interrupted` (idempotent if the run is already terminal).
- stale runs are auto-closed by the worker/API sweep with an explicit `status_reason`:
  - `timeout` for `running` runs exceeding action TTL.
  - `abandoned` for `queued` runs exceeding queue TTL.

Each run response includes SLA hints for consistent polling/timeout behavior:

- `poll_after_ms`
- `queued_timeout_seconds`
- `running_timeout_seconds`
- `sla_seconds`
- `interruptible`
- `status_reason` (for failed terminal diagnostics such as `timeout`, `abandoned`, `interrupted`)

Operator stale TTLs are configurable per action:

- `VAR_PROJECT_OPERATOR_STALE_QUEUED_SYNC|SNAPSHOT|BACKTEST|REPORT`
- `VAR_PROJECT_OPERATOR_STALE_RUNNING_SYNC|SNAPSHOT|BACKTEST|REPORT`

Worker monitoring (`GET /jobs/status`) now includes `operator_runs` counters (`status_counts`, `stale_closed_total`, `stale_reason_counts`, `recent_stale`) for stale-cleanup observability.

## Report Contract (VAR-016)

Report rendering is now driven by a versioned payload shared by backend, UI, and export:

- `GET /reports/latest` and `POST /reports/run` expose `report_contract`.
- `report_contract.version` is currently `report.v1`.
- Canonical VaR/ES/PnL values are under `report_contract.metrics`.
- Formatting/rounding and timezone metadata are carried by `report_contract.rounding` and `report_contract.timezone` (`UTC`).

## Backtest Confidence Guardrails (VAR-015)

Backtest validation now includes sample-size guardrails and an explicit confidence score:

- Minimum observation floors are applied per horizon (`1d=60`, `5d=80`, `10d=120`, then +20 every extra 5 days) and combined with the expected-exception floor.
- Under-sized samples are marked as low confidence in:
  - validation governance payloads (`confidence_score`, `confidence_level`, `confidence_reason`)
  - generated markdown reports (`Model Governance Surface` + `Multi-Horizon Validation`)
  - frontend model/report surfaces (confidence badges and explanatory text)
- Interpretation:
  - `HIGH`: sample size is sufficient across points.
  - `MEDIUM`: mostly sufficient, but with reduced margin.
- `LOW`: insufficient observations; results should be treated as indicative only.

## Dashboard View Controls (VAR-017)

The operator dashboard now supports user-level personalization through the `Configure` drawer in the top bar:

- Toggle overview widgets on/off.
- Show/hide sidebar pages.
- Apply global filters for symbol, horizon (`1d/5d/10d`), and model preference.
- Start from presets (`trading`, `risk-monitoring`, `minimal`) and fine-tune as needed.
- Preferences auto-save in browser storage and persist after refresh.

Global symbol filtering is applied consistently on live operational pages (MT5 Ops, Blotter, Execution, Decisions, Attribution, Capital, Universe). Model/horizon preferences are used by risk-oriented surfaces (Overview and Models).

Frontend:

```bash
python scripts/generate_frontend_api_types.py
docker compose up frontend
```

Bootstrap a fresh demo state:

```bash
var-project seed-demo
```

Run a pre-demo smoke check (expects API already running):

```bash
var-project demo-smoke --base-url http://127.0.0.1:8000
```

The Docker path is the canonical frontend workflow. Host-side `npm` remains optional for local debugging only.

## Docker

The Docker stack exposes:

- API debug root: `http://localhost:8000/`
- API debug health: `http://localhost:8000/health`
- API debug docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:3000`
- Nginx front door: `http://localhost:8081` (override with `VAR_PROJECT_NGINX_PORT`)
- Frontend through Nginx: `http://localhost:8081/`
- API through Nginx: `http://localhost:8081/backend/health`

Run (optimized default profile):

```bash
docker compose up -d --build
docker compose logs -f
```

Enable observability when needed:

```bash
docker compose --profile observability up -d --build
docker compose logs -f prometheus alertmanager grafana
```

Windows safe helper (recommended on Docker Compose `v5.x`):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker_up_safe.ps1 -FollowLogs
```

With observability:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker_up_safe.ps1 -WithObservability -FollowLogs
```

If you use Git Bash, use forward slashes:

```bash
powershell -ExecutionPolicy Bypass -File ./scripts/docker_up_safe.ps1 -FollowLogs
```

Or use the cmd wrapper (works from `cmd.exe` / PowerShell):

```cmd
scripts\docker_up_safe.cmd -FollowLogs
```

Notes:

- `db-migrate` is a one-shot service that runs `var-project db upgrade` before `api`, `worker`, `celery-worker`, and `celery-worker-2`.
- `api` and `worker` no longer create the schema silently at startup.
- `GET /health` is strict on DB schema integrity (critical tables/columns + Alembic head revision) and returns `status=unhealthy` when drift is detected.
- `redis` + `celery-worker` + `celery-worker-2` provide asynchronous execution for operator actions triggered from the frontend (two parallel operator workers by default).
- `seed-demo` is the supported way to generate a demo-ready database state beyond the tracked fixtures.
- `localhost:${VAR_PROJECT_NGINX_PORT:-8081}` is the operator-facing front door. `localhost:8000` remains the direct FastAPI debug surface.
- `GET /` on `localhost:8000` now returns a small discovery payload that points to `/health` and `/docs` instead of a raw `404`.
- On Docker Compose `v5.x`, `docker compose up --build` in attached mode can panic in the Compose monitor on some hosts. Prefer detached mode (`-d`) plus `docker compose logs -f`.
- `GET /api/alerts` returning `404` in Grafana logs is expected on Grafana `11.x` (legacy endpoint removed); use `/api/alertmanager/grafana/api/v2/alerts` instead.

Resource tuning defaults:

- Docker defaults are optimized for lower RAM pressure (`VAR_PROJECT_API_WORKERS=1`, reduced DB pools, slower worker polling cadence).
- Worker cadence can be tuned with:
  - `VAR_PROJECT_WORKER_LOOP_SLEEP_SECONDS`
  - `VAR_PROJECT_WORKER_SNAPSHOT_INTERVAL_SECONDS`
  - `VAR_PROJECT_WORKER_LIVE_REFRESH_INTERVAL_SECONDS`
  - `VAR_PROJECT_WORKER_BACKTEST_INTERVAL_SECONDS`
  - `VAR_PROJECT_WORKER_REPORT_INTERVAL_SECONDS`
- Operator queue throughput can be tuned with:
  - `VAR_PROJECT_CELERY_WORKER_CONCURRENCY`
- Frontend live/refresh cadence can be tuned with:
  - `NEXT_PUBLIC_MT5_STREAM_MODE`
  - `NEXT_PUBLIC_MT5_POLL_INTERVAL_MS`
  - `NEXT_PUBLIC_MT5_MAX_POLL_INTERVAL_MS`
  - `NEXT_PUBLIC_DESK_ARTIFACT_QUERY_REFETCH_MS`

## Observability & Alerting (VAR-019)

Observability services are profile-gated and only run when `--profile observability` is used.

Backend telemetry now exposes:

- structured/correlated logs with `request_id`, `run_id`, `account_id`, and `action`.
- Prometheus metrics at `GET /metrics` (not part of OpenAPI schema).
- critical operational metrics for:
  - API latency/error rate
  - operator run status/failure ratio/stale closures
  - market-data sync status
  - MT5 bridge health
  - reconciliation mismatch/drift counters

Grafana is provisioned with a ready-to-use dashboard:

- **VaR Platform Observability** (folder: `VaR Platform`)

Prometheus alert rules are preloaded for:

- sustained API 5xx rate
- operator failure bursts
- MT5 bridge disconnection
- incomplete market-data sync runs
- stale operator run closures
- critical reconciliation mismatches

Quick links:

- [`docs/observability_runbook.md`](docs/observability_runbook.md)
- [`infra/observability/prometheus/alerts/var_project_alerts.yml`](infra/observability/prometheus/alerts/var_project_alerts.yml)

Logging defaults to `INFO` for both console and rotating file output. Raise verbosity explicitly only when needed:

```bash
VAR_PROJECT_LOG_LEVEL=DEBUG
# optional fine-grained overrides
VAR_PROJECT_LOG_CONSOLE_LEVEL=INFO
VAR_PROJECT_LOG_FILE_LEVEL=DEBUG
```

Regenerate frontend API types after backend schema changes:

```bash
python scripts/generate_frontend_api_types.py
```

The Python image now ships only the minimal processed demo fixtures needed for non-live snapshot/backtest/report flows.

## Platform Workflow

Recommended order on a fresh workspace:

```bash
var-project db upgrade
var-project seed-demo
docker compose up -d --build
docker compose logs -f
```

This produces a migrated database and the API-worker-frontend stack with memory-friendly defaults.

## MT5 Demo Execution

Main endpoints:

- `GET /mt5/status`
- `GET /mt5/account`
- `GET /mt5/positions`
- `GET /mt5/orders`
- `GET /market-data/status`
- `POST /market-data/sync`
- `GET /market-data/sync/runs`
- `POST /execution/preview`
- `POST /execution/submit`
- `GET /execution/recent`

Recommended workflow:

1. Start the Windows-side MT5 agent close to the terminal:

```bash
var-project mt5-agent --host 0.0.0.0 --port 8010
```

Or directly from the repo:

```bash
python scripts/run_mt5_agent.py --host 0.0.0.0 --port 8010
```

2. Point the API to that agent:

```bash
VAR_PROJECT_MT5_AGENT_BASE_URL=http://host.docker.internal:8010
VAR_PROJECT_MT5_AGENT_API_KEY=change_me_local_only
```

3. Restart the relevant services:

```bash
docker compose up -d --build api worker celery-worker frontend nginx
```

Notes:

- The browser never talks directly to MT5.
- MT5 execution remains demo-only and risk-guarded.
- Portfolios in `live_mt5` are MT5-only: configured holdings/exposure are not used as a fallback for live risk/capital compute.
- If MT5 is unavailable or returns an empty live book, live compute endpoints fail fast with actionable `mt5_live_unavailable` guidance.
- Risk analytics now emit `data_quality.status=no_exposure` when all symbol exposures stay under configured epsilon thresholds (`risk.no_exposure_epsilon_eur` + optional `risk.no_exposure_epsilon_by_symbol`).
- Multi-model VaR diagnostics are available through `GET /risk/diagnostics` (input trace, debug rows, and coherence checks).
- Suspicious non-zero equalities (`VaR` + `ES`) across models are exposed in `model_diagnostics.coherence_checks.suspicious_equalities` and surface as `VAR_MODEL_EQUALITY_SUSPECT` live alerts.
- Blotter history is read-only from MT5 via `GET /mt5/history/transactions` (filters + pagination) and `GET /mt5/history/transactions/export` (CSV).
- `/execution/recent` now keeps both pre-trade dry runs (`PREVIEW`) and executed attempts for a single end-to-end history.
- MT5 reconciliation is exposed via `GET /reconciliation/summary` (exposure/volume/PnL drift, severity, probable cause) and `GET /reconciliation/history` (historized divergence snapshots).
- Rebuild `api` and `worker` after changing `config/` or tracked fixture data.
- Rebuild `frontend` after changing code under `frontend/`.

## V1 Acceptance Runbook

The internal MT5 acceptance recipe and soak checklist live in:

- [`docs/mt5_v1_acceptance_runbook.md`](docs/mt5_v1_acceptance_runbook.md)
- [`docs/mt5_sync_pipeline.md`](docs/mt5_sync_pipeline.md)

## Demo Docs

For presentation and soutenance support:

- [`docs/demo_runbook.md`](docs/demo_runbook.md)
- [`docs/demo_checklist.md`](docs/demo_checklist.md)
- [`docs/demo_pitch.md`](docs/demo_pitch.md)

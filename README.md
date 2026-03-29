# VaR Risk Desk Platform

FX risk desk platform built around a FastAPI backend, a Next.js frontend, a worker for recurring analytics, and an optional Windows-side MT5 agent for demo execution.

## Supported Runtime Modes

- `var-project api`: serve the FastAPI backend.
- `var-project worker`: run snapshot, backtest, and report jobs.
- `var-project mt5-agent`: expose a local MT5 terminal to the backend when Docker is running on Linux containers.
- `var-project db upgrade`: apply the versioned SQL schema with Alembic.
- `var-project seed-demo`: bootstrap the local platform state from the tracked 60-day fixtures.
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

Worker:

```bash
var-project worker --once
```

Frontend:

```bash
python scripts/generate_frontend_api_types.py
docker compose up frontend
```

Bootstrap a fresh demo state:

```bash
var-project seed-demo
```

The Docker path is the canonical frontend workflow. Host-side `npm` remains optional for local debugging only.

## Docker

The Docker stack exposes:

- API: `http://localhost:8000/health`
- Frontend: `http://localhost:3000`
- Nginx front door: `http://localhost:8080`
- API through Nginx: `http://localhost:8080/backend/health`

Run:

```bash
docker compose build
docker compose up
```

Notes:

- `db-migrate` is a one-shot service that runs `var-project db upgrade` before `api` and `worker`.
- `api` and `worker` no longer create the schema silently at startup.
- `seed-demo` is the supported way to generate a demo-ready database state beyond the tracked fixtures.

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
docker compose up --build
```

This produces a migrated database, a seeded report/snapshot/backtest baseline, and the full API-worker-frontend stack.

## MT5 Demo Execution

Main endpoints:

- `GET /mt5/status`
- `GET /mt5/account`
- `GET /mt5/positions`
- `GET /mt5/orders`
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
docker compose up -d --build api worker frontend nginx
```

Notes:

- The browser never talks directly to MT5.
- MT5 execution remains demo-only and risk-guarded.
- Rebuild `api` and `worker` after changing `config/` or tracked fixture data.
- Rebuild `frontend` after changing code under `frontend/`.

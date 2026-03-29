# Frontend

Next.js operator frontend for the VaR Risk Desk Platform.

## Main Routes

- `/en`
- `/fr`
- `/desk`
- `/desk/models`
- `/desk/attribution`
- `/desk/capital`
- `/desk/decisions`
- `/desk/simulation`
- `/desk/reports`
- `/desk/live`
- `/desk/execution`

## Environment

Copy `.env.example` if you want to point the frontend to a custom backend:

```bash
VAR_PROJECT_API_BASE_URL=http://127.0.0.1:8000
```

## API Types

The frontend API types are generated from the backend OpenAPI schema:

```bash
python ../scripts/generate_frontend_api_types.py
```

The generated files are:

- `src/lib/api/generated-schema.ts`
- `src/lib/api/types.ts`

## Docker-First Dev

Canonical workflow:

```bash
python ../scripts/generate_frontend_api_types.py
docker compose up frontend
```

Optional host-side workflow:

```bash
npm ci
npm run dev
```

## Production Build

```bash
npm run typecheck
npm run lint
npm run build
```

The Docker image uses Next.js standalone output and runs `node server.js`.
Its Docker quality stage runs `typecheck`, `lint`, and `build` before producing the runtime image.

The operator shell expects the backend to expose:

- `GET /health`
- `GET /jobs/status`
- `POST /snapshots/run`
- `POST /backtests/run`
- `POST /reports/run`

Known frontend debt:

- TanStack Table still emits the upstream `react-hooks/incompatible-library` warning during lint on `data-grid.tsx`. It is tracked as a known non-blocking warning.

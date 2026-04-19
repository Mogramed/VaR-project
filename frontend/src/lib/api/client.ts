import type {
  AlertSummary,
  AuditEventResponse,
  BacktestFrameResponse,
  BacktestRunResponse,
  CapitalRebalanceRequest,
  CapitalUsageSnapshotResponse,
  DealHistoryEntryResponse,
  DeskDefinitionResponse,
  DeskSnapshotResponse,
  ExecutionFillResponse,
  ExecutionPreviewResponse,
  ExecutionRequest,
  ExecutionResultResponse,
  HealthResponse,
  HealthDependenciesResponse,
  HoldingSnapshotResponse,
  InstrumentDefinitionResponse,
  MarketDataSyncRequest,
  MarketDataSyncStatusResponse,
  ModelComparisonResponse,
  MT5AccountSnapshotResponse,
  MT5LiveEventResponse,
  MT5LiveStateResponse,
  MT5PendingOrderResponse,
  MT5PositionResponse,
  MT5TerminalStatusResponse,
  OperatorRunResponse,
  OrderHistoryEntryResponse,
  PortfolioExposureResponse,
  PortfolioSummary,
  ReconciliationSummaryResponse,
  ReconciliationAcknowledgementResponse,
  ReportContentResponse,
  ReportRunResponse,
  RiskAttributionResponse,
  RiskBudgetResponse,
  RiskDecisionResponse,
  RunBacktestRequest,
  RunSnapshotRequest,
  SnapshotRunResponse,
  SnapshotSummary,
  StressReportResponse,
  StressScenarioRequest,
  TradeProposalRequest,
  LiveRiskSummaryResponse,
  ValidationRunSummary,
  WorkerStatusResponse,
} from "@/lib/api/types";

type QueryValue =
  | string
  | number
  | boolean
  | undefined
  | null
  | Array<string | number | boolean>;
type Query = Record<string, QueryValue>;

type ErrorPayload = {
  detail?: string | Record<string, unknown>;
  error_code?: string;
  error_message?: string;
  hint?: string;
  request_id?: string;
  run_id?: number | string;
  stage?: string;
};

const NETWORK_TIMEOUT_CODES = new Set([
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT",
  "UND_ERR_CONNECT_TIMEOUT",
  "ABORT_ERR",
]);
const NETWORK_RETRY_CODES = new Set([
  ...NETWORK_TIMEOUT_CODES,
  "ECONNRESET",
  "ECONNREFUSED",
  "EPIPE",
  "ENOTFOUND",
]);
const RETRYABLE_STATUS_CODES = new Set([429, 502, 503, 504]);
const SERVER_SUMMARY_CACHE = new Map<string, { expiresAt: number; payload: unknown }>();
const SERVER_SUMMARY_INFLIGHT = new Map<string, Promise<unknown>>();

export class ApiError extends Error {
  status: number;
  errorCode?: string;
  requestId?: string;
  runId?: number | string;
  hint?: string;

  constructor(status: number, payload: ErrorPayload, fallbackDetail: string) {
    const detail =
      (typeof payload.detail === "string" ? payload.detail : undefined)
      ?? payload.error_message
      ?? fallbackDetail
      ?? "API request failed.";
    const suffix = [
      payload.hint,
      payload.run_id != null ? `run ${payload.run_id}` : null,
      payload.request_id ? `request ${payload.request_id}` : null,
    ]
      .filter(Boolean)
      .join(" | ");
    super(suffix ? `${detail} (${suffix})` : detail);
    this.name = "ApiError";
    this.status = status;
    this.errorCode = payload.error_code;
    this.requestId = payload.request_id;
    this.runId = payload.run_id;
    this.hint = payload.hint;
  }
}

function normalizeErrorPayload(raw: unknown): ErrorPayload {
  const payload = (raw ?? {}) as ErrorPayload;
  if (payload.detail && typeof payload.detail === "object") {
    const nested = payload.detail as ErrorPayload;
    return {
      ...payload,
      ...nested,
      detail:
        typeof nested.detail === "string"
          ? nested.detail
          : payload.error_message
            ?? nested.error_message
            ?? "API request failed.",
    };
  }
  return payload;
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractNestedErrorCode(error: unknown): string | null {
  let cursor: unknown = error;
  const seen = new Set<unknown>();
  while (cursor && typeof cursor === "object" && !seen.has(cursor)) {
    seen.add(cursor);
    const record = cursor as Record<string, unknown>;
    if (typeof record.code === "string" && record.code.trim()) {
      return record.code;
    }
    cursor = record.cause;
  }
  return null;
}

function isTimeoutLikeError(error: unknown, code: string | null): boolean {
  if (code && NETWORK_TIMEOUT_CODES.has(code)) {
    return true;
  }
  const message = error instanceof Error ? error.message.toLowerCase() : "";
  return (
    message.includes("timeout")
    || message.includes("timed out")
    || message.includes("aborted")
  );
}

function isRetryableNetworkError(error: unknown, code: string | null): boolean {
  if (code && NETWORK_RETRY_CODES.has(code)) {
    return true;
  }
  const message = error instanceof Error ? error.message.toLowerCase() : "";
  return (
    message.includes("socket")
    || message.includes("connect")
    || message.includes("terminated")
    || message.includes("fetch failed")
  );
}

function resolveServerSummaryCacheTtlMs(path: string, query?: Query): number {
  const normalized = path.trim().toLowerCase();
  if (/^\/desks\/[^/]+\/overview$/.test(normalized)) {
    return 1_500;
  }
  if (normalized === "/mt5/live/state") {
    const detailLevel = String(query?.detail_level ?? "").toLowerCase();
    if (detailLevel === "summary") {
      return 1_500;
    }
  }
  return 0;
}

function ensureSlash(value: string) {
  return value.endsWith("/") ? value : `${value}/`;
}

function getServerApiBaseUrl() {
  return ensureSlash(
    process.env.VAR_PROJECT_API_BASE_URL ??
      process.env.NEXT_PUBLIC_API_BASE_URL ??
      "http://127.0.0.1:8000",
  );
}

function buildUrl(path: string, query?: Query) {
  const cleanPath = path.startsWith("/") ? path.slice(1) : path;
  const base =
    typeof window === "undefined"
      ? new URL(cleanPath, getServerApiBaseUrl())
      : new URL(`/api/proxy/${cleanPath}`, window.location.origin);

  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value == null || value === "") {
        continue;
      }
      if (Array.isArray(value)) {
        for (const item of value) {
          if (item == null || item === "") {
            continue;
          }
          base.searchParams.append(key, String(item));
        }
        continue;
      }
      base.searchParams.set(key, String(value));
    }
  }

  return base.toString();
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    let payload: ErrorPayload = {};
    try {
      payload = normalizeErrorPayload(await response.json());
      detail =
        (typeof payload.detail === "string" ? payload.detail : undefined)
        ?? payload.error_message
        ?? detail;
    } catch {
      const text = await response.text();
      if (text) {
        detail = text;
      }
    }
    throw new ApiError(response.status, payload, detail);
  }

  return (await response.json()) as T;
}

async function request<T>(
  path: string,
  options: {
    method?: "GET" | "POST";
    query?: Query;
    json?: unknown;
    revalidateSeconds?: number;
    timeoutMs?: number;
  } = {},
) {
  const isServer = typeof window === "undefined";
  const method = options.method ?? "GET";
  const normalizedPath = path.trim().toLowerCase();
  const isOperatorEnqueueRequest =
    method === "POST" && normalizedPath.startsWith("/operator/actions/");
  const isOperatorRunStatusRequest =
    method === "GET"
    && (normalizedPath === "/operator/runs" || normalizedPath.startsWith("/operator/runs/"));
  const isOperatorInterruptRequest =
    method === "POST"
    && normalizedPath.startsWith("/operator/runs/")
    && normalizedPath.endsWith("/interrupt");
  const retryableUnsafeMethod = isOperatorEnqueueRequest || isOperatorInterruptRequest;
  const url = buildUrl(path, options.query);
  const isCacheableServerGet =
    isServer
    && method === "GET"
    && (options.revalidateSeconds ?? 0) > 0;
  const serverSummaryCacheTtlMs =
    isServer && method === "GET"
      ? resolveServerSummaryCacheTtlMs(path, options.query)
      : 0;
  const timeoutMs =
    options.timeoutMs
    ?? (isOperatorEnqueueRequest ? 12_000
      : isOperatorInterruptRequest ? 12_000
        : isOperatorRunStatusRequest ? 10_000
          : method === "GET" ? 12_000 : 20_000);
  const maxAttempts =
    isOperatorRunStatusRequest ? 3 : (method === "GET" || retryableUnsafeMethod ? 2 : 1);

  const nextRetryDelayMs = (attempt: number) =>
    Math.min(1_200, Math.max(120, 180 * (2 ** Math.max(attempt - 1, 0))));

  const execute = async (): Promise<T> => {
    let response: Response | null = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      try {
        response = await fetch(url, {
          method,
          cache: isCacheableServerGet ? "force-cache" : "no-store",
          ...(isCacheableServerGet
            ? { next: { revalidate: options.revalidateSeconds } }
            : {}),
          headers: options.json ? { "Content-Type": "application/json" } : undefined,
          body: options.json ? JSON.stringify(options.json) : undefined,
          signal: AbortSignal.timeout(Math.max(timeoutMs, 1)),
        });
      } catch (error) {
        const errorCode = extractNestedErrorCode(error);
        const timedOut = isTimeoutLikeError(error, errorCode);
        if (
          attempt < maxAttempts
          && isRetryableNetworkError(error, errorCode)
          && (method === "GET" || retryableUnsafeMethod)
        ) {
          await delay(nextRetryDelayMs(attempt));
          continue;
        }
        if (timedOut) {
          throw new ApiError(
            504,
            {
              error_code: "frontend_request_timeout",
              error_message: "The request timed out.",
              hint: "Retry or use sync/snapshot actions to refresh data.",
            },
            "The request timed out.",
          );
        }
        throw new ApiError(
          502,
          {
            error_code: "frontend_request_failed",
            error_message: "The upstream API request failed.",
            hint: "Verify backend health and retry.",
          },
          "The upstream API request failed.",
        );
      }
      if (
        response
        && attempt < maxAttempts
        && (method === "GET" || retryableUnsafeMethod)
        && RETRYABLE_STATUS_CODES.has(response.status)
      ) {
        await delay(nextRetryDelayMs(attempt));
        continue;
      }
      if (response) {
        return parseResponse<T>(response);
      }
    }
    throw new ApiError(
      502,
      {
        error_code: "frontend_request_failed",
        error_message: "No response received from upstream API.",
        hint: "Retry once backend services are healthy.",
      },
      "No response received from upstream API.",
    );
  };

  if (serverSummaryCacheTtlMs <= 0) {
    return execute();
  }
  const cached = SERVER_SUMMARY_CACHE.get(url);
  const now = Date.now();
  if (cached && cached.expiresAt > now) {
    return cached.payload as T;
  }
  const inflight = SERVER_SUMMARY_INFLIGHT.get(url);
  if (inflight) {
    return (await inflight) as T;
  }
  const pending = execute();
  SERVER_SUMMARY_INFLIGHT.set(url, pending);
  try {
    const payload = await pending;
    SERVER_SUMMARY_CACHE.set(url, {
      expiresAt: now + serverSummaryCacheTtlMs,
      payload,
    });
    return payload;
  } finally {
    SERVER_SUMMARY_INFLIGHT.delete(url);
  }
}

const FALLBACK_HEALTH: HealthResponse = {
  status: "unavailable",
  repo_root: "",
  database_url: "",
  portfolio_slug: "",
  portfolio_mode: null,
  portfolio_count: 0,
  desk_slug: "main",
  latest_artifacts: {},
  defaults: {},
};

export const api = {
  health: () => request<HealthResponse>("/health", { revalidateSeconds: 10 }),
  healthDependencies: () =>
    request<HealthDependenciesResponse>("/health/dependencies", { revalidateSeconds: 5 }),
  safeHealth: () =>
    request<HealthResponse>("/health", { revalidateSeconds: 10 }).catch(
      () => FALLBACK_HEALTH,
    ),
  jobsStatus: () => request<WorkerStatusResponse>("/jobs/status", { revalidateSeconds: 10 }),
  portfolios: () => request<PortfolioSummary[]>("/portfolios", { revalidateSeconds: 30 }),
  desks: () => request<DeskDefinitionResponse[]>("/desks", { revalidateSeconds: 30 }),
  deskOverview: (deskSlug: string) =>
    request<DeskSnapshotResponse>(`/desks/${deskSlug}/overview`, {
      revalidateSeconds: 10,
    }),
  latestSnapshot: (portfolioSlug?: string, source?: string) =>
    request<SnapshotSummary>("/snapshots/latest", {
      query: { source: source ?? "auto", portfolio_slug: portfolioSlug },
    }),
  latestAttribution: (portfolioSlug?: string, source?: string) =>
    request<RiskAttributionResponse>("/snapshots/attribution/latest", {
      query: { source: source ?? "auto", portfolio_slug: portfolioSlug },
    }),
  latestBudget: (portfolioSlug?: string, source?: string) =>
    request<RiskBudgetResponse>("/snapshots/budget/latest", {
      query: { source: source ?? "auto", portfolio_slug: portfolioSlug },
    }),
  riskSummary: (portfolioSlug?: string) =>
    request<LiveRiskSummaryResponse>("/risk/summary", {
      query: { portfolio_slug: portfolioSlug },
    }),
  riskContributions: (portfolioSlug?: string, source?: string) =>
    request<RiskAttributionResponse>("/risk/contributions", {
      query: { portfolio_slug: portfolioSlug, source },
    }),
  mt5Status: () => request<MT5TerminalStatusResponse>("/mt5/status"),
  mt5Account: () => request<MT5AccountSnapshotResponse>("/mt5/account"),
  mt5Positions: (portfolioSlug?: string) =>
    request<MT5PositionResponse[]>("/mt5/positions", {
      query: { portfolio_slug: portfolioSlug },
    }),
  mt5Orders: (portfolioSlug?: string) =>
    request<MT5PendingOrderResponse[]>("/mt5/orders", {
      query: { portfolio_slug: portfolioSlug },
    }),
  mt5LiveState: (
    portfolioSlug?: string,
    options?: { detailLevel?: "summary" | "full" | "inspector" },
  ) =>
    request<MT5LiveStateResponse>("/mt5/live/state", {
      query: {
        portfolio_slug: portfolioSlug,
        detail_level: options?.detailLevel,
      },
    }),
  mt5AnalyticsSeries: (
    portfolioSlug?: string,
    options?: { windowMinutes?: number; maxPoints?: number },
  ) =>
    request<{
      generated_at: string;
      portfolio_slug: string;
      window_minutes: number;
      market_closed: boolean;
      market_reference_timestamp?: string | null;
      points: Array<{
        timestamp: string;
        balance?: number | null;
        equity?: number | null;
        margin_free?: number | null;
        margin_level?: number | null;
        profit?: number | null;
        avg_spread_bps?: number | null;
        tick_age_seconds?: number | null;
      }>;
    }>("/mt5/analytics/series", {
      query: {
        portfolio_slug: portfolioSlug,
        window_minutes: options?.windowMinutes,
        max_points: options?.maxPoints,
      },
    }),
  mt5LiveEvents: (portfolioSlug?: string, after = 0, limit = 100) =>
    request<MT5LiveEventResponse[]>("/mt5/live/events", {
      query: { portfolio_slug: portfolioSlug, after, limit },
    }),
  mt5HistoryOrders: (portfolioSlug?: string, limit = 100) =>
    request<OrderHistoryEntryResponse[]>("/mt5/history/orders", {
      query: { portfolio_slug: portfolioSlug, limit },
    }),
  mt5HistoryDeals: (portfolioSlug?: string, limit = 100) =>
    request<DealHistoryEntryResponse[]>("/mt5/history/deals", {
      query: { portfolio_slug: portfolioSlug, limit },
    }),
  marketDataStatus: (portfolioSlug?: string) =>
    request<MarketDataSyncStatusResponse>("/market-data/status", {
      query: { portfolio_slug: portfolioSlug },
    }),
  syncMarketData: (payload: MarketDataSyncRequest) =>
    request<MarketDataSyncStatusResponse>("/market-data/sync", {
      method: "POST",
      json: payload,
    }),
  enqueueOperatorSync: (payload: MarketDataSyncRequest) =>
    request<OperatorRunResponse>("/operator/actions/sync", {
      method: "POST",
      json: payload,
    }),
  enqueueOperatorSnapshot: (payload: RunSnapshotRequest) =>
    request<OperatorRunResponse>("/operator/actions/snapshot", {
      method: "POST",
      json: payload,
    }),
  enqueueOperatorBacktest: (payload: RunBacktestRequest) =>
    request<OperatorRunResponse>("/operator/actions/backtest", {
      method: "POST",
      json: payload,
    }),
  enqueueOperatorReport: (payload: { compare_path?: string | null; portfolio_slug?: string | null }) =>
    request<OperatorRunResponse>("/operator/actions/report", {
      method: "POST",
      json: payload,
    }),
  operatorRun: (runId: number) =>
    request<OperatorRunResponse>(`/operator/runs/${runId}`),
  interruptOperatorRun: (runId: number, reason?: string | null) =>
    request<OperatorRunResponse>(`/operator/runs/${runId}/interrupt`, {
      method: "POST",
      query: { reason: reason ?? undefined },
    }),
  operatorRuns: (options?: {
    portfolioSlug?: string;
    action?: string;
    statuses?: string[];
    limit?: number;
  }) =>
    request<OperatorRunResponse[]>("/operator/runs", {
      query: {
        portfolio_slug: options?.portfolioSlug,
        action: options?.action,
        limit: options?.limit,
        status: options?.statuses,
      },
    }),
  instruments: (portfolioSlug?: string) =>
    request<InstrumentDefinitionResponse[]>("/instruments", {
      query: { portfolio_slug: portfolioSlug },
    }),
  reconciliationSummary: (portfolioSlug?: string) =>
    request<ReconciliationSummaryResponse>("/reconciliation/summary", {
      query: { portfolio_slug: portfolioSlug },
    }),
  liveHoldings: (portfolioSlug?: string) =>
    request<HoldingSnapshotResponse[]>("/portfolio/live-holdings", {
      query: { portfolio_slug: portfolioSlug },
    }),
  liveExposure: (portfolioSlug?: string) =>
    request<PortfolioExposureResponse>("/portfolio/live-exposure", {
      query: { portfolio_slug: portfolioSlug },
    }),
  latestCapital: (portfolioSlug?: string, source?: string) =>
    request<CapitalUsageSnapshotResponse>("/capital/latest", {
      query: { portfolio_slug: portfolioSlug, source: source ?? "auto" },
    }),
  capitalHistory: (portfolioSlug?: string, limit = 24, source?: string) =>
    request<CapitalUsageSnapshotResponse[]>("/capital/history", {
      query: { portfolio_slug: portfolioSlug, limit, source },
    }),
  portfolioCapital: (portfolioSlug: string, source?: string) =>
    request<CapitalUsageSnapshotResponse>(`/portfolios/${portfolioSlug}/capital`, {
      query: { source: source ?? "auto" },
    }),
  latestValidation: (portfolioSlug?: string) =>
    request<ValidationRunSummary>("/validations/latest", {
      query: { portfolio_slug: portfolioSlug },
    }),
  latestModelComparison: (portfolioSlug?: string) =>
    request<ModelComparisonResponse>("/models/compare/latest", {
      query: { portfolio_slug: portfolioSlug },
    }),
  latestBacktestFrame: (portfolioSlug?: string, limit = 320) =>
    request<BacktestFrameResponse>("/backtests/frame/latest", {
      query: { portfolio_slug: portfolioSlug, limit },
    }),
  recentAlerts: (limit = 20) =>
    request<AlertSummary[]>("/alerts", { query: { limit } }),
  activeAlerts: (limit = 20, portfolioSlug?: string) =>
    request<AlertSummary[]>("/alerts/active", {
      query: { limit, portfolio_slug: portfolioSlug },
    }),
  runSnapshot: (payload: RunSnapshotRequest) =>
    request<SnapshotRunResponse>("/snapshots/run", {
      method: "POST",
      json: payload,
    }),
  runBacktest: (payload: RunBacktestRequest) =>
    request<BacktestRunResponse>("/backtests/run", {
      method: "POST",
      json: payload,
    }),
  recentDecisions: (portfolioSlug?: string, limit = 20) =>
    request<RiskDecisionResponse[]>("/decisions/recent", {
      query: { portfolio_slug: portfolioSlug, limit },
    }),
  evaluateDecision: (payload: TradeProposalRequest) =>
    request<RiskDecisionResponse>("/decisions/evaluate", {
      method: "POST",
      json: payload,
    }),
  previewExecution: (payload: ExecutionRequest) =>
    request<ExecutionPreviewResponse>("/execution/preview", {
      method: "POST",
      json: payload,
    }),
  submitExecution: (payload: ExecutionRequest) =>
    request<ExecutionResultResponse>("/execution/submit", {
      method: "POST",
      json: payload,
    }),
  recentExecutionResults: (portfolioSlug?: string, limit = 20) =>
    request<ExecutionResultResponse[]>("/execution/recent", {
      query: { portfolio_slug: portfolioSlug, limit },
    }),
  recentExecutionFills: (portfolioSlug?: string, limit = 50) =>
    request<ExecutionFillResponse[]>("/execution/fills/recent", {
      query: { portfolio_slug: portfolioSlug, limit },
    }),
  latestReport: (portfolioSlug?: string, reportId?: number) =>
    request<ReportContentResponse>("/reports/latest", {
      query: { portfolio_slug: portfolioSlug, report_id: reportId },
    }),
  runReport: (comparePath?: string, portfolioSlug?: string) =>
    request<ReportRunResponse>("/reports/run", {
      method: "POST",
      json: {
        compare_path: comparePath ?? null,
        portfolio_slug: portfolioSlug ?? null,
      },
    }),
  reportDecisionHistory: (portfolioSlug?: string, limit = 25) =>
    request<RiskDecisionResponse[]>("/reports/decision-history", {
      query: { portfolio_slug: portfolioSlug, limit },
    }),
  reportCapitalHistory: (portfolioSlug?: string, limit = 25, source?: string) =>
    request<CapitalUsageSnapshotResponse[]>("/reports/capital-history", {
      query: { portfolio_slug: portfolioSlug, limit, source },
    }),
  recentAudit: (portfolioSlug?: string, limit = 50) =>
    request<AuditEventResponse[]>("/audit/recent", {
      query: { portfolio_slug: portfolioSlug, limit },
    }),
  rebalanceCapital: (payload: CapitalRebalanceRequest) =>
    request<CapitalUsageSnapshotResponse>("/capital/rebalance", {
      method: "POST",
      json: payload,
    }),
  acknowledgeReconciliation: (payload: {
    portfolio_slug?: string;
    symbol: string;
    reason?: string;
    operator_note?: string;
    incident_status?: string;
    resolution_note?: string;
  }) =>
    request<{ acknowledged: boolean; symbol: string; audit_event_id: number }>(
      "/reconciliation/acknowledge",
      { method: "POST", json: payload },
    ),
  reconciliationIncidents: (options?: {
    portfolioSlug?: string;
    symbol?: string;
    incidentStatus?: string;
    includeResolved?: boolean;
    limit?: number;
  }) =>
    request<ReconciliationAcknowledgementResponse[]>("/reconciliation/incidents", {
      query: {
        portfolio_slug: options?.portfolioSlug,
        symbol: options?.symbol,
        incident_status: options?.incidentStatus,
        include_resolved: options?.includeResolved,
        limit: options?.limit,
      },
    }),
  updateReconciliationIncident: (payload: {
    portfolio_slug?: string;
    symbol: string;
    reason?: string;
    operator_note?: string;
    incident_status?: string;
    resolution_note?: string;
  }) =>
    request<{ acknowledged: boolean; symbol: string; audit_event_id: number }>(
      "/reconciliation/incidents/update",
      { method: "POST", json: payload },
    ),
  runStressTest: (payload: {
    portfolio_slug?: string;
    scenarios?: StressScenarioRequest[];
    alpha?: number;
  }) =>
    request<StressReportResponse>("/snapshots/stress", {
      method: "POST",
      json: payload,
    }),
};

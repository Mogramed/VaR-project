import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { Agent } from "undici";

export const runtime = "nodejs";

const upstreamAgent = new Agent({
  connect: { timeout: 10_000 },
  bodyTimeout: 0,
  headersTimeout: 0,
  keepAliveTimeout: 30_000,
  keepAliveMaxTimeout: 60_000,
});

function getBackendBaseUrl() {
  return (
    process.env.VAR_PROJECT_API_BASE_URL
    ?? process.env.NEXT_PUBLIC_API_BASE_URL
    ?? "http://api:8000"
  ).replace(/\/+$/, "");
}

function resolveUpstreamTimeoutMs(method: string, path: string[], isEventStreamRequest: boolean): number | undefined {
  if (isEventStreamRequest) {
    return undefined;
  }
  const normalizedMethod = method.toUpperCase();
  const normalizedPath = path.join("/").toLowerCase();

  if (
    normalizedPath.startsWith("operator/actions/")
    || normalizedPath.startsWith("operator/runs/")
  ) {
    return 20_000;
  }

  if (
    normalizedPath.startsWith("reports/")
    || normalizedPath.startsWith("backtests/")
    || normalizedPath.startsWith("snapshots/")
  ) {
    return 40_000;
  }

  return normalizedMethod === "GET" ? 20_000 : 25_000;
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractErrorCode(error: unknown): string | null {
  let cursor: unknown = error;
  const visited = new Set<unknown>();
  while (cursor && typeof cursor === "object" && !visited.has(cursor)) {
    visited.add(cursor);
    const payload = cursor as Record<string, unknown>;
    if (typeof payload.code === "string" && payload.code.trim()) {
      return payload.code;
    }
    cursor = payload.cause;
  }
  return null;
}

function shouldRetryStatus(status: number) {
  return status === 502 || status === 503 || status === 504;
}

function isLikelyTransientError(error: unknown): boolean {
  const errorCode = extractErrorCode(error);
  if (errorCode && (
    errorCode === "UND_ERR_HEADERS_TIMEOUT"
    || errorCode === "UND_ERR_BODY_TIMEOUT"
    || errorCode === "UND_ERR_CONNECT_TIMEOUT"
    || errorCode === "ECONNRESET"
    || errorCode === "ECONNREFUSED"
    || errorCode === "ENOTFOUND"
  )) {
    return true;
  }
  const message = error instanceof Error ? error.message.toLowerCase() : "";
  return (
    message.includes("timeout")
    || message.includes("timed out")
    || message.includes("aborted")
    || message.includes("socket")
    || message.includes("connect")
    || message.includes("terminated")
  );
}

async function fetchUpstreamWithRetry(
  target: string,
  init: RequestInit & { dispatcher?: Agent },
  timeoutMs: number | undefined,
  isEventStreamRequest: boolean,
): Promise<Response> {
  const method = String(init.method ?? "GET").toUpperCase();
  const retryableMethod = method === "GET" || method === "HEAD";
  const maxAttempts = retryableMethod && !isEventStreamRequest ? 2 : 1;

  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await fetch(target, {
        ...init,
        signal: timeoutMs == null ? undefined : AbortSignal.timeout(timeoutMs),
      });
      if (attempt < maxAttempts && shouldRetryStatus(response.status)) {
        await delay(200 * attempt);
        continue;
      }
      return response;
    } catch (error) {
      lastError = error;
      if (attempt >= maxAttempts || !isLikelyTransientError(error)) {
        break;
      }
      await delay(200 * attempt);
    }
  }

  throw lastError instanceof Error ? lastError : new Error("Upstream request failed.");
}

async function handle(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const target = `${getBackendBaseUrl()}/${path.join("/")}${request.nextUrl.search}`;
  const body =
    request.method === "GET" || request.method === "HEAD" ? undefined : await request.text();
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  const accept = request.headers.get("accept");
  if (contentType && body !== undefined) {
    headers.set("Content-Type", contentType);
  }
  if (accept) {
    headers.set("Accept", accept);
  }
  const requestId = request.headers.get("x-request-id");
  if (requestId) {
    headers.set("X-Request-ID", requestId);
  }
  const isEventStreamRequest = (accept ?? "").includes("text/event-stream");
  const upstreamTimeoutMs = resolveUpstreamTimeoutMs(request.method, path, isEventStreamRequest);

  let response: Response;
  try {
    response = await fetchUpstreamWithRetry(
      target,
      {
        method: request.method,
        headers,
        body,
        cache: "no-store",
        dispatcher: upstreamAgent,
      } as RequestInit & { dispatcher?: Agent },
      upstreamTimeoutMs,
      isEventStreamRequest,
    );
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Backend API is unreachable.";
    const bodyTimeoutCode = extractErrorCode(error);
    const isTimeout =
      bodyTimeoutCode === "UND_ERR_HEADERS_TIMEOUT"
      || bodyTimeoutCode === "UND_ERR_BODY_TIMEOUT"
      || bodyTimeoutCode === "UND_ERR_CONNECT_TIMEOUT"
      || errorMessage.toLowerCase().includes("timeout")
      || errorMessage.toLowerCase().includes("timed out")
      || errorMessage.toLowerCase().includes("aborted");

    return NextResponse.json(
      {
        detail: errorMessage,
        error_code: isTimeout ? "backend_timeout" : "backend_unreachable",
        hint: isTimeout
          ? "The upstream service took too long to respond. Please retry."
          : "Verify backend health and network reachability from the frontend container.",
      },
      { status: isTimeout ? 504 : 502 },
    );
  }

  const responseHeaders = new Headers();
  const responseContentType = response.headers.get("content-type");
  const cacheControl = response.headers.get("cache-control");
  const isEventStream = (responseContentType ?? "").includes("text/event-stream");
  const upstreamRequestId = response.headers.get("x-request-id");
  const upstreamRunId = response.headers.get("x-operator-run-id");
  if (responseContentType) {
    responseHeaders.set("Content-Type", responseContentType);
  }
  if (cacheControl && !isEventStream) {
    responseHeaders.set("Cache-Control", cacheControl);
  }
  if (upstreamRequestId) {
    responseHeaders.set("X-Request-ID", upstreamRequestId);
  }
  if (upstreamRunId) {
    responseHeaders.set("X-Operator-Run-ID", upstreamRunId);
  }
  if (isEventStream) {
    responseHeaders.set("Cache-Control", "no-cache, no-transform");
    responseHeaders.set("Connection", "keep-alive");
    responseHeaders.set("X-Accel-Buffering", "no");
  }

  if (!isEventStream) {
    const buffer = response.status === 204 ? null : await response.arrayBuffer();
    const text =
      buffer == null ? "" : new TextDecoder().decode(buffer);

    if (!response.ok) {
      let payload: unknown = {
        detail: text || response.statusText || "Upstream request failed.",
      };
      try {
        payload = text ? JSON.parse(text) : payload;
      } catch {
        // Keep the plain-text fallback payload when the upstream body is not JSON.
      }
      return NextResponse.json(payload, {
        status: response.status,
        headers: responseHeaders,
      });
    }

    return new NextResponse(buffer, {
      status: response.status,
      headers: responseHeaders,
    });
  }

  if (!response.ok) {
    const text = await response.text();
    let payload: unknown = {
      detail: text || response.statusText || "Upstream request failed.",
    };
    try {
      payload = text ? JSON.parse(text) : payload;
    } catch {
      // Keep the plain-text fallback payload when the upstream body is not JSON.
    }
    return NextResponse.json(payload, {
      status: response.status,
      headers: responseHeaders,
    });
  }

  return new NextResponse(response.body, {
    status: response.status,
    headers: responseHeaders,
  });
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return handle(request, context);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return handle(request, context);
}

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return handle(request, context);
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return handle(request, context);
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return handle(request, context);
}

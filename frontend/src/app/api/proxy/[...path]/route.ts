import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

function getBackendBaseUrl() {
  return (process.env.VAR_PROJECT_API_BASE_URL ?? "http://api:8000").replace(/\/+$/, "");
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

  const response = await fetch(target, {
    method: request.method,
    headers,
    body,
    cache: "no-store",
  });

  const responseHeaders = new Headers();
  const responseContentType = response.headers.get("content-type");
  const cacheControl = response.headers.get("cache-control");
  if (responseContentType) {
    responseHeaders.set("Content-Type", responseContentType);
  }
  if (cacheControl) {
    responseHeaders.set("Cache-Control", cacheControl);
  }
  if ((responseContentType ?? "").includes("text/event-stream")) {
    responseHeaders.set("Connection", "keep-alive");
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

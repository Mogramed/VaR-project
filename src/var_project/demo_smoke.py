from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import httpx


def _coerce_base_url(value: str) -> str:
    base = str(value or "").strip()
    if not base:
        return "http://127.0.0.1:8000"
    return base.rstrip("/")


def _request_json(
    client: httpx.Client,
    *,
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.get(path, params=params)
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "payload": None,
            "error": str(exc),
        }
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    payload: Any = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return {
        "ok": bool(response.status_code < 400),
        "status_code": int(response.status_code),
        "latency_ms": latency_ms,
        "payload": payload,
        "error": None if response.status_code < 400 else str(response.text or "").strip(),
    }


def _build_check(
    *,
    name: str,
    path: str,
    critical: bool,
    request_result: dict[str, Any],
    ok: bool,
    detail: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "critical": bool(critical),
        "ok": bool(ok),
        "status_code": request_result.get("status_code"),
        "latency_ms": request_result.get("latency_ms"),
        "detail": str(detail),
        "error": request_result.get("error"),
    }


def run_demo_smoke(
    *,
    base_url: str = "http://127.0.0.1:8000",
    portfolio_slug: str | None = None,
    timeout_seconds: float = 5.0,
    max_wait_ms: int = 1200,
    strict: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    normalized_base_url = _coerce_base_url(base_url)
    timeout = max(float(timeout_seconds), 1.0)
    readiness_params: dict[str, Any] = {
        "refresh_live": True,
        "max_wait_ms": max(int(max_wait_ms), 100),
    }
    live_params: dict[str, Any] = {"detail_level": "summary"}
    if portfolio_slug:
        readiness_params["portfolio_slug"] = str(portfolio_slug)
        live_params["portfolio_slug"] = str(portfolio_slug)

    checks: list[dict[str, Any]] = []
    readiness_status = "unknown"

    with httpx.Client(base_url=normalized_base_url, timeout=timeout, transport=transport) as client:
        root_result = _request_json(client, path="/")
        root_payload = root_result.get("payload") if isinstance(root_result.get("payload"), dict) else {}
        checks.append(
            _build_check(
                name="root",
                path="/",
                critical=True,
                request_result=root_result,
                ok=bool(root_result["ok"]) and root_payload.get("health") == "/health",
                detail="API discovery endpoint reachable.",
            )
        )

        health_result = _request_json(client, path="/health")
        health_payload = health_result.get("payload") if isinstance(health_result.get("payload"), dict) else {}
        checks.append(
            _build_check(
                name="health",
                path="/health",
                critical=True,
                request_result=health_result,
                ok=bool(health_result["ok"]) and str(health_payload.get("status") or "").lower() == "ok",
                detail="API health status should be ok.",
            )
        )

        dependencies_result = _request_json(client, path="/health/dependencies")
        dependencies_payload = (
            dependencies_result.get("payload")
            if isinstance(dependencies_result.get("payload"), dict)
            else {}
        )
        dependency_map = dict(dependencies_payload.get("dependencies") or {})
        db_dependency = dict(dependency_map.get("database") or {})
        checks.append(
            _build_check(
                name="dependencies",
                path="/health/dependencies",
                critical=True,
                request_result=dependencies_result,
                ok=bool(dependencies_result["ok"]) and bool(db_dependency.get("schema_ready", False)),
                detail="Database dependency should be reachable and migrated.",
            )
        )

        readiness_result = _request_json(client, path="/health/readiness", params=readiness_params)
        readiness_payload = readiness_result.get("payload") if isinstance(readiness_result.get("payload"), dict) else {}
        readiness_status = str(readiness_payload.get("status") or "unknown").strip().lower()
        checks.append(
            _build_check(
                name="readiness",
                path="/health/readiness",
                critical=True,
                request_result=readiness_result,
                ok=bool(readiness_result["ok"]) and readiness_status in {"ready", "degraded"},
                detail=str(readiness_payload.get("summary") or "Platform readiness check."),
            )
        )

        live_state_result = _request_json(client, path="/mt5/live/state", params=live_params)
        live_state_payload = (
            live_state_result.get("payload")
            if isinstance(live_state_result.get("payload"), dict)
            else {}
        )
        checks.append(
            _build_check(
                name="live_state",
                path="/mt5/live/state",
                critical=False,
                request_result=live_state_result,
                ok=bool(live_state_result["ok"])
                and str(live_state_payload.get("status") or "").strip().lower() in {"ok", "degraded"},
                detail="Live bridge state should be available (ok/degraded).",
            )
        )

        risk_summary_result = _request_json(client, path="/risk/summary", params={"portfolio_slug": portfolio_slug} if portfolio_slug else None)
        risk_summary_payload = (
            risk_summary_result.get("payload")
            if isinstance(risk_summary_result.get("payload"), dict)
            else {}
        )
        checks.append(
            _build_check(
                name="risk_summary",
                path="/risk/summary",
                critical=False,
                request_result=risk_summary_result,
                ok=bool(risk_summary_result["ok"]) and bool(risk_summary_payload),
                detail="Risk summary endpoint should return a payload.",
            )
        )

        capital_result = _request_json(client, path="/capital/latest", params={"portfolio_slug": portfolio_slug} if portfolio_slug else None)
        capital_payload = capital_result.get("payload") if isinstance(capital_result.get("payload"), dict) else {}
        checks.append(
            _build_check(
                name="capital",
                path="/capital/latest",
                critical=False,
                request_result=capital_result,
                ok=bool(capital_result["ok"]) and capital_payload.get("portfolio_slug") is not None,
                detail="Capital snapshot should be available.",
            )
        )

        exposure_result = _request_json(
            client,
            path="/portfolio/live-exposure",
            params={"portfolio_slug": portfolio_slug} if portfolio_slug else None,
        )
        exposure_payload = exposure_result.get("payload") if isinstance(exposure_result.get("payload"), dict) else {}
        checks.append(
            _build_check(
                name="live_exposure",
                path="/portfolio/live-exposure",
                critical=False,
                request_result=exposure_result,
                ok=bool(exposure_result["ok"]) and isinstance(exposure_payload.get("items"), list),
                detail="Live exposure endpoint should return instrument items.",
            )
        )

    failed_critical = [item["name"] for item in checks if item["critical"] and not item["ok"]]
    failed_optional = [item["name"] for item in checks if (not item["critical"]) and not item["ok"]]
    ok = len(failed_critical) == 0 and (len(failed_optional) == 0 or not bool(strict))

    status = "ok"
    if failed_critical:
        status = "failed"
    elif failed_optional:
        status = "degraded"

    return {
        "ok": ok,
        "status": status,
        "strict": bool(strict),
        "base_url": normalized_base_url,
        "portfolio_slug": portfolio_slug,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "readiness_status": readiness_status,
        "failed_critical": failed_critical,
        "failed_optional": failed_optional,
        "checks": checks,
    }


def format_demo_smoke_report(result: dict[str, Any]) -> list[str]:
    header = (
        f"[demo-smoke] {str(result.get('status') or 'unknown').upper()} "
        f"(base_url={result.get('base_url')}, readiness={result.get('readiness_status')}, strict={bool(result.get('strict', False))})"
    )
    lines = [header]
    for check in list(result.get("checks") or []):
        marker = "OK" if bool(check.get("ok")) else "FAIL"
        level = "critical" if bool(check.get("critical")) else "optional"
        status_code = check.get("status_code")
        code_text = "n/a" if status_code is None else str(status_code)
        latency = check.get("latency_ms")
        latency_text = "n/a" if latency is None else f"{float(latency):.1f}ms"
        detail = str(check.get("detail") or "")
        error = check.get("error")
        suffix = f" ({error})" if error else ""
        lines.append(f" - [{marker}] {check.get('name')} [{level}] HTTP {code_text} in {latency_text}: {detail}{suffix}")
    return lines

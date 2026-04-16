from __future__ import annotations

import httpx

from var_project.demo_smoke import run_demo_smoke


def _healthy_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/":
        return httpx.Response(200, json={"status": "ok", "health": "/health", "docs": "/docs"})
    if path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if path == "/health/dependencies":
        return httpx.Response(
            200,
            json={"dependencies": {"database": {"schema_ready": True}, "mt5_live": {"reachable": True}}},
        )
    if path == "/health/readiness":
        return httpx.Response(200, json={"status": "degraded", "summary": "demo continuity"})
    if path == "/mt5/live/state":
        return httpx.Response(200, json={"status": "degraded", "generated_at": "2026-04-10T10:00:00+00:00"})
    if path == "/risk/summary":
        return httpx.Response(200, json={"source": "historical", "var": {"hist": 12.0}})
    if path == "/capital/latest":
        return httpx.Response(200, json={"portfolio_slug": "fx_eur_20k", "total_capital_budget_eur": 1000.0})
    if path == "/portfolio/live-exposure":
        return httpx.Response(200, json={"items": []})
    return httpx.Response(404, json={"detail": f"Unhandled path {path}"})


def test_demo_smoke_passes_when_core_checks_are_ready_or_degraded() -> None:
    result = run_demo_smoke(
        base_url="http://desk.local",
        transport=httpx.MockTransport(_healthy_handler),
    )

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["failed_critical"] == []
    assert result["failed_optional"] == []


def test_demo_smoke_fails_when_readiness_is_not_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health/readiness":
            return httpx.Response(200, json={"status": "not_ready", "summary": "mt5 missing"})
        return _healthy_handler(request)

    result = run_demo_smoke(
        base_url="http://desk.local",
        transport=httpx.MockTransport(handler),
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert "readiness" in result["failed_critical"]


def test_demo_smoke_marks_optional_failures_as_degraded_in_non_strict_mode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/risk/summary":
            return httpx.Response(503, json={"detail": "temporary unavailable"})
        return _healthy_handler(request)

    result = run_demo_smoke(
        base_url="http://desk.local",
        strict=False,
        transport=httpx.MockTransport(handler),
    )

    assert result["ok"] is True
    assert result["status"] == "degraded"
    assert "risk_summary" in result["failed_optional"]

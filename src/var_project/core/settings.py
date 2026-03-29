from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from var_project.core.types import MT5Config
from var_project.portfolio.holdings import configured_holdings
from var_project.storage import slugify_label


def find_repo_root(start: Path | None = None) -> Path:
    candidates = []
    env_root = os.getenv("VAR_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).resolve())
    app_root = Path("/app")
    if app_root.exists():
        candidates.append(app_root.resolve())
    if start is not None:
        candidates.append(start.resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve()])

    for candidate in candidates:
        for path in [candidate] + list(candidate.parents):
            if (path / "pyproject.toml").exists() and (path / "config" / "settings.yaml").exists():
                return path
    raise RuntimeError("Repo root not found (pyproject.toml + config/settings.yaml).")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_settings(root: Path) -> dict[str, Any]:
    return load_yaml(root / "config" / "settings.yaml")


def load_risk_limits(root: Path) -> dict[str, Any]:
    path = root / "config" / "risk_limits.yaml"
    if not path.exists():
        return {}
    return load_yaml(path)


def get_symbols(raw_cfg: Mapping[str, Any]) -> list[str]:
    return [str(symbol) for symbol in raw_cfg.get("symbols") or []]


def get_positions(raw_cfg: Mapping[str, Any], symbols: list[str]) -> dict[str, float]:
    positions = ((raw_cfg.get("portfolio") or {}).get("positions_eur")) or {}
    if not positions:
        return {symbol: 10_000.0 for symbol in symbols}
    return {symbol: float(positions.get(symbol, 0.0)) for symbol in symbols}


def get_data_defaults(raw_cfg: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(raw_cfg.get("data") or {})
    timeframes = data.get("timeframes") or []
    if isinstance(timeframes, str):
        timeframes = [timeframes]
    days_list = data.get("history_days_list") or []
    if isinstance(days_list, int):
        days_list = [days_list]
    return {
        "timeframes": [str(item) for item in timeframes],
        "history_days_list": [int(item) for item in days_list],
        "min_coverage": float(data.get("min_coverage", 0.90)),
        "storage_format": str(data.get("storage_format", "csv")),
    }


def get_risk_defaults(raw_cfg: Mapping[str, Any]) -> dict[str, Any]:
    risk = dict(raw_cfg.get("risk") or {})
    ewma = dict(risk.get("ewma") or {})
    fhs = dict(risk.get("fhs") or {})
    mc = dict(risk.get("mc") or {})
    garch = dict(risk.get("garch") or {})
    return {
        "alpha": float(risk.get("alpha", 0.95)),
        "window": int(risk.get("window", 250)),
        "ewma_lambda": float(ewma.get("lambda", 0.94)),
        "fhs_lambda": float(fhs.get("lambda", 0.94)),
        "mc": {
            "n_sims": int(mc.get("n_sims", 20_000)),
            "dist": str(mc.get("dist", "normal")),
            "df_t": int(mc.get("df_t", 6)),
            "seed": None if mc.get("seed") is None else int(mc.get("seed")),
        },
        "garch": {
            "enabled": bool(garch.get("enabled", True)),
            "p": int(garch.get("p", 1)),
            "q": int(garch.get("q", 1)),
            "dist": str(garch.get("dist", "normal")),
            "mean": str(garch.get("mean", "constant")),
        },
    }


def get_mt5_config(raw_cfg: Mapping[str, Any]) -> MT5Config:
    mt5_cfg = dict(raw_cfg.get("mt5") or {})
    login = os.getenv("VAR_PROJECT_MT5_LOGIN", mt5_cfg.get("login"))
    password = os.getenv("VAR_PROJECT_MT5_PASSWORD", mt5_cfg.get("password"))
    server = os.getenv("VAR_PROJECT_MT5_SERVER", mt5_cfg.get("server"))
    path = os.getenv("VAR_PROJECT_MT5_PATH", mt5_cfg.get("path"))
    timeout_ms = os.getenv("VAR_PROJECT_MT5_TIMEOUT_MS", mt5_cfg.get("timeout_ms"))
    agent_base_url = os.getenv("VAR_PROJECT_MT5_AGENT_BASE_URL", mt5_cfg.get("agent_base_url"))
    agent_api_key = os.getenv("VAR_PROJECT_MT5_AGENT_API_KEY", mt5_cfg.get("agent_api_key"))
    execution_enabled = os.getenv("VAR_PROJECT_MT5_EXECUTION_ENABLED", mt5_cfg.get("execution_enabled"))
    magic = os.getenv("VAR_PROJECT_MT5_MAGIC", mt5_cfg.get("magic"))
    deviation_points = os.getenv("VAR_PROJECT_MT5_DEVIATION_POINTS", mt5_cfg.get("deviation_points"))
    comment_prefix = os.getenv("VAR_PROJECT_MT5_COMMENT_PREFIX", mt5_cfg.get("comment_prefix"))
    portable = os.getenv("VAR_PROJECT_MT5_PORTABLE", mt5_cfg.get("portable"))
    live_enabled = os.getenv("VAR_PROJECT_MT5_LIVE_ENABLED", mt5_cfg.get("live_enabled"))
    live_poll_seconds = os.getenv("VAR_PROJECT_MT5_LIVE_POLL_SECONDS", mt5_cfg.get("live_poll_seconds"))
    live_history_poll_seconds = os.getenv(
        "VAR_PROJECT_MT5_LIVE_HISTORY_POLL_SECONDS",
        mt5_cfg.get("live_history_poll_seconds"),
    )
    live_history_lookback_minutes = os.getenv(
        "VAR_PROJECT_MT5_LIVE_HISTORY_LOOKBACK_MINUTES",
        mt5_cfg.get("live_history_lookback_minutes"),
    )
    live_event_buffer_size = os.getenv(
        "VAR_PROJECT_MT5_LIVE_EVENT_BUFFER_SIZE",
        mt5_cfg.get("live_event_buffer_size"),
    )
    live_stale_after_seconds = os.getenv(
        "VAR_PROJECT_MT5_LIVE_STALE_AFTER_SECONDS",
        mt5_cfg.get("live_stale_after_seconds"),
    )

    return MT5Config(
        login=None if login in {None, "", "null"} else int(login),
        password=None if password in {None, "", "null"} else str(password),
        server=None if server in {None, "", "null"} else str(server),
        path=None if path in {None, "", "null"} else str(path),
        timeout_ms=None if timeout_ms in {None, "", "null"} else int(timeout_ms),
        portable=str(portable).lower() in {"1", "true", "yes", "on"},
        agent_base_url=None if agent_base_url in {None, "", "null"} else str(agent_base_url).rstrip("/"),
        agent_api_key=None if agent_api_key in {None, "", "null"} else str(agent_api_key),
        execution_enabled=str(execution_enabled).lower() in {"1", "true", "yes", "on"},
        magic=420001 if magic in {None, "", "null"} else int(magic),
        deviation_points=20 if deviation_points in {None, "", "null"} else int(deviation_points),
        comment_prefix="var_risk_desk" if comment_prefix in {None, "", "null"} else str(comment_prefix),
        live_enabled=True if live_enabled in {None, "", "null"} else str(live_enabled).lower() in {"1", "true", "yes", "on"},
        live_poll_seconds=2.0 if live_poll_seconds in {None, "", "null"} else float(live_poll_seconds),
        live_history_poll_seconds=30.0
        if live_history_poll_seconds in {None, "", "null"}
        else float(live_history_poll_seconds),
        live_history_lookback_minutes=180
        if live_history_lookback_minutes in {None, "", "null"}
        else int(live_history_lookback_minutes),
        live_event_buffer_size=500
        if live_event_buffer_size in {None, "", "null"}
        else int(live_event_buffer_size),
        live_stale_after_seconds=6.0
        if live_stale_after_seconds in {None, "", "null"}
        else float(live_stale_after_seconds),
    )


def get_portfolio_context(raw_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return get_portfolio_contexts(raw_cfg)[0]


def _build_portfolio_context(
    raw_cfg: Mapping[str, Any],
    portfolio_cfg: Mapping[str, Any] | None,
    *,
    default_name: str | None = None,
) -> dict[str, Any]:
    mt5_cfg = dict(raw_cfg.get("mt5") or {})
    root_symbols = get_symbols(raw_cfg)
    base_currency = str((portfolio_cfg or {}).get("base_currency") or raw_cfg.get("base_currency", "EUR"))
    portfolio_symbols = [str(symbol) for symbol in ((portfolio_cfg or {}).get("symbols") or root_symbols)]
    portfolio_name = str((portfolio_cfg or {}).get("name") or default_name or f"{base_currency}_{'_'.join(portfolio_symbols)}")
    configured_positions = dict((portfolio_cfg or {}).get("positions_eur") or {})
    portfolio_mode = str(
        (portfolio_cfg or {}).get("mode")
        or (portfolio_cfg or {}).get("portfolio_mode")
        or (
            "hybrid"
            if any(mt5_cfg.get(key) for key in ("agent_base_url", "path", "login", "server"))
            else "offline_fixture"
        )
    ).strip().lower()
    if configured_positions:
        positions = {symbol: float(configured_positions.get(symbol, 0.0)) for symbol in portfolio_symbols}
    else:
        positions = get_positions(raw_cfg, portfolio_symbols)
    configured = configured_holdings(
        portfolio_symbols,
        positions,
        base_currency=base_currency,
        source="configured",
    )
    return {
        "name": portfolio_name,
        "slug": slugify_label(portfolio_name),
        "base_currency": base_currency,
        "mode": portfolio_mode,
        "symbols": portfolio_symbols,
        "watchlist_symbols": list(portfolio_symbols),
        "positions": positions,
        "configured_holdings": [holding.to_dict() for holding in configured],
    }


def get_portfolio_contexts(raw_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    portfolios_cfg = raw_cfg.get("portfolios") or []
    if isinstance(portfolios_cfg, list) and portfolios_cfg:
        contexts = [
            _build_portfolio_context(
                raw_cfg,
                dict(item or {}),
                default_name=f"portfolio_{index + 1}",
            )
            for index, item in enumerate(portfolios_cfg)
        ]
        return [context for context in contexts if context["symbols"]]

    portfolio_cfg = dict(raw_cfg.get("portfolio") or {})
    return [_build_portfolio_context(raw_cfg, portfolio_cfg)]


def get_desk_context(raw_cfg: Mapping[str, Any]) -> dict[str, Any]:
    portfolios = get_portfolio_contexts(raw_cfg)
    desk_cfg = dict(raw_cfg.get("desk") or {})
    base_currency = str(desk_cfg.get("base_currency") or raw_cfg.get("base_currency", "EUR"))
    desk_name = str(desk_cfg.get("name") or "FX Risk Desk")
    desk_slug = str(desk_cfg.get("slug") or slugify_label(desk_name))
    return {
        "name": desk_name,
        "slug": desk_slug,
        "base_currency": base_currency,
        "portfolio_slugs": [portfolio["slug"] for portfolio in portfolios],
    }

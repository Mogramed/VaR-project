from __future__ import annotations

from pathlib import Path
from typing import Any

from var_project.api.service import DeskApiService
from var_project.storage import upgrade_database


def seed_demo_environment(root: Path, *, portfolio_slug: str | None = None) -> dict[str, Any]:
    repo_root = root.resolve()
    upgrade_database(repo_root)
    service = DeskApiService(repo_root)

    selected = (
        [service.runtime._resolve_portfolio_context(portfolio_slug)]
        if portfolio_slug is not None
        else [dict(portfolio) for portfolio in service.portfolios]
    )

    seeded: list[dict[str, Any]] = []
    for portfolio in selected:
        slug = str(portfolio["slug"])
        symbols = [str(symbol) for symbol in portfolio["symbols"]]
        primary_symbol = symbols[0] if symbols else None
        secondary_symbol = symbols[-1] if symbols else None

        primary_position = abs(float((portfolio["positions"] or {}).get(primary_symbol or "", 0.0)))
        decision_delta = max(500.0, round(primary_position * 0.10, 2)) if primary_symbol else 0.0

        snapshot = service.run_snapshot(portfolio_slug=slug)
        backtest = service.run_backtest(portfolio_slug=slug)
        validation = service.run_validation(compare_path=backtest["compare_csv"])
        decision = (
            service.evaluate_trade_decision(
                portfolio_slug=slug,
                symbol=primary_symbol,
                delta_position_eur=decision_delta,
                note="demo bootstrap decision",
            )
            if primary_symbol
            else None
        )
        preview = None
        if (secondary_symbol or primary_symbol) and service.runtime.market_data.should_use_mt5_market_data(portfolio):
            preview = service.preview_execution(
                portfolio_slug=slug,
                symbol=secondary_symbol or primary_symbol,
                delta_position_eur=-decision_delta if secondary_symbol else decision_delta,
                note="demo bootstrap execution preview",
            )
        report = service.run_report(compare_path=backtest["compare_csv"], portfolio_slug=slug)

        seeded.append(
            {
                "portfolio_slug": slug,
                "snapshot_id": snapshot["snapshot_id"],
                "backtest_run_id": backtest["backtest_run_id"],
                "validation_run_id": validation["validation_run_id"],
                "decision_id": None if decision is None else decision.get("id"),
                "execution_preview_symbol": None if preview is None else preview.get("symbol"),
                "report_markdown": report["report_markdown"],
            }
        )

    return {
        "database_url": service.storage.settings.database_url,
        "portfolio_count": len(seeded),
        "seeded": seeded,
    }

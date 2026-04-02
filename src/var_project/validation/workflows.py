from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from var_project.storage import AppStorage, slugify_label
from var_project.validation.model_validation import ValidationSummary, validate_compare_frame, validate_compare_surface


def infer_portfolio_metadata(frame: pd.DataFrame) -> tuple[str, str, list[str], dict[str, float], str]:
    if frame.empty:
        return "portfolio", "EUR", [], {}, "portfolio"

    first = frame.iloc[0]
    name = str(first.get("portfolio") or "portfolio")
    base_currency = str(first.get("base_currency") or "EUR")

    symbols_value = first.get("symbols")
    if isinstance(symbols_value, str) and symbols_value.strip():
        symbols = [item for item in symbols_value.split(",") if item]
    else:
        symbols = [str(col).replace("ret_", "") for col in frame.columns if str(col).startswith("ret_")]

    exposure_raw = first.get("exposure_by_symbol_json") or first.get("positions_eur_json")
    if isinstance(exposure_raw, str) and exposure_raw.strip():
        exposure_by_symbol = {str(key): float(value) for key, value in json.loads(exposure_raw).items()}
    else:
        exposure_by_symbol = {}

    return name, base_currency, symbols, exposure_by_symbol, slugify_label(name)


def persist_validation_summary(
    *,
    storage: AppStorage,
    compare_csv: Path,
    alpha: float,
    alphas: Sequence[float] | None = None,
    horizons: Sequence[int] | None = None,
    source_artifact_id: int | None = None,
    portfolio_id: int | None = None,
    portfolio_name: str | None = None,
    portfolio_slug: str | None = None,
    base_currency: str | None = None,
    symbols: Sequence[str] | None = None,
    positions: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    from var_project.alerts.engine import alerts_from_validation_summary

    frame = pd.read_csv(compare_csv)
    summary = validate_compare_frame(frame, alpha)
    surface = validate_compare_surface(frame, alphas=list(alphas or [alpha]), horizons=list(horizons or [1]))
    summary = ValidationSummary(
        alpha=summary.alpha,
        expected_rate=summary.expected_rate,
        model_results=summary.model_results,
        best_model=summary.best_model,
        champion_model_live=surface.get("champion_model_live"),
        champion_model_reporting=surface.get("champion_model_reporting"),
        surface=surface,
    )

    inferred_name, inferred_currency, inferred_symbols, inferred_positions, inferred_slug = infer_portfolio_metadata(frame)
    resolved_name = str(portfolio_name or inferred_name)
    resolved_currency = str(base_currency or inferred_currency)
    resolved_symbols = [str(item) for item in (symbols or inferred_symbols)]
    resolved_positions = {
        str(symbol): float(value) for symbol, value in dict(positions or inferred_positions).items()
    }
    resolved_slug = str(portfolio_slug or inferred_slug)

    if portfolio_id is None:
        portfolio_id = storage.upsert_portfolio(
            name=resolved_name,
            base_currency=resolved_currency,
            symbols=resolved_symbols,
            positions=resolved_positions,
            slug=resolved_slug,
        )

    artifact_id = source_artifact_id
    if artifact_id is None:
        artifact_id = storage.register_artifact(
            compare_csv,
            artifact_type="backtest_compare",
            details={"alpha": float(alpha), "rows": int(len(frame))},
        )

    validation_run_id = storage.record_validation_run(
        summary,
        portfolio_id=portfolio_id,
        source_artifact_id=artifact_id,
    )
    validation_path = compare_csv.with_name(f"{compare_csv.stem}_validation.json")
    validation_artifact_id = storage.write_json_artifact(
        summary,
        validation_path,
        artifact_type="validation_summary",
        details={
            "portfolio": resolved_name,
            "portfolio_slug": resolved_slug,
            "alpha": float(alpha),
            "best_model": summary.best_model,
            "validation_run_id": validation_run_id,
        },
    )
    alerts = alerts_from_validation_summary(summary)
    if alerts:
        storage.record_alerts(alerts, portfolio_id=portfolio_id, validation_run_id=validation_run_id)

    return {
        "portfolio_id": int(portfolio_id),
        "portfolio_slug": resolved_slug,
        "validation_run_id": validation_run_id,
        "validation_artifact_id": validation_artifact_id,
        "validation_json": str(validation_path.resolve()),
        "best_model": summary.best_model,
        "alert_count": len(alerts),
        "summary": summary.to_dict(),
    }

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _positive_ints(values: Sequence[Any] | None) -> list[int]:
    items: list[int] = []
    for value in values or []:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            continue
        if normalized > 0:
            items.append(normalized)
    return items


def validate_backtest_history_compatibility(
    data_defaults: Mapping[str, Any],
    risk_defaults: Mapping[str, Any],
    *,
    days: int | None = None,
    window: int | None = None,
    horizons: Sequence[int] | None = None,
    context: str = "backtest",
) -> None:
    selected_window = int(window or risk_defaults.get("window") or 0)
    selected_horizons = _positive_ints(horizons or risk_defaults.get("horizons"))
    history_days_list = _positive_ints(data_defaults.get("history_days_list"))
    configured_history_days = max(history_days_list, default=0)
    market_history_days = max(_positive_ints([data_defaults.get("market_history_days")]), default=0)
    tracked_history_days = max(configured_history_days, market_history_days)
    selected_days = int(days or max(tracked_history_days, configured_history_days, 0))

    if selected_window <= 0:
        raise ValueError(f"{context} requires a strictly positive risk window.")

    if not selected_horizons:
        raise ValueError(f"{context} requires at least one positive risk horizon.")

    max_horizon = max(selected_horizons)
    required_days = selected_window + max_horizon
    positive_days = [value for value in (selected_days, tracked_history_days) if value > 0]
    if not positive_days:
        raise ValueError(
            f"{context} requires a positive tracked history budget via data.history_days_list, "
            "data.market_history_days or an explicit days override."
        )
    available_days = min(positive_days)

    if available_days >= required_days:
        return

    raise ValueError(
        f"{context} configuration is incompatible with the tracked history: "
        f"days={selected_days}, window={selected_window}, max_horizon={max_horizon} "
        f"require at least {required_days} days, but tracked history only provides up to "
        f"{tracked_history_days} days (history_days_list={configured_history_days}, market_history_days={market_history_days}). "
        "Reduce risk.window / risk.horizons or extend the tracked fixtures."
    )

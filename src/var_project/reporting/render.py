from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from var_project.alerts.engine import alerts_from_live_snapshot, alerts_from_validation_summary
from var_project.reporting.charts import _plot_exceptions, _plot_pnl_vs_var
from var_project.reporting.metrics import _count_exceptions, _infer_models, _parse_alpha_from_name
from var_project.validation.model_validation import validate_compare_frame


def _latest_file(dir_path: Path, pattern: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def _load_limits(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_latest_snapshot(snapshot_dir: Path) -> Optional[Dict]:
    snap_path = _latest_file(snapshot_dir, "live_snapshot_*.json")
    if snap_path is None:
        return None
    try:
        return json.loads(snap_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _snapshot_payload(snapshot: Optional[Dict]) -> Dict:
    if snapshot is None:
        return {}
    return dict(snapshot.get("payload") or snapshot)


def _snapshot_source(snapshot: Optional[Dict]) -> str:
    if snapshot is None:
        return "n/a"
    payload = _snapshot_payload(snapshot)
    return str(snapshot.get("source") or payload.get("source") or "unknown")


def _snapshot_timestamp(snapshot: Optional[Dict]) -> str:
    if snapshot is None:
        return "n/a"
    payload = _snapshot_payload(snapshot)
    return str(
        snapshot.get("created_at")
        or payload.get("snapshot_timestamp")
        or payload.get("time_utc")
        or "n/a"
    )


def _fmt_money(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "n/a"


def _fmt_number(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "n/a"


def render_daily_markdown(
    compare_csv: Path,
    out_dir: Path,
    snapshot: Optional[Dict] = None,
    snapshot_dir: Optional[Path] = None,
    risk_limits_yaml: Optional[Path] = None,
    report_label: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(compare_csv)
    models = _infer_models(df)
    alpha = _parse_alpha_from_name(compare_csv.stem)

    n = len(df)
    start = str(df["date"].iloc[0]) if "date" in df.columns and n > 0 else "n/a"
    end = str(df["date"].iloc[-1]) if "date" in df.columns and n > 0 else "n/a"

    exceptions = {model: _count_exceptions(df, model) for model in models}
    validation = validate_compare_frame(df, alpha if alpha is not None else 0.95)

    limits = _load_limits(risk_limits_yaml) if risk_limits_yaml else {}
    traffic_light_limits = limits.get("backtest_traffic_light_99_250") or {}
    green_max = int(traffic_light_limits.get("green_max", 4))
    yellow_max = int(traffic_light_limits.get("yellow_max", 9))

    selected_snapshot = snapshot if snapshot is not None else (_load_latest_snapshot(snapshot_dir) if snapshot_dir else None)
    snapshot_payload = _snapshot_payload(selected_snapshot)
    snapshot_source = _snapshot_source(selected_snapshot)
    snapshot_timestamp = _snapshot_timestamp(selected_snapshot)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    exc_chart = out_dir / f"{compare_csv.stem}_exceptions.png"
    pnl_chart = out_dir / f"{compare_csv.stem}_pnl_var.png"

    try:
        _plot_exceptions(df, exc_chart)
    except Exception:
        exc_chart = None

    plot_model = "hist" if "hist" in models else (models[0] if models else None)
    if plot_model:
        try:
            _plot_pnl_vs_var(df, pnl_chart, model=plot_model)
        except Exception:
            pnl_chart = None
    else:
        pnl_chart = None

    validation_alerts = alerts_from_validation_summary(validation)
    snapshot_alerts = (
        alerts_from_live_snapshot(snapshot_payload, limits)
        if selected_snapshot is not None
        else []
    )

    md_path = out_dir / f"{compare_csv.stem}.md"
    lines: List[str] = []

    lines.append(f"# Risk Report - {report_label or compare_csv.stem}")
    lines.append("")
    lines.append(f"- Generated (UTC): **{now}**")
    lines.append(f"- Compare CSV: **{compare_csv.name}**")
    lines.append(f"- Preferred snapshot source: **{snapshot_source}**")
    if alpha is not None:
        lines.append(f"- Alpha: **{alpha:.2f}** (q={1 - alpha:.2f})")
    lines.append(f"- Sample (aligned days): **{n}**")
    lines.append(f"- Range: **{start} -> {end}**")
    lines.append("")

    lines.append("## Portfolio Snapshot")
    if selected_snapshot is None:
        lines.append("_No portfolio snapshot is available for this report._")
    else:
        capital_usage = dict(snapshot_payload.get("capital_usage") or {})
        lines.append(f"- Snapshot time (UTC): **{snapshot_timestamp}**")
        lines.append(f"- Snapshot source: **{snapshot_source}**")
        if capital_usage:
            lines.append(f"- Capital status: **{capital_usage.get('status') or 'n/a'}**")
            lines.append(
                f"- Remaining capital: **{_fmt_money(capital_usage.get('total_capital_remaining_eur'))} "
                f"{capital_usage.get('base_currency') or 'EUR'}**"
            )
        lines.append("")

        holdings = list(snapshot_payload.get("holdings") or [])
        if holdings:
            lines.append("### Holdings")
            lines.append("| Symbol | Asset class | Side | Lots | Signed exposure | Exposure (base) | Unrealized PnL |")
            lines.append("|---|---|---|---:|---:|---:|---:|")
            for item in holdings:
                lines.append(
                    "| "
                    f"{item.get('symbol') or 'n/a'} | "
                    f"{item.get('asset_class') or 'n/a'} | "
                    f"{item.get('side') or 'n/a'} | "
                    f"{_fmt_number(item.get('volume_lots'))} | "
                    f"{_fmt_money(item.get('signed_exposure_base_ccy', item.get('signed_position_eur')))} | "
                    f"{_fmt_money(item.get('exposure_base_ccy'))} | "
                    f"{_fmt_money(item.get('unrealized_pnl_base_ccy'))} |"
                )
            lines.append("")
        else:
            exposure = snapshot_payload.get("exposure_by_symbol", {}) or snapshot_payload.get("positions_eur", {}) or {}
            if exposure:
                lines.append("### Exposure by Symbol")
                lines.append("| Symbol | Exposure |")
                lines.append("|---|---:|")
                for symbol, value in exposure.items():
                    lines.append(f"| {symbol} | {_fmt_money(value)} |")
                lines.append("")

        vars_map = snapshot_payload.get("var", {}) or {}
        es_map = snapshot_payload.get("es", {}) or {}
        headline_risk = list(snapshot_payload.get("headline_risk") or [])
        stress_surface = dict(snapshot_payload.get("stress_surface") or {})
        attribution = dict(snapshot_payload.get("attribution") or {})
        risk_nowcast = dict(snapshot_payload.get("risk_nowcast") or {})
        microstructure = dict(snapshot_payload.get("microstructure") or {})
        tick_quality = dict(snapshot_payload.get("tick_quality") or {})
        pnl_explain = dict(snapshot_payload.get("pnl_explain") or {})
        if vars_map:
            lines.append("### VaR / ES")
            lines.append("| Model | VaR | ES |")
            lines.append("|---|---:|---:|")
            model_names = sorted(set(list(vars_map.keys()) + list(es_map.keys())))
            for model_name in model_names:
                lines.append(
                    f"| {model_name} | {_fmt_money(vars_map.get(model_name))} | {_fmt_money(es_map.get(model_name))} |"
                )
            lines.append("")

        if headline_risk:
            lines.append("### Headline Risk Surface")
            lines.append("| View | Model | Horizon | Confidence | VaR | ES | Status |")
            lines.append("|---|---|---:|---:|---:|---:|---|")
            for item in headline_risk:
                lines.append(
                    f"| {item.get('label') or item.get('key') or 'n/a'} | "
                    f"{str(item.get('model') or 'n/a').upper()} | "
                    f"{item.get('horizon_days') or 'n/a'}d | "
                    f"{_fmt_number(item.get('alpha'), 2)} | "
                    f"{_fmt_money(item.get('var'))} | "
                    f"{_fmt_money(item.get('es'))} | "
                    f"{item.get('status') or 'n/a'} |"
                )
            lines.append("")

        if attribution:
            models_payload = dict(attribution.get("models") or {})
            preferred_model = None
            if headline_risk:
                preferred_model = str((headline_risk[0] or {}).get("model") or "").lower() or None
            if preferred_model not in models_payload:
                preferred_model = next(iter(models_payload.keys()), None)
            model_payload = dict(models_payload.get(preferred_model or "", {}) or {})
            position_rows = sorted(
                list(dict(model_payload.get("positions") or {}).values()),
                key=lambda item: abs(float(item.get("component_es") or item.get("component_var") or 0.0)),
                reverse=True,
            )
            asset_class_rows = sorted(
                list(dict(model_payload.get("asset_classes") or {}).values()),
                key=lambda item: abs(float(item.get("component_es") or item.get("component_var") or 0.0)),
                reverse=True,
            )
            if position_rows or asset_class_rows:
                lines.append("### Risk Contributions")
                if preferred_model:
                    lines.append(f"- Attribution model: **{str(preferred_model).upper()}**")
                if position_rows:
                    lines.append("| Symbol | Asset class | Exposure | cVaR | cES | iVaR | iES | Contrib ES |")
                    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
                    for item in position_rows[:5]:
                        lines.append(
                            f"| {item.get('symbol') or 'n/a'} | "
                            f"{item.get('asset_class') or 'n/a'} | "
                            f"{_fmt_money(item.get('exposure_base_ccy'))} | "
                            f"{_fmt_money(item.get('component_var'))} | "
                            f"{_fmt_money(item.get('component_es'))} | "
                            f"{_fmt_money(item.get('incremental_var'))} | "
                            f"{_fmt_money(item.get('incremental_es'))} | "
                            f"{_fmt_number(None if item.get('contribution_pct_es') is None else float(item.get('contribution_pct_es')) * 100.0, 1)}% |"
                        )
                    lines.append("")
                if asset_class_rows:
                    lines.append("| Asset class | Symbols | Exposure | cVaR | cES | iVaR | iES | Contrib ES |")
                    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
                    for item in asset_class_rows:
                        lines.append(
                            f"| {item.get('asset_class') or 'n/a'} | "
                            f"{item.get('symbol_count') or 0} | "
                            f"{_fmt_money(item.get('exposure_base_ccy'))} | "
                            f"{_fmt_money(item.get('component_var'))} | "
                            f"{_fmt_money(item.get('component_es'))} | "
                            f"{_fmt_money(item.get('incremental_var'))} | "
                            f"{_fmt_money(item.get('incremental_es'))} | "
                            f"{_fmt_number(None if item.get('contribution_pct_es') is None else float(item.get('contribution_pct_es')) * 100.0, 1)}% |"
                        )
                    lines.append("")

        if stress_surface:
            lines.append("### Stress Surface")
            historical_extremes = list(stress_surface.get("historical_extremes") or [])
            scenarios = list(stress_surface.get("scenarios") or [])
            if scenarios:
                lines.append("| Scenario | VaR | ES |")
                lines.append("|---|---:|---:|")
                for scenario in scenarios:
                    primary = dict(scenario.get("primary_metric") or {})
                    lines.append(
                        f"| {scenario.get('name') or 'n/a'} | "
                        f"{_fmt_money(primary.get('var', scenario.get('var')))} | "
                        f"{_fmt_money(primary.get('es', scenario.get('es')))} |"
                    )
                lines.append("")
            if historical_extremes:
                lines.append("| Historical window | Worst loss | Tail mean loss | End date |")
                lines.append("|---|---:|---:|---|")
                for item in historical_extremes:
                    lines.append(
                        f"| {item.get('horizon_days') or 'n/a'}d | "
                        f"{_fmt_money(item.get('worst_loss'))} | "
                        f"{_fmt_money(item.get('tail_mean_loss'))} | "
                        f"{item.get('worst_end_date') or 'n/a'} |"
                    )
                lines.append("")

        if risk_nowcast:
            lines.append("### Live Risk Nowcast")
            live_nowcast = dict(risk_nowcast.get("live_1d_99") or {})
            governance_nowcast = dict(risk_nowcast.get("governance_10d_99") or {})
            lines.append(f"- Regime: **{risk_nowcast.get('regime') or 'n/a'}**")
            lines.append(f"- Scale factor: **{_fmt_number(risk_nowcast.get('scale_factor'), 2)}**")
            if live_nowcast:
                lines.append(
                    f"- 1D 99% nowcast: VaR **{_fmt_money(live_nowcast.get('nowcast_var'))}**, "
                    f"ES **{_fmt_money(live_nowcast.get('nowcast_es'))}**"
                )
            if governance_nowcast:
                lines.append(
                    f"- Governance nowcast: VaR **{_fmt_money(governance_nowcast.get('nowcast_var'))}**, "
                    f"ES **{_fmt_money(governance_nowcast.get('nowcast_es'))}**"
                )
            lines.append("")

        if microstructure or tick_quality:
            lines.append("### Market Microstructure")
            if microstructure:
                lines.append(f"- Market regime: **{microstructure.get('regime') or 'n/a'}**")
                lines.append(f"- Average spread: **{_fmt_number(microstructure.get('avg_spread_bps'), 2)} bps**")
                lines.append(
                    f"- Widest spread: **{_fmt_number(microstructure.get('widest_spread_bps'), 2)} bps** on **{microstructure.get('widest_symbol') or 'n/a'}**"
                )
            if tick_quality:
                lines.append(f"- Tick quality: **{tick_quality.get('status') or 'n/a'}**")
                lines.append(
                    f"- Healthy/stale/incomplete symbols: **{tick_quality.get('healthy_symbols', 0)} / {tick_quality.get('stale_symbols', 0)} / {tick_quality.get('incomplete_symbols', 0)}**"
                )
            lines.append("")

        if pnl_explain:
            lines.append("### PnL Explain")
            lines.append(
                f"- Realized: **{_fmt_money(pnl_explain.get('realized'))}** | "
                f"Unrealized: **{_fmt_money(pnl_explain.get('unrealized'))}** | "
                f"Swap: **{_fmt_money(pnl_explain.get('swap'))}**"
            )
            lines.append(
                f"- Commission/fee: **{_fmt_money(pnl_explain.get('commission'))} / {_fmt_money(pnl_explain.get('fee'))}** | "
                f"Estimated spread cost: **{_fmt_money(pnl_explain.get('estimated_spread_cost'))}**"
            )
            lines.append("")

        snapshot_limits = snapshot_payload.get("limits", {}) or {}
        if snapshot_limits:
            lines.append("### Live Zones")
            lines.append(f"- zone_hist: **{snapshot_limits.get('zone_hist')}**")
            lines.append(f"- zone_ewma: **{snapshot_limits.get('zone_ewma')}**")
            lines.append("")

    lines.append("## Backtest Summary (from compare CSV)")
    lines.append("| Model | Exceptions | Traffic light (99%/250 only) |")
    lines.append("|---|---:|---|")
    for model_name, exception_count in exceptions.items():
        traffic_light = "n/a"
        if alpha is not None and abs(alpha - 0.99) < 1e-6:
            if exception_count <= green_max:
                traffic_light = "GREEN"
            elif exception_count <= yellow_max:
                traffic_light = "YELLOW"
            else:
                traffic_light = "RED"
        lines.append(f"| {model_name} | {exception_count} | {traffic_light} |")
    lines.append("")

    lines.append("## Model Validation")
    lines.append("| Model | Score | Rate | Kupiec p | Ind p | CC p | Traffic light |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for model_name, result in validation.model_results.items():
        traffic_light = result.traffic_light or "n/a"
        lines.append(
            f"| {model_name} | {result.score:.2f} | {result.actual_rate:.4f} "
            f"| {result.p_uc:.4f} | {result.p_ind:.4f} | {result.p_cc:.4f} | {traffic_light} |"
        )
    if validation.best_model:
        lines.append("")
        lines.append(f"- Best model by score: **{validation.best_model}**")
    lines.append("")

    combined_alerts = validation_alerts + snapshot_alerts
    lines.append("## Alerts")
    if not combined_alerts:
        lines.append("_No active alerts from validation or the selected portfolio snapshot._")
    else:
        for alert in combined_alerts:
            lines.append(f"- [{alert.severity}] {alert.code}: {alert.message}")
    lines.append("")

    if exc_chart:
        lines.append("## Charts")
        lines.append(f"![Exceptions]({exc_chart.name})")
        lines.append("")
    if pnl_chart:
        lines.append(f"![PnL vs VaR]({pnl_chart.name})")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path

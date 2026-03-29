from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from var_project.alerts.engine import alerts_from_live_snapshot, alerts_from_validation_summary
from var_project.reporting.charts import _plot_exceptions, _plot_pnl_vs_var
from var_project.reporting.metrics import _infer_models, _parse_alpha_from_name, _count_exceptions
from var_project.validation.model_validation import validate_compare_frame


def _latest_file(dir_path: Path, pattern: str) -> Optional[Path]:
    if not dir_path.exists():
        return None
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


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


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "—"


def render_daily_markdown(
    compare_csv: Path,
    out_dir: Path,
    snapshot_dir: Optional[Path] = None,
    risk_limits_yaml: Optional[Path] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(compare_csv)
    models = _infer_models(df)
    alpha = _parse_alpha_from_name(compare_csv.stem)

    n = len(df)
    start = str(df["date"].iloc[0]) if "date" in df.columns and n > 0 else "—"
    end = str(df["date"].iloc[-1]) if "date" in df.columns and n > 0 else "—"

    exceptions = {m: _count_exceptions(df, m) for m in models}
    validation = validate_compare_frame(df, alpha if alpha is not None else 0.95)

    # limits
    limits = _load_limits(risk_limits_yaml) if risk_limits_yaml else {}
    tl_99 = (limits.get("backtest_traffic_light_99_250") or {})
    green_max = int(tl_99.get("green_max", 4))
    yellow_max = int(tl_99.get("yellow_max", 9))

    # snapshot
    snap = _load_latest_snapshot(snapshot_dir) if snapshot_dir else None
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # charts
    exc_chart = out_dir / f"{compare_csv.stem}_exceptions.png"
    pnl_chart = out_dir / f"{compare_csv.stem}_pnl_var.png"

    try:
        _plot_exceptions(df, exc_chart)
    except Exception:
        exc_chart = None

    # choose a model for pnl vs var
    plot_model = "hist" if "hist" in models else (models[0] if models else None)
    if plot_model:
        try:
            _plot_pnl_vs_var(df, pnl_chart, model=plot_model)
        except Exception:
            pnl_chart = None
    else:
        pnl_chart = None

    validation_alerts = alerts_from_validation_summary(validation)
    live_alerts = alerts_from_live_snapshot(snap, limits) if snap is not None else []

    # markdown
    md_path = out_dir / f"{compare_csv.stem}.md"
    lines: List[str] = []

    lines.append(f"# Risk Report — {compare_csv.stem}")
    lines.append("")
    lines.append(f"- Generated (UTC): **{now}**")
    lines.append(f"- Compare CSV: **{compare_csv.name}**")
    if alpha is not None:
        lines.append(f"- Alpha: **{alpha:.2f}** (q={1-alpha:.2f})")
    lines.append(f"- Sample (aligned days): **{n}**")
    lines.append(f"- Range: **{start} → {end}**")
    lines.append("")

    # snapshot section
    lines.append("## Latest Live Snapshot")
    if snap is None:
        lines.append("_No snapshot found in data/snapshots (run live once to generate)._")
    else:
        t = snap.get("time_utc", "—")
        live_pnl = snap.get("live_pnl", None) or snap.get("live_pnl_proxy", None)
        live_loss = snap.get("live_loss", None) or snap.get("live_loss_proxy", None)

        lines.append(f"- Snapshot time (UTC): **{t}**")
        lines.append(f"- live_pnl: **{_fmt_money(live_pnl)} EUR**")
        lines.append(f"- live_loss: **{_fmt_money(live_loss)} EUR**")
        lines.append("")
        # holdings / exposure
        pos = snap.get("exposure_by_symbol", {}) or snap.get("positions_eur", {}) or {}
        if pos:
            lines.append("### Portfolio Exposure (Base Currency)")
            lines.append("| Symbol | Exposure |")
            lines.append("|---|---:|")
            for k, v in pos.items():
                lines.append(f"| {k} | {_fmt_money(v)} |")
            lines.append("")

        # VaR/ES
        v = snap.get("var", {}) or {}
        e = snap.get("es", {}) or {}
        if v:
            lines.append("### VaR / ES (EUR)")
            lines.append("| Model | VaR | ES |")
            lines.append("|---|---:|---:|")
            all_models = sorted(set(list(v.keys()) + list(e.keys())))
            for m in all_models:
                lines.append(f"| {m} | {_fmt_money(v.get(m))} | {_fmt_money(e.get(m))} |")
            lines.append("")

        # zones if present
        lim = snap.get("limits", {}) or {}
        if lim:
            zh = lim.get("zone_hist", None)
            ze = lim.get("zone_ewma", None)
            lines.append("### Live Zones")
            lines.append(f"- zone_hist: **{zh}**")
            lines.append(f"- zone_ewma: **{ze}**")
            lines.append("")

    # backtest section
    lines.append("## Backtest Summary (from compare CSV)")
    lines.append("| Model | Exceptions | Traffic light (99%/250 only) |")
    lines.append("|---|---:|---|")
    for m, exc in exceptions.items():
        tl = "—"
        # apply classic TL only if alpha≈0.99 and window≈250 (we infer only from filename)
        if alpha is not None and abs(alpha - 0.99) < 1e-6:
            if exc <= green_max:
                tl = "GREEN"
            elif exc <= yellow_max:
                tl = "YELLOW"
            else:
                tl = "RED"
        lines.append(f"| {m} | {exc} | {tl} |")
    lines.append("")

    lines.append("## Model Validation")
    lines.append("| Model | Score | Rate | Kupiec p | Ind p | CC p | Traffic light |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for model_name, result in validation.model_results.items():
        tl = result.traffic_light or "—"
        lines.append(
            f"| {model_name} | {result.score:.2f} | {result.actual_rate:.4f} "
            f"| {result.p_uc:.4f} | {result.p_ind:.4f} | {result.p_cc:.4f} | {tl} |"
        )
    if validation.best_model:
        lines.append("")
        lines.append(f"- Best model by score: **{validation.best_model}**")
    lines.append("")

    combined_alerts = validation_alerts + live_alerts
    lines.append("## Alerts")
    if not combined_alerts:
        lines.append("_No active alerts from validation or latest live snapshot._")
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

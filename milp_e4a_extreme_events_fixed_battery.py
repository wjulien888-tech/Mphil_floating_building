"""
E4-a MILP dispatch for extracted extreme low-PV event windows.

Scope:
- PV input: pv_floating_disturbed_raw_kW only
- Battery: fixed 215 kWh / 50 kW / round-trip efficiency 0.92
- Mode: islanded, no grid import/export
- Objective: minimize unmet load first, then curtailment

The script is intentionally event-based. It solves the 24 h, 48 h, 72 h,
and 7 d low-PV windows separately, writes a CSV and SVG for each event, and
updates a progress SVG after every event so the run can be monitored visually.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dispatch_rbc_15min import BatteryParams, DT_HOURS
from optimizer_solver import build_pulp_solver


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

DISPATCH_INPUT_CSV = OUT_DIR / "dispatch_input_15min_disturbed_2023.csv"
EVENTS_CSV = OUT_DIR / "extreme_low_pv_events_15min.csv"
RBC_SUMMARY_CSV = OUT_DIR / "rbc_extreme_events_fixed_battery_summary.csv"

SUMMARY_CSV = OUT_DIR / "milp_e4a_extreme_events_summary.csv"
SUMMARY_MD = OUT_DIR / "milp_e4a_extreme_events_summary.md"
PROGRESS_SVG = PLOTS_DIR / "milp_e4a_progress.svg"
SUMMARY_SVG = PLOTS_DIR / "milp_e4a_extreme_events_summary.svg"

PV_COL = "pv_floating_disturbed_raw_kW"
LOAD_COL = "load_kW"


@dataclass(frozen=True)
class MilpConfig:
    solver: str = "gurobi"
    unmet_weight: float = 1000.0
    curtailment_weight: float = 1.0
    solver_time_limit_s: int | None = None
    mip_gap: float | None = None
    solver_log: bool = False


def event_label(hours: int) -> str:
    return f"{hours}h" if hours < 168 else "7d"


def ensure_pulp():
    try:
        import pulp  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pulp\n"
            "Install it first, for example:\n"
            "  python3 -m pip install pulp\n"
            "or, inside your project virtual environment:\n"
            "  pip install -r requirements.txt\n"
        ) from exc
    return pulp


def print_step(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def iter_with_progress(items: Iterable, total: int):
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(items, total=total, desc="E4-a MILP events", unit="event")
    except ModuleNotFoundError:
        for idx, item in enumerate(items, 1):
            print_step(f"Progress {idx}/{total}")
            yield item


def solve_event_milp(event_df: pd.DataFrame, config: MilpConfig) -> tuple[pd.DataFrame, dict]:
    pulp = ensure_pulp()

    battery = BatteryParams()
    eta_ch = battery.charge_efficiency
    eta_dis = battery.discharge_efficiency
    n = len(event_df)
    if n == 0:
        raise ValueError("Empty event dataframe")

    pv = event_df[PV_COL].to_numpy(dtype=float)
    load = event_df[LOAD_COL].to_numpy(dtype=float)

    model = pulp.LpProblem("E4a_Islanded_Battery_Dispatch", pulp.LpMinimize)

    p_ch = pulp.LpVariable.dicts("p_ch", range(n), lowBound=0, upBound=battery.power_kW)
    p_dis = pulp.LpVariable.dicts("p_dis", range(n), lowBound=0, upBound=battery.power_kW)
    p_unmet = pulp.LpVariable.dicts("p_unmet", range(n), lowBound=0)
    p_curt = pulp.LpVariable.dicts("p_curt", range(n), lowBound=0)
    soc = pulp.LpVariable.dicts(
        "soc",
        range(n),
        lowBound=battery.min_soc_kWh,
        upBound=battery.max_soc_kWh,
    )
    u_ch = pulp.LpVariable.dicts("u_ch", range(n), lowBound=0, upBound=1, cat="Binary")
    u_dis = pulp.LpVariable.dicts("u_dis", range(n), lowBound=0, upBound=1, cat="Binary")

    for t in range(n):
        model += p_ch[t] <= battery.power_kW * u_ch[t], f"charge_limit_{t}"
        model += p_dis[t] <= battery.power_kW * u_dis[t], f"discharge_limit_{t}"
        model += u_ch[t] + u_dis[t] <= 1, f"charge_discharge_mutex_{t}"

        model += (
            pv[t] + p_dis[t] + p_unmet[t]
            == load[t] + p_ch[t] + p_curt[t]
        ), f"power_balance_{t}"

        soc_prev = battery.initial_soc_kWh if t == 0 else soc[t - 1]
        model += (
            soc[t]
            == soc_prev + eta_ch * p_ch[t] * DT_HOURS - (p_dis[t] / eta_dis) * DT_HOURS
        ), f"soc_dynamics_{t}"

    model += (
        config.unmet_weight * pulp.lpSum(p_unmet[t] * DT_HOURS for t in range(n))
        + config.curtailment_weight * pulp.lpSum(p_curt[t] * DT_HOURS for t in range(n))
    )

    solver = build_pulp_solver(
        pulp,
        solver_name=config.solver,
        solver_log=config.solver_log,
        time_limit_s=config.solver_time_limit_s,
        mip_gap=config.mip_gap,
    )

    started = time.perf_counter()
    status_code = model.solve(solver)
    elapsed_s = time.perf_counter() - started
    status = pulp.LpStatus[status_code]

    out = event_df.copy().reset_index(drop=True)
    out["pv_kW"] = pv
    out["milp_battery_charge_kW"] = [pulp.value(p_ch[t]) or 0.0 for t in range(n)]
    out["milp_battery_discharge_kW"] = [pulp.value(p_dis[t]) or 0.0 for t in range(n)]
    out["milp_battery_soc_kWh"] = [pulp.value(soc[t]) or 0.0 for t in range(n)]
    out["milp_curtailment_kW"] = [pulp.value(p_curt[t]) or 0.0 for t in range(n)]
    out["milp_unmet_load_kW"] = [pulp.value(p_unmet[t]) or 0.0 for t in range(n)]
    out["milp_served_load_kW"] = out[LOAD_COL] - out["milp_unmet_load_kW"]
    out["milp_charge_binary"] = [pulp.value(u_ch[t]) or 0.0 for t in range(n)]
    out["milp_discharge_binary"] = [pulp.value(u_dis[t]) or 0.0 for t in range(n)]

    summary = summarize_event(out, status=status, solve_time_s=elapsed_s)
    return out, summary


def summarize_event(out: pd.DataFrame, status: str, solve_time_s: float) -> dict:
    load_kwh = float(out[LOAD_COL].sum() * DT_HOURS)
    pv_kwh = float(out["pv_kW"].sum() * DT_HOURS)
    served_kwh = float(out["milp_served_load_kW"].sum() * DT_HOURS)
    unmet_kwh = float(out["milp_unmet_load_kW"].sum() * DT_HOURS)
    curtail_kwh = float(out["milp_curtailment_kW"].sum() * DT_HOURS)
    charge_kwh = float(out["milp_battery_charge_kW"].sum() * DT_HOURS)
    discharge_kwh = float(out["milp_battery_discharge_kW"].sum() * DT_HOURS)
    min_soc = float(out["milp_battery_soc_kWh"].min())
    end_soc = float(out["milp_battery_soc_kWh"].iloc[-1])

    return {
        "solver_status": status,
        "solve_time_s": solve_time_s,
        "time_steps": int(len(out)),
        "load_kWh": load_kwh,
        "pv_kWh": pv_kwh,
        "served_load_kWh": served_kwh,
        "unmet_load_kWh": unmet_kwh,
        "unmet_load_fraction": unmet_kwh / load_kwh if load_kwh else math.nan,
        "curtailment_kWh": curtail_kwh,
        "curtailment_fraction": curtail_kwh / pv_kwh if pv_kwh else math.nan,
        "battery_charge_kWh_ac": charge_kwh,
        "battery_discharge_kWh_ac": discharge_kwh,
        "min_soc_kWh": min_soc,
        "end_soc_kWh": end_soc,
        "max_continuous_unmet_hours": max_continuous_hours(out["milp_unmet_load_kW"]),
    }


def max_continuous_hours(power_kw: pd.Series, threshold: float = 1e-6) -> float:
    active = power_kw.to_numpy(dtype=float) > threshold
    max_run = 0
    current = 0
    for flag in active:
        if flag:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run * DT_HOURS


def load_event_slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    event = df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()
    if event.empty:
        raise ValueError(f"No rows found between {start} and {end}")
    return event.reset_index(drop=True)


def merge_rbc_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    if not RBC_SUMMARY_CSV.exists():
        summary_df["rbc_unmet_load_kWh"] = np.nan
        summary_df["rbc_unmet_load_fraction"] = np.nan
        summary_df["rbc_max_continuous_unmet_hours"] = np.nan
        summary_df["milp_unmet_reduction_kWh"] = np.nan
        summary_df["milp_unmet_reduction_pct_of_rbc"] = np.nan
        return summary_df

    rbc = pd.read_csv(RBC_SUMMARY_CSV)
    keep = rbc[
        [
            "event_label",
            "unmet_load_kWh",
            "unmet_load_fraction",
            "max_continuous_unmet_hours",
        ]
    ].rename(
        columns={
            "unmet_load_kWh": "rbc_unmet_load_kWh",
            "unmet_load_fraction": "rbc_unmet_load_fraction",
            "max_continuous_unmet_hours": "rbc_max_continuous_unmet_hours",
        }
    )
    merged = summary_df.merge(keep, on="event_label", how="left")
    merged["milp_unmet_reduction_kWh"] = (
        merged["rbc_unmet_load_kWh"] - merged["unmet_load_kWh"]
    )
    merged["milp_unmet_reduction_pct_of_rbc"] = (
        merged["milp_unmet_reduction_kWh"] / merged["rbc_unmet_load_kWh"] * 100.0
    )
    return merged


def x_scale(value: float, min_value: float, max_value: float, left: float, width: float) -> float:
    if max_value == min_value:
        return left
    return left + (value - min_value) / (max_value - min_value) * width


def y_scale(value: float, max_value: float, top: float, height: float) -> float:
    if max_value <= 0:
        return top + height
    return top + (max_value - value) / max_value * height


def line_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def write_event_svg(out: pd.DataFrame, hours: int, start: pd.Timestamp, end: pd.Timestamp) -> None:
    label = event_label(hours)
    width = 1060
    height = 650
    margin_left = 72
    margin_right = 34
    top_1 = 72
    h_1 = 320
    top_2 = 455
    h_2 = 130
    chart_w = width - margin_left - margin_right

    plot = out.copy().reset_index(drop=True)
    plot["hour_from_start"] = (
        (plot["datetime"] - plot["datetime"].iloc[0]).dt.total_seconds() / 3600.0
    )
    x_min = 0.0
    x_max = max(float(plot["hour_from_start"].iloc[-1]), 1.0)
    power_max = float(
        max(
            plot[LOAD_COL].max(),
            plot["pv_kW"].max(),
            plot["milp_battery_discharge_kW"].max(),
            plot["milp_unmet_load_kW"].max(),
        )
        * 1.16
    )
    soc_max = BatteryParams().capacity_kWh

    load_points = []
    pv_points = []
    soc_points = []
    for row in plot.itertuples():
        x = x_scale(float(row.hour_from_start), x_min, x_max, margin_left, chart_w)
        load_points.append((x, y_scale(float(getattr(row, LOAD_COL)), power_max, top_1, h_1)))
        pv_points.append((x, y_scale(float(row.pv_kW), power_max, top_1, h_1)))
        soc_points.append((x, y_scale(float(row.milp_battery_soc_kWh), soc_max, top_2, h_2)))

    bar_w = max(1.0, chart_w / max(len(plot), 1) * 0.72)
    unmet_bars = []
    discharge_bars = []
    charge_bars = []
    for row in plot.itertuples():
        x = x_scale(float(row.hour_from_start), x_min, x_max, margin_left, chart_w) - bar_w / 2
        if row.milp_unmet_load_kW > 1e-6:
            y = y_scale(float(row.milp_unmet_load_kW), power_max, top_1, h_1)
            unmet_bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{top_1 + h_1 - y:.1f}" fill="#c44e52" opacity="0.45" />'
            )
        if row.milp_battery_discharge_kW > 1e-6:
            y = y_scale(float(row.milp_battery_discharge_kW), power_max, top_1, h_1)
            discharge_bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{top_1 + h_1 - y:.1f}" fill="#8f4f4f" opacity="0.22" />'
            )
        if row.milp_battery_charge_kW > 1e-6:
            y = y_scale(float(row.milp_battery_charge_kW), power_max, top_1, h_1)
            charge_bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{top_1 + h_1 - y:.1f}" fill="#5aa469" opacity="0.20" />'
            )

    x_ticks = [0, hours / 4, hours / 2, hours * 3 / 4, hours]
    x_ticks = [tick for tick in x_ticks if tick <= x_max + 0.5]
    y_ticks_power = [power_max * i / 4 for i in range(5)]
    y_ticks_soc = [soc_max * i / 4 for i in range(5)]

    x_grid = "\n".join(
        f'<line x1="{x_scale(t, x_min, x_max, margin_left, chart_w):.1f}" y1="{top_1}" '
        f'x2="{x_scale(t, x_min, x_max, margin_left, chart_w):.1f}" y2="{top_2 + h_2}" '
        f'stroke="#e6e0d6" stroke-width="1" />'
        for t in x_ticks
    )
    power_y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{y_scale(y, power_max, top_1, h_1):.1f}" '
        f'x2="{margin_left + chart_w}" y2="{y_scale(y, power_max, top_1, h_1):.1f}" '
        f'stroke="#e6e0d6" stroke-width="1" />'
        for y in y_ticks_power
    )
    soc_y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{y_scale(y, soc_max, top_2, h_2):.1f}" '
        f'x2="{margin_left + chart_w}" y2="{y_scale(y, soc_max, top_2, h_2):.1f}" '
        f'stroke="#e6e0d6" stroke-width="1" />'
        for y in y_ticks_soc
    )
    x_labels = "\n".join(
        f'<text x="{x_scale(t, x_min, x_max, margin_left, chart_w):.1f}" y="{height - 24}" '
        f'text-anchor="middle" font-size="12" fill="#2f3437">{t:.0f}</text>'
        for t in x_ticks
    )
    power_y_labels = "\n".join(
        f'<text x="{margin_left - 10}" y="{y_scale(y, power_max, top_1, h_1) + 4:.1f}" '
        f'text-anchor="end" font-size="12" fill="#2f3437">{y:.0f}</text>'
        for y in y_ticks_power
    )
    soc_y_labels = "\n".join(
        f'<text x="{margin_left - 10}" y="{y_scale(y, soc_max, top_2, h_2) + 4:.1f}" '
        f'text-anchor="end" font-size="12" fill="#2f3437">{y:.0f}</text>'
        for y in y_ticks_soc
    )

    total_load = float(plot[LOAD_COL].sum() * DT_HOURS)
    total_pv = float(plot["pv_kW"].sum() * DT_HOURS)
    total_unmet = float(plot["milp_unmet_load_kW"].sum() * DT_HOURS)
    unmet_pct = total_unmet / total_load * 100 if total_load else 0.0

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{margin_left}" y="34" font-size="22" font-weight="700" fill="#263238">E4-a MILP dispatch during {label} low-PV event</text>
  <text x="{margin_left}" y="55" font-size="13" fill="#687076">{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}; fixed battery 215 kWh / 50 kW</text>
  {x_grid}
  {power_y_grid}
  {soc_y_grid}
  {''.join(charge_bars)}
  {''.join(discharge_bars)}
  {''.join(unmet_bars)}
  <polyline points="{line_points(load_points)}" fill="none" stroke="#334e68" stroke-width="2.3" stroke-linejoin="round" />
  <polyline points="{line_points(pv_points)}" fill="none" stroke="#d99000" stroke-width="2.3" stroke-linejoin="round" />
  <polyline points="{line_points(soc_points)}" fill="none" stroke="#2f6f73" stroke-width="2.6" stroke-linejoin="round" />
  <line x1="{margin_left}" y1="{top_1 + h_1}" x2="{margin_left + chart_w}" y2="{top_1 + h_1}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{top_1}" x2="{margin_left}" y2="{top_1 + h_1}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{top_2 + h_2}" x2="{margin_left + chart_w}" y2="{top_2 + h_2}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{top_2}" x2="{margin_left}" y2="{top_2 + h_2}" stroke="#333" stroke-width="1.4" />
  {x_labels}
  {power_y_labels}
  {soc_y_labels}
  <text x="{margin_left + chart_w / 2:.1f}" y="{height - 5}" text-anchor="middle" font-size="14" fill="#263238">Hours from event start</text>
  <text x="18" y="{top_1 + h_1 / 2:.1f}" transform="rotate(-90 18 {top_1 + h_1 / 2:.1f})" text-anchor="middle" font-size="14" fill="#263238">Power (kW)</text>
  <text x="18" y="{top_2 + h_2 / 2:.1f}" transform="rotate(-90 18 {top_2 + h_2 / 2:.1f})" text-anchor="middle" font-size="14" fill="#263238">SOC (kWh)</text>
  <text x="{width - 310}" y="92" font-size="13" fill="#263238">Load {total_load:.1f} kWh</text>
  <text x="{width - 310}" y="113" font-size="13" fill="#263238">PV {total_pv:.1f} kWh</text>
  <text x="{width - 310}" y="134" font-size="13" fill="#263238">Unmet {total_unmet:.1f} kWh ({unmet_pct:.1f}%)</text>
  <line x1="{width - 310}" y1="166" x2="{width - 282}" y2="166" stroke="#334e68" stroke-width="3" />
  <text x="{width - 274}" y="170" font-size="12" fill="#263238">Load</text>
  <line x1="{width - 310}" y1="188" x2="{width - 282}" y2="188" stroke="#d99000" stroke-width="3" />
  <text x="{width - 274}" y="192" font-size="12" fill="#263238">Raw disturbed PV</text>
  <line x1="{width - 310}" y1="210" x2="{width - 282}" y2="210" stroke="#2f6f73" stroke-width="3" />
  <text x="{width - 274}" y="214" font-size="12" fill="#263238">MILP SOC</text>
  <rect x="{width - 310}" y="226" width="18" height="12" fill="#c44e52" opacity="0.45" />
  <text x="{width - 274}" y="237" font-size="12" fill="#263238">Unmet load</text>
</svg>
"""
    (PLOTS_DIR / f"milp_e4a_event_{label}_dispatch.svg").write_text(svg, encoding="utf-8")


def write_progress_svg(status_rows: list[dict], active_label: str | None = None) -> None:
    width = 760
    height = 280
    completed = sum(1 for row in status_rows if row["status"] == "done")
    total = len(status_rows)
    pct = completed / total if total else 0

    rows_svg = []
    for idx, row in enumerate(status_rows):
        y = 92 + idx * 36
        color = "#5aa469" if row["status"] == "done" else "#d99000" if row["label"] == active_label else "#c9c3b8"
        text = row["status"]
        if row.get("unmet_kWh") is not None:
            text += f" | unmet {row['unmet_kWh']:.1f} kWh"
        rows_svg.append(
            f'<circle cx="72" cy="{y}" r="8" fill="{color}" />'
            f'<text x="94" y="{y + 5}" font-size="14" fill="#263238">{row["label"]}: {text}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="48" y="36" font-size="22" font-weight="700" fill="#263238">E4-a MILP run progress</text>
  <text x="48" y="58" font-size="13" fill="#687076">Fixed battery 215 kWh / 50 kW; islanded extreme low-PV windows</text>
  <rect x="48" y="70" width="520" height="12" rx="6" fill="#e4ded4" />
  <rect x="48" y="70" width="{520 * pct:.1f}" height="12" rx="6" fill="#5aa469" />
  <text x="590" y="82" font-size="13" fill="#263238">{completed}/{total} complete</text>
  {''.join(rows_svg)}
</svg>
"""
    PROGRESS_SVG.write_text(svg, encoding="utf-8")


def write_summary_svg(summary: pd.DataFrame) -> None:
    width = 980
    height = 560
    margin_left = 80
    margin_right = 40
    margin_top = 72
    margin_bottom = 78
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    labels = list(summary["event_label"])
    group_w = chart_w / len(labels)
    bar_w = group_w * 0.24
    y_max = float(
        np.nanmax(
            [
                summary["unmet_load_kWh"].max(),
                summary["rbc_unmet_load_kWh"].max()
                if "rbc_unmet_load_kWh" in summary
                else summary["unmet_load_kWh"].max(),
            ]
        )
        * 1.18
    )

    def y(value: float) -> float:
        return y_scale(value, y_max, margin_top, chart_h)

    bars = []
    xlabels = []
    for idx, row in enumerate(summary.itertuples()):
        center = margin_left + group_w * (idx + 0.5)
        rbc_val = getattr(row, "rbc_unmet_load_kWh", math.nan)
        milp_val = row.unmet_load_kWh
        if not math.isnan(rbc_val):
            y_top = y(float(rbc_val))
            bars.append(
                f'<rect x="{center - bar_w * 1.1:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" '
                f'height="{margin_top + chart_h - y_top:.1f}" fill="#8b95a1" opacity="0.85" />'
            )
        y_top = y(float(milp_val))
        bars.append(
            f'<rect x="{center + bar_w * 0.1:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" '
            f'height="{margin_top + chart_h - y_top:.1f}" fill="#5aa469" opacity="0.90" />'
        )
        xlabels.append(
            f'<text x="{center:.1f}" y="{height - 42}" text-anchor="middle" font-size="13" fill="#263238">{row.event_label}</text>'
        )
        xlabels.append(
            f'<text x="{center:.1f}" y="{height - 22}" text-anchor="middle" font-size="11" fill="#687076">MILP {row.unmet_load_fraction * 100:.1f}%</text>'
        )

    y_ticks = [y_max * i / 5 for i in range(6)]
    y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{y(t):.1f}" x2="{margin_left + chart_w}" y2="{y(t):.1f}" stroke="#e6e0d6" />'
        for t in y_ticks
    )
    y_labels = "\n".join(
        f'<text x="{margin_left - 10}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="12" fill="#263238">{t:.0f}</text>'
        for t in y_ticks
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{margin_left}" y="34" font-size="22" font-weight="700" fill="#263238">E4-a extreme-event unmet load: RBC vs MILP</text>
  <text x="{margin_left}" y="55" font-size="13" fill="#687076">Same raw disturbed PV, same load, same fixed 215 kWh / 50 kW battery</text>
  {y_grid}
  {''.join(bars)}
  <line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#333" stroke-width="1.4" />
  {y_labels}
  {''.join(xlabels)}
  <text x="20" y="{margin_top + chart_h / 2:.1f}" transform="rotate(-90 20 {margin_top + chart_h / 2:.1f})" text-anchor="middle" font-size="14" fill="#263238">Unmet load (kWh)</text>
  <rect x="{width - 220}" y="84" width="16" height="12" fill="#8b95a1" opacity="0.85" />
  <text x="{width - 196}" y="95" font-size="13" fill="#263238">RBC</text>
  <rect x="{width - 220}" y="108" width="16" height="12" fill="#5aa469" opacity="0.90" />
  <text x="{width - 196}" y="119" font-size="13" fill="#263238">MILP</text>
</svg>
"""
    SUMMARY_SVG.write_text(svg, encoding="utf-8")


def write_summary_md(summary: pd.DataFrame) -> None:
    lines = [
        "# E4-a MILP Extreme-Event Dispatch Summary",
        "",
        "Battery is fixed at `215 kWh / 50 kW`; PV uses `pv_floating_disturbed_raw_kW` only.",
        "",
        "| Event | Status | PV | Load | MILP unmet | MILP unmet % | RBC unmet | Reduction | Min SOC | Solve time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        rbc_unmet = getattr(row, "rbc_unmet_load_kWh", math.nan)
        reduction = getattr(row, "milp_unmet_reduction_kWh", math.nan)
        lines.append(
            f"| {row.event_label} | {row.solver_status} | {row.pv_kWh:.1f} kWh | "
            f"{row.load_kWh:.1f} kWh | {row.unmet_load_kWh:.1f} kWh | "
            f"{row.unmet_load_fraction * 100:.1f}% | "
            f"{rbc_unmet:.1f} kWh | {reduction:.1f} kWh | "
            f"{row.min_soc_kWh:.1f} kWh | {row.solve_time_s:.1f} s |"
        )
    lines.extend(
        [
            "",
            "Figures:",
            "",
            "- `plots/milp_e4a_extreme_events_summary.svg`",
            "- `plots/milp_e4a_event_24h_dispatch.svg`",
            "- `plots/milp_e4a_event_48h_dispatch.svg`",
            "- `plots/milp_e4a_event_72h_dispatch.svg`",
            "- `plots/milp_e4a_event_7d_dispatch.svg`",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def run_e4a(config: MilpConfig, only_hours: list[int] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DISPATCH_INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {DISPATCH_INPUT_CSV}")
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"Missing {EVENTS_CSV}")

    print_step("Loading dispatch input and extreme event table")
    df = pd.read_csv(DISPATCH_INPUT_CSV, parse_dates=["datetime"])
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start", "end"])
    if only_hours:
        events = events[events["window_hours"].isin(only_hours)].copy()
    if events.empty:
        raise ValueError("No events selected")

    status_rows = [
        {"label": event_label(int(row.window_hours)), "status": "pending", "unmet_kWh": None}
        for row in events.itertuples()
    ]
    write_progress_svg(status_rows)

    summaries = []
    for idx, event in enumerate(iter_with_progress(list(events.itertuples()), total=len(events))):
        hours = int(event.window_hours)
        label = event_label(hours)
        start = pd.Timestamp(event.start)
        end = pd.Timestamp(event.end)

        status_rows[idx]["status"] = "solving"
        write_progress_svg(status_rows, active_label=label)

        event_df = load_event_slice(df, start, end)
        print_step(f"Solving {label}: {len(event_df)} steps, {start} -> {end}")
        out, summary = solve_event_milp(event_df, config)

        detail_csv = OUT_DIR / f"milp_e4a_event_{label}_fixed_battery_15min.csv"
        out.to_csv(detail_csv, index=False)
        write_event_svg(out, hours, start, end)

        summary.update(
            {
                "window_hours": hours,
                "event_label": label,
                "start": start,
                "end": end,
                "detail_csv": str(detail_csv.relative_to(ROOT)),
                "dispatch_svg": str((PLOTS_DIR / f"milp_e4a_event_{label}_dispatch.svg").relative_to(ROOT)),
            }
        )
        summaries.append(summary)

        status_rows[idx]["status"] = "done"
        status_rows[idx]["unmet_kWh"] = summary["unmet_load_kWh"]
        write_progress_svg(status_rows)
        print_step(
            f"{label} complete: status={summary['solver_status']}, "
            f"unmet={summary['unmet_load_kWh']:.1f} kWh "
            f"({summary['unmet_load_fraction'] * 100:.1f}%), "
            f"solve_time={summary['solve_time_s']:.1f}s"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df = merge_rbc_comparison(summary_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    write_summary_svg(summary_df)
    write_summary_md(summary_df)

    print_step(f"Saved summary CSV: {SUMMARY_CSV.relative_to(ROOT)}")
    print_step(f"Saved summary MD: {SUMMARY_MD.relative_to(ROOT)}")
    print_step(f"Saved summary SVG: {SUMMARY_SVG.relative_to(ROOT)}")
    print_step(f"Saved progress SVG: {PROGRESS_SVG.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E4-a MILP dispatch for extreme low-PV events.")
    parser.add_argument("--solver", choices=["gurobi", "cbc"], default="gurobi")
    parser.add_argument(
        "--only",
        nargs="*",
        type=int,
        choices=[24, 48, 72, 168],
        help="Only run selected event durations, e.g. --only 24 72",
    )
    parser.add_argument(
        "--solver-log",
        action="store_true",
        help="Show CBC solver log in the terminal.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Optional solver time limit per event in seconds.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Optional relative MIP gap, e.g. 0.001.",
    )
    parser.add_argument(
        "--unmet-weight",
        type=float,
        default=1000.0,
        help="Objective weight for unmet load energy.",
    )
    parser.add_argument(
        "--curtailment-weight",
        type=float,
        default=1.0,
        help="Objective weight for curtailed PV energy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MilpConfig(
        solver=args.solver,
        unmet_weight=args.unmet_weight,
        curtailment_weight=args.curtailment_weight,
        solver_time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        solver_log=args.solver_log,
    )
    run_e4a(config=config, only_hours=args.only)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)

"""
E5-a receding-horizon MPC dispatch for extracted extreme low-PV events.

Scope:
- PV input: pv_floating_disturbed_raw_kW only
- Battery: fixed 215 kWh / 50 kW / round-trip efficiency 0.92
- Mode: islanded, no grid import/export
- Default horizon: 24 h = 96 steps
- Objective: minimize unmet load first, then curtailment

The script writes event-level MPC CSVs, a summary table, and slide-ready SVGs.
It intentionally mirrors the E4-a outputs so RBC / MILP / MPC can be compared
under the same fixed hardware and event windows.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dispatch_rbc_15min import BatteryParams, DT_HOURS
from milp_e4a_extreme_events_fixed_battery import (
    EVENTS_CSV,
    LOAD_COL,
    PV_COL,
    ROOT,
    DISPATCH_INPUT_CSV,
    line_points,
    x_scale,
    y_scale,
)
from optimizer_solver import build_pulp_solver


OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

RBC_SUMMARY_CSV = OUT_DIR / "rbc_extreme_events_fixed_battery_summary.csv"
MILP_SUMMARY_CSV = OUT_DIR / "milp_e4a_extreme_events_summary.csv"

SUMMARY_CSV = OUT_DIR / "mpc_e5a_extreme_events_summary.csv"
SUMMARY_MD = OUT_DIR / "mpc_e5a_extreme_events_summary.md"
PROGRESS_SVG = PLOTS_DIR / "mpc_e5a_progress.svg"
SUMMARY_SVG = PLOTS_DIR / "mpc_e5a_extreme_events_summary.svg"
SOC_COMPARE_SVG = PLOTS_DIR / "mpc_e5a_extreme_events_soc_comparison.svg"


@dataclass(frozen=True)
class MpcIslandConfig:
    solver: str = "gurobi"
    horizon_steps: int = 96
    unmet_weight: float = 1000.0
    curtailment_weight: float = 1.0
    terminal_soc_weight: float = 0.0
    solver_time_limit_s: int | None = None
    solver_log: bool = False


def ensure_pulp():
    try:
        import pulp  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pulp\n"
            "Install it first with:\n"
            "  python3 -m pip install -r requirements.txt\n"
        ) from exc
    return pulp


def print_step(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_soc(value: float) -> float:
    battery = BatteryParams()
    clipped = min(max(float(value), battery.min_soc_kWh), battery.max_soc_kWh)
    return round(clipped, 6)


def event_label(hours: int) -> str:
    return f"{hours}h" if hours < 168 else "7d"


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


def solve_horizon(
    horizon_df: pd.DataFrame,
    initial_soc_kWh: float,
    config: MpcIslandConfig,
) -> tuple[dict, float, str, float]:
    pulp = ensure_pulp()
    battery = BatteryParams()
    initial_soc_kWh = normalize_soc(initial_soc_kWh)
    eta_ch = battery.charge_efficiency
    eta_dis = battery.discharge_efficiency
    n = len(horizon_df)
    pv = horizon_df[PV_COL].to_numpy(dtype=float)
    load = horizon_df[LOAD_COL].to_numpy(dtype=float)

    model = pulp.LpProblem("E5a_Islanded_MPC_Horizon", pulp.LpMinimize)
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
        model += p_ch[t] <= battery.power_kW * u_ch[t]
        model += p_dis[t] <= battery.power_kW * u_dis[t]
        model += u_ch[t] + u_dis[t] <= 1
        model += pv[t] + p_dis[t] + p_unmet[t] == load[t] + p_ch[t] + p_curt[t]
        soc_prev = initial_soc_kWh if t == 0 else soc[t - 1]
        model += soc[t] == soc_prev + eta_ch * p_ch[t] * DT_HOURS - (p_dis[t] / eta_dis) * DT_HOURS

    terminal_penalty = 0.0
    if config.terminal_soc_weight:
        terminal_penalty = config.terminal_soc_weight * (battery.max_soc_kWh - soc[n - 1])

    model += (
        config.unmet_weight * pulp.lpSum(p_unmet[t] * DT_HOURS for t in range(n))
        + config.curtailment_weight * pulp.lpSum(p_curt[t] * DT_HOURS for t in range(n))
        + terminal_penalty
    )

    solver = build_pulp_solver(
        pulp,
        solver_name=config.solver,
        solver_log=config.solver_log,
        time_limit_s=config.solver_time_limit_s,
        mip_gap=None,
    )

    started = time.perf_counter()
    status_code = model.solve(solver)
    elapsed_s = time.perf_counter() - started
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        raise RuntimeError(f"MPC horizon solve failed with status={status}")

    first = {
        "mpc_battery_charge_kW": pulp.value(p_ch[0]) or 0.0,
        "mpc_battery_discharge_kW": pulp.value(p_dis[0]) or 0.0,
        "mpc_unmet_load_kW": pulp.value(p_unmet[0]) or 0.0,
        "mpc_curtailment_kW": pulp.value(p_curt[0]) or 0.0,
        "mpc_horizon_status": status,
    }
    soc_value = pulp.value(soc[0])
    next_soc = normalize_soc(initial_soc_kWh if soc_value is None else float(soc_value))
    return first, next_soc, status, elapsed_s


def run_mpc_event(event_df: pd.DataFrame, config: MpcIslandConfig) -> tuple[pd.DataFrame, dict]:
    battery = BatteryParams()
    n = len(event_df)
    soc_now = normalize_soc(battery.initial_soc_kWh)
    rows = []
    total_solve_time = 0.0
    statuses: list[str] = []

    for k in range(n):
        horizon = event_df.iloc[k : min(n, k + config.horizon_steps)].copy()
        decision, soc_next, status, elapsed = solve_horizon(horizon, soc_now, config)
        total_solve_time += elapsed
        statuses.append(status)

        row = event_df.iloc[k].to_dict()
        row["pv_kW"] = float(row[PV_COL])
        row.update(decision)
        row["mpc_battery_soc_kWh"] = soc_next
        row["mpc_served_load_kW"] = float(row[LOAD_COL]) - float(row["mpc_unmet_load_kW"])
        rows.append(row)
        soc_now = normalize_soc(soc_next)

        if (k + 1) % 96 == 0 or k == n - 1:
            print_step(f"  MPC executed {k + 1}/{n} steps; SOC={soc_now:.1f} kWh")

    out = pd.DataFrame(rows)
    summary = summarize_event(out, total_solve_time, statuses, config)
    return out, summary


def summarize_event(
    out: pd.DataFrame,
    solve_time_s: float,
    statuses: list[str],
    config: MpcIslandConfig,
) -> dict:
    load_kwh = float(out[LOAD_COL].sum() * DT_HOURS)
    pv_kwh = float(out["pv_kW"].sum() * DT_HOURS)
    served_kwh = float(out["mpc_served_load_kW"].sum() * DT_HOURS)
    unmet_kwh = float(out["mpc_unmet_load_kW"].sum() * DT_HOURS)
    curtail_kwh = float(out["mpc_curtailment_kW"].sum() * DT_HOURS)
    return {
        "solver_status": "all_optimal" if set(statuses) == {"Optimal"} else ";".join(sorted(set(statuses))),
        "solve_time_s": solve_time_s,
        "horizon_steps": config.horizon_steps,
        "horizon_hours": config.horizon_steps * DT_HOURS,
        "time_steps": int(len(out)),
        "load_kWh": load_kwh,
        "pv_kWh": pv_kwh,
        "served_load_kWh": served_kwh,
        "unmet_load_kWh": unmet_kwh,
        "unmet_load_fraction": unmet_kwh / load_kwh if load_kwh else math.nan,
        "curtailment_kWh": curtail_kwh,
        "curtailment_fraction": curtail_kwh / pv_kwh if pv_kwh else math.nan,
        "battery_charge_kWh_ac": float(out["mpc_battery_charge_kW"].sum() * DT_HOURS),
        "battery_discharge_kWh_ac": float(out["mpc_battery_discharge_kW"].sum() * DT_HOURS),
        "min_soc_kWh": float(out["mpc_battery_soc_kWh"].min()),
        "end_soc_kWh": float(out["mpc_battery_soc_kWh"].iloc[-1]),
        "max_continuous_unmet_hours": max_continuous_hours(out["mpc_unmet_load_kW"]),
    }


def merge_strategy_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if RBC_SUMMARY_CSV.exists():
        rbc = pd.read_csv(RBC_SUMMARY_CSV)
        rbc = rbc[["event_label", "unmet_load_kWh", "unmet_load_fraction"]].rename(
            columns={
                "unmet_load_kWh": "rbc_unmet_load_kWh",
                "unmet_load_fraction": "rbc_unmet_load_fraction",
            }
        )
        out = out.merge(rbc, on="event_label", how="left")
    if MILP_SUMMARY_CSV.exists():
        milp = pd.read_csv(MILP_SUMMARY_CSV)
        milp = milp[["event_label", "unmet_load_kWh", "unmet_load_fraction"]].rename(
            columns={
                "unmet_load_kWh": "milp_unmet_load_kWh",
                "unmet_load_fraction": "milp_unmet_load_fraction",
            }
        )
        out = out.merge(milp, on="event_label", how="left")
    if "rbc_unmet_load_kWh" in out:
        out["mpc_unmet_reduction_vs_rbc_kWh"] = out["rbc_unmet_load_kWh"] - out["unmet_load_kWh"]
    if "milp_unmet_load_kWh" in out:
        out["mpc_unmet_gap_vs_milp_kWh"] = out["unmet_load_kWh"] - out["milp_unmet_load_kWh"]
    return out


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
  <text x="48" y="36" font-size="22" font-weight="700" fill="#263238">E5-a MPC run progress</text>
  <text x="48" y="58" font-size="13" fill="#687076">Fixed battery 215 kWh / 50 kW; islanded rolling-horizon control</text>
  <rect x="48" y="70" width="520" height="12" rx="6" fill="#e4ded4" />
  <rect x="48" y="70" width="{520 * pct:.1f}" height="12" rx="6" fill="#5aa469" />
  <text x="590" y="82" font-size="13" fill="#263238">{completed}/{total} complete</text>
  {''.join(rows_svg)}
</svg>
"""
    PROGRESS_SVG.write_text(svg, encoding="utf-8")


def write_event_svg(out: pd.DataFrame, hours: int, start: pd.Timestamp, end: pd.Timestamp) -> None:
    label = event_label(hours)
    width = 1060
    height = 650
    ml, mr = 72, 34
    top_1, h_1 = 72, 320
    top_2, h_2 = 455, 130
    cw = width - ml - mr

    plot = out.copy().reset_index(drop=True)
    plot["hour_from_start"] = (plot["datetime"] - plot["datetime"].iloc[0]).dt.total_seconds() / 3600.0
    x_min = 0.0
    x_max = max(float(plot["hour_from_start"].iloc[-1]), 1.0)
    power_max = float(
        max(
            plot[LOAD_COL].max(),
            plot["pv_kW"].max(),
            plot["mpc_battery_discharge_kW"].max(),
            plot["mpc_unmet_load_kW"].max(),
        )
        * 1.16
    )
    soc_max = BatteryParams().capacity_kWh

    load_points = []
    pv_points = []
    soc_points = []
    for row in plot.itertuples():
        x = x_scale(float(row.hour_from_start), x_min, x_max, ml, cw)
        load_points.append((x, y_scale(float(getattr(row, LOAD_COL)), power_max, top_1, h_1)))
        pv_points.append((x, y_scale(float(row.pv_kW), power_max, top_1, h_1)))
        soc_points.append((x, y_scale(float(row.mpc_battery_soc_kWh), soc_max, top_2, h_2)))

    bar_w = max(1.0, cw / max(len(plot), 1) * 0.72)
    bars = []
    for row in plot.itertuples():
        x = x_scale(float(row.hour_from_start), x_min, x_max, ml, cw) - bar_w / 2
        for key, color, opacity in [
            ("mpc_battery_charge_kW", "#5aa469", 0.20),
            ("mpc_battery_discharge_kW", "#8f4f4f", 0.22),
            ("mpc_unmet_load_kW", "#c44e52", 0.45),
        ]:
            val = float(getattr(row, key))
            if val > 1e-6:
                y = y_scale(val, power_max, top_1, h_1)
                bars.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top_1 + h_1 - y:.1f}" fill="{color}" opacity="{opacity}" />'
                )

    x_ticks = [tick for tick in [0, hours / 4, hours / 2, hours * 3 / 4, hours] if tick <= x_max + 0.5]
    y_ticks_power = [power_max * i / 4 for i in range(5)]
    y_ticks_soc = [soc_max * i / 4 for i in range(5)]
    x_grid = "\n".join(
        f'<line x1="{x_scale(t, x_min, x_max, ml, cw):.1f}" y1="{top_1}" x2="{x_scale(t, x_min, x_max, ml, cw):.1f}" y2="{top_2 + h_2}" stroke="#e6e0d6" />'
        for t in x_ticks
    )
    y_grid = "\n".join(
        f'<line x1="{ml}" y1="{y_scale(y, power_max, top_1, h_1):.1f}" x2="{ml + cw}" y2="{y_scale(y, power_max, top_1, h_1):.1f}" stroke="#e6e0d6" />'
        for y in y_ticks_power
    )
    soc_grid = "\n".join(
        f'<line x1="{ml}" y1="{y_scale(y, soc_max, top_2, h_2):.1f}" x2="{ml + cw}" y2="{y_scale(y, soc_max, top_2, h_2):.1f}" stroke="#e6e0d6" />'
        for y in y_ticks_soc
    )
    x_labels = "\n".join(
        f'<text x="{x_scale(t, x_min, x_max, ml, cw):.1f}" y="{height - 24}" text-anchor="middle" font-size="12" fill="#2f3437">{t:.0f}</text>'
        for t in x_ticks
    )
    y_labels = "\n".join(
        f'<text x="{ml - 10}" y="{y_scale(y, power_max, top_1, h_1) + 4:.1f}" text-anchor="end" font-size="12" fill="#2f3437">{y:.0f}</text>'
        for y in y_ticks_power
    )
    soc_labels = "\n".join(
        f'<text x="{ml - 10}" y="{y_scale(y, soc_max, top_2, h_2) + 4:.1f}" text-anchor="end" font-size="12" fill="#2f3437">{y:.0f}</text>'
        for y in y_ticks_soc
    )

    total_load = float(plot[LOAD_COL].sum() * DT_HOURS)
    total_pv = float(plot["pv_kW"].sum() * DT_HOURS)
    total_unmet = float(plot["mpc_unmet_load_kW"].sum() * DT_HOURS)
    unmet_pct = total_unmet / total_load * 100 if total_load else 0.0
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">E5-a MPC dispatch during {label} low-PV event</text>
  <text x="{ml}" y="55" font-size="13" fill="#687076">{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}; fixed battery 215 kWh / 50 kW</text>
  {x_grid}
  {y_grid}
  {soc_grid}
  {''.join(bars)}
  <polyline points="{line_points(load_points)}" fill="none" stroke="#334e68" stroke-width="2.3" stroke-linejoin="round" />
  <polyline points="{line_points(pv_points)}" fill="none" stroke="#d99000" stroke-width="2.3" stroke-linejoin="round" />
  <polyline points="{line_points(soc_points)}" fill="none" stroke="#2f6f73" stroke-width="2.6" stroke-linejoin="round" />
  <line x1="{ml}" y1="{top_1 + h_1}" x2="{ml + cw}" y2="{top_1 + h_1}" stroke="#333" />
  <line x1="{ml}" y1="{top_1}" x2="{ml}" y2="{top_1 + h_1}" stroke="#333" />
  <line x1="{ml}" y1="{top_2 + h_2}" x2="{ml + cw}" y2="{top_2 + h_2}" stroke="#333" />
  <line x1="{ml}" y1="{top_2}" x2="{ml}" y2="{top_2 + h_2}" stroke="#333" />
  {x_labels}
  {y_labels}
  {soc_labels}
  <text x="{ml + cw / 2:.1f}" y="{height - 5}" text-anchor="middle" font-size="14" fill="#263238">Hours from event start</text>
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
  <text x="{width - 274}" y="214" font-size="12" fill="#263238">MPC SOC</text>
  <rect x="{width - 310}" y="226" width="18" height="12" fill="#c44e52" opacity="0.45" />
  <text x="{width - 274}" y="237" font-size="12" fill="#263238">Unmet load</text>
</svg>
"""
    (PLOTS_DIR / f"mpc_e5a_event_{label}_dispatch.svg").write_text(svg, encoding="utf-8")


def write_summary_svg(summary: pd.DataFrame) -> None:
    width = 1040
    height = 560
    ml, mr, mt, mb = 82, 42, 72, 78
    cw, ch = width - ml - mr, height - mt - mb
    labels = list(summary["event_label"])
    group_w = cw / len(labels)
    bar_w = group_w * 0.18
    cols = [
        ("rbc_unmet_load_kWh", "#8b95a1", "RBC"),
        ("milp_unmet_load_kWh", "#5aa469", "MILP"),
        ("unmet_load_kWh", "#2f6f73", "MPC"),
    ]
    max_val = float(summary[[c for c, _, _ in cols if c in summary]].max().max()) * 1.18
    bars = []
    xlabels = []
    for idx, row in enumerate(summary.itertuples()):
        center = ml + group_w * (idx + 0.5)
        for j, (col, color, _) in enumerate(cols):
            if not hasattr(row, col):
                continue
            val = float(getattr(row, col))
            y = y_scale(val, max_val, mt, ch)
            x = center + (j - 1.2) * bar_w
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{mt + ch - y:.1f}" fill="{color}" opacity="0.88" />')
        xlabels.append(f'<text x="{center:.1f}" y="{height - 42}" text-anchor="middle" font-size="13" fill="#263238">{row.event_label}</text>')
        xlabels.append(f'<text x="{center:.1f}" y="{height - 22}" text-anchor="middle" font-size="11" fill="#687076">MPC {row.unmet_load_fraction * 100:.1f}%</text>')
    y_ticks = [max_val * i / 5 for i in range(6)]
    grid = "".join(f'<line x1="{ml}" y1="{y_scale(t, max_val, mt, ch):.1f}" x2="{ml + cw}" y2="{y_scale(t, max_val, mt, ch):.1f}" stroke="#e6e0d6" />' for t in y_ticks)
    ylabels = "".join(f'<text x="{ml - 10}" y="{y_scale(t, max_val, mt, ch) + 4:.1f}" text-anchor="end" font-size="12" fill="#263238">{t:.0f}</text>' for t in y_ticks)
    legend = []
    for i, (_, color, name) in enumerate(cols):
        legend.append(f'<rect x="{width - 230}" y="{84 + i * 24}" width="16" height="12" fill="{color}" opacity="0.88" />')
        legend.append(f'<text x="{width - 206}" y="{95 + i * 24}" font-size="13" fill="#263238">{name}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">E5-a extreme-event unmet load: RBC vs MILP vs MPC</text>
  <text x="{ml}" y="55" font-size="13" fill="#687076">Same raw disturbed PV, same load, same fixed 215 kWh / 50 kW battery</text>
  {grid}
  {''.join(bars)}
  <line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#333" />
  <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#333" />
  {ylabels}
  {''.join(xlabels)}
  {''.join(legend)}
  <text x="20" y="{mt + ch / 2:.1f}" transform="rotate(-90 20 {mt + ch / 2:.1f})" text-anchor="middle" font-size="14" fill="#263238">Unmet load (kWh)</text>
</svg>
"""
    SUMMARY_SVG.write_text(svg, encoding="utf-8")


def write_soc_comparison_svg(summary: pd.DataFrame) -> None:
    width = 1080
    height = 600
    ml, mr, mt, mb = 74, 38, 68, 58
    panel_h = (height - mt - mb - 28) / 2
    panel_w = (width - ml - mr - 28) / 2
    panels = []
    for idx, row in enumerate(summary.itertuples()):
        label = row.event_label
        col = idx % 2
        line = idx // 2
        x0 = ml + col * (panel_w + 28)
        y0 = mt + line * (panel_h + 28)
        paths = []
        event_hours = int(row.window_hours)
        files = [
            (OUT_DIR / f"rbc_extreme_event_{label}_fixed_battery_15min.csv", "battery_soc_kWh", "#8b95a1", "RBC"),
            (OUT_DIR / f"milp_e4a_event_{label}_fixed_battery_15min.csv", "milp_battery_soc_kWh", "#5aa469", "MILP"),
            (OUT_DIR / f"mpc_e5a_event_{label}_fixed_battery_15min.csv", "mpc_battery_soc_kWh", "#2f6f73", "MPC"),
        ]
        for path, soc_col, color, _ in files:
            if not path.exists():
                continue
            data = pd.read_csv(path, parse_dates=["datetime"]).reset_index(drop=True)
            pts = []
            for i, item in enumerate(data.itertuples()):
                x = x_scale(i * DT_HOURS, 0, max(event_hours, 1), x0, panel_w)
                y = y_scale(float(getattr(item, soc_col)), BatteryParams().capacity_kWh, y0, panel_h)
                pts.append((x, y))
            paths.append(f'<polyline points="{line_points(pts)}" fill="none" stroke="{color}" stroke-width="2" />')
        panels.append(
            f'<text x="{x0}" y="{y0 - 10}" font-size="14" font-weight="700" fill="#263238">{label}</text>'
            f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#d8d1c6" />'
            f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" stroke="#333" />'
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" stroke="#333" />'
            f'{"".join(paths)}'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">Extreme-event SOC comparison</text>
  <text x="{ml}" y="55" font-size="13" fill="#687076">RBC, full-horizon MILP, and rolling-horizon MPC under the same events</text>
  {''.join(panels)}
  <line x1="{width - 260}" y1="34" x2="{width - 236}" y2="34" stroke="#8b95a1" stroke-width="3" /><text x="{width - 228}" y="38" font-size="12" fill="#263238">RBC</text>
  <line x1="{width - 260}" y1="54" x2="{width - 236}" y2="54" stroke="#5aa469" stroke-width="3" /><text x="{width - 228}" y="58" font-size="12" fill="#263238">MILP</text>
  <line x1="{width - 180}" y1="34" x2="{width - 156}" y2="34" stroke="#2f6f73" stroke-width="3" /><text x="{width - 148}" y="38" font-size="12" fill="#263238">MPC</text>
</svg>
"""
    SOC_COMPARE_SVG.write_text(svg, encoding="utf-8")


def write_summary_md(summary: pd.DataFrame) -> None:
    lines = [
        "# E5-a MPC Extreme-Event Dispatch Summary",
        "",
        "Battery is fixed at `215 kWh / 50 kW`; PV uses `pv_floating_disturbed_raw_kW` only.",
        "",
        "| Event | Status | PV | Load | MPC unmet | MPC unmet % | RBC unmet | MILP unmet | Min SOC | Solve time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        rbc = getattr(row, "rbc_unmet_load_kWh", math.nan)
        milp = getattr(row, "milp_unmet_load_kWh", math.nan)
        lines.append(
            f"| {row.event_label} | {row.solver_status} | {row.pv_kWh:.1f} kWh | "
            f"{row.load_kWh:.1f} kWh | {row.unmet_load_kWh:.1f} kWh | "
            f"{row.unmet_load_fraction * 100:.1f}% | {rbc:.1f} kWh | {milp:.1f} kWh | "
            f"{row.min_soc_kWh:.1f} kWh | {row.solve_time_s:.1f} s |"
        )
    lines.extend(
        [
            "",
            "Figures:",
            "",
            "- `plots/mpc_e5a_extreme_events_summary.svg`",
            "- `plots/mpc_e5a_extreme_events_soc_comparison.svg`",
            "- `plots/mpc_e5a_event_24h_dispatch.svg`",
            "- `plots/mpc_e5a_event_48h_dispatch.svg`",
            "- `plots/mpc_e5a_event_72h_dispatch.svg`",
            "- `plots/mpc_e5a_event_7d_dispatch.svg`",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def run_e5a(config: MpcIslandConfig, only_hours: list[int] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DISPATCH_INPUT_CSV, parse_dates=["datetime"])
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start", "end"])
    if only_hours:
        events = events[events["window_hours"].isin(only_hours)].copy()
    if events.empty:
        raise ValueError("No events selected")

    status_rows = [{"label": event_label(int(row.window_hours)), "status": "pending", "unmet_kWh": None} for row in events.itertuples()]
    write_progress_svg(status_rows)

    summaries = []
    for idx, event in enumerate(events.itertuples()):
        hours = int(event.window_hours)
        label = event_label(hours)
        start = pd.Timestamp(event.start)
        end = pd.Timestamp(event.end)
        status_rows[idx]["status"] = "solving"
        write_progress_svg(status_rows, active_label=label)

        event_df = df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy().reset_index(drop=True)
        print_step(f"Running E5-a {label}: {len(event_df)} steps, horizon={config.horizon_steps} steps")
        out, summary = run_mpc_event(event_df, config)
        detail_csv = OUT_DIR / f"mpc_e5a_event_{label}_fixed_battery_15min.csv"
        out.to_csv(detail_csv, index=False)
        write_event_svg(out, hours, start, end)

        summary.update(
            {
                "window_hours": hours,
                "event_label": label,
                "start": start,
                "end": end,
                "detail_csv": str(detail_csv.relative_to(ROOT)),
                "dispatch_svg": str((PLOTS_DIR / f"mpc_e5a_event_{label}_dispatch.svg").relative_to(ROOT)),
            }
        )
        summaries.append(summary)
        status_rows[idx]["status"] = "done"
        status_rows[idx]["unmet_kWh"] = summary["unmet_load_kWh"]
        write_progress_svg(status_rows)
        print_step(f"{label} complete: unmet={summary['unmet_load_kWh']:.1f} kWh, solve_time={summary['solve_time_s']:.1f}s")

    summary_df = pd.DataFrame(summaries)
    summary_df = merge_strategy_comparison(summary_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    write_summary_svg(summary_df)
    write_soc_comparison_svg(summary_df)
    write_summary_md(summary_df)
    print_step(f"Saved summary CSV: {SUMMARY_CSV.relative_to(ROOT)}")
    print_step(f"Saved summary MD: {SUMMARY_MD.relative_to(ROOT)}")
    print_step(f"Saved figures: {SUMMARY_SVG.relative_to(ROOT)}, {SOC_COMPARE_SVG.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E5-a rolling-horizon MPC for extreme low-PV events.")
    parser.add_argument("--solver", choices=["gurobi", "cbc"], default="gurobi")
    parser.add_argument("--only", nargs="*", type=int, choices=[24, 48, 72, 168], help="Only run selected event durations.")
    parser.add_argument("--horizon-steps", type=int, default=96, help="MPC horizon in 15-minute steps. Default 96 = 24 h.")
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument("--time-limit", type=int, default=None, help="Optional solver time limit per MPC step in seconds.")
    parser.add_argument("--unmet-weight", type=float, default=1000.0)
    parser.add_argument("--curtailment-weight", type=float, default=1.0)
    parser.add_argument("--terminal-soc-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MpcIslandConfig(
        solver=args.solver,
        horizon_steps=args.horizon_steps,
        unmet_weight=args.unmet_weight,
        curtailment_weight=args.curtailment_weight,
        terminal_soc_weight=args.terminal_soc_weight,
        solver_time_limit_s=args.time_limit,
        solver_log=args.solver_log,
    )
    run_e5a(config, only_hours=args.only)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)

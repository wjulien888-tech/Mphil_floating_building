"""
E4-b grid-connected MILP dispatch for the full 2023 raw disturbed PV year.

Scope:
- PV input: pv_floating_disturbed_raw_kW only
- Load input: load_kW
- Battery: fixed 215 kWh / 50 kW / round-trip efficiency 0.92
- Mode: grid-connected
- Grid charging is disabled; the battery charges only from PV.
- Simultaneous grid import and PV export are disabled.
- Objective: minimize net grid cost, with small penalties on curtailment/cycling.

The default run solves the full year month by month with SOC carried across
month boundaries. This keeps the MILP size visible and manageable while still
covering all 35,040 15-minute time steps.
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
from optimizer_solver import build_pulp_solver


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

DISPATCH_INPUT_CSV = OUT_DIR / "dispatch_input_15min_disturbed_2023.csv"
RBC_ECONOMICS_CSV = OUT_DIR / "dispatch_grid_vpp_economics_summary.csv"

DETAIL_CSV = OUT_DIR / "milp_e4b_full_year_grid_dispatch_2023.csv"
SUMMARY_CSV = OUT_DIR / "milp_e4b_full_year_grid_summary.csv"
SUMMARY_MD = OUT_DIR / "milp_e4b_full_year_grid_summary.md"
MONTHLY_CSV = OUT_DIR / "milp_e4b_full_year_grid_monthly_summary.csv"

PROGRESS_SVG = PLOTS_DIR / "milp_e4b_progress.svg"
SOC_SVG = PLOTS_DIR / "milp_e4b_full_year_soc.svg"
MONTHLY_SVG = PLOTS_DIR / "milp_e4b_monthly_import_export_cost.svg"
COMPARE_SVG = PLOTS_DIR / "milp_e4b_rbc_vs_milp_grid_cost.svg"

PV_COL = "pv_floating_disturbed_raw_kW"
LOAD_COL = "load_kW"

EXPORT_PRICE_CNY_PER_KWH = 0.36
IMPORT_TARIFFS = {
    "valley": 0.29526875,
    "flat": 0.73196875,
    "peak": 1.22496875,
    "sharp_peak": 1.52436875,
}


@dataclass(frozen=True)
class MilpGridConfig:
    solver: str = "gurobi"
    curtailment_penalty: float = 0.001
    cycle_penalty: float = 0.0001
    solver_time_limit_s: int | None = None
    mip_gap: float | None = None
    solver_log: bool = False


def ensure_pulp():
    try:
        import pulp  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pulp\n"
            "Install it first with:\n"
            "  .venv/bin/python -m pip install pulp\n"
            "or:\n"
            "  .venv/bin/python -m pip install -r requirements.txt\n"
        ) from exc
    return pulp


def print_step(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_high_temperature_days(df: pd.DataFrame) -> set[pd.Timestamp]:
    daily_max = df.groupby(df["datetime"].dt.floor("D"))["temperature_C"].max()
    return set(daily_max[daily_max >= 35.0].index)


def tariff_period(ts: pd.Timestamp, high_temperature_days: set[pd.Timestamp]) -> str:
    month = int(ts.month)
    hour = ts.hour + ts.minute / 60.0
    is_sharp_month = 7 <= month <= 9
    is_high_temperature_day = ts.floor("D") in high_temperature_days
    is_sharp_hour = (11 <= hour < 12) or (15 <= hour < 17)

    if (is_sharp_month or is_high_temperature_day) and is_sharp_hour:
        return "sharp_peak"
    if 0 <= hour < 8:
        return "valley"
    if (10 <= hour < 12) or (14 <= hour < 19):
        return "peak"
    return "flat"


def add_tariff_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hot_days = build_high_temperature_days(out)
    out["tariff_period"] = [tariff_period(ts, hot_days) for ts in out["datetime"]]
    out["import_tariff_CNY_per_kWh"] = out["tariff_period"].map(IMPORT_TARIFFS)
    out["export_price_CNY_per_kWh"] = EXPORT_PRICE_CNY_PER_KWH
    return out


def solve_grid_milp(month_df: pd.DataFrame, initial_soc_kWh: float, config: MilpGridConfig):
    pulp = ensure_pulp()
    battery = BatteryParams()
    eta_ch = battery.charge_efficiency
    eta_dis = battery.discharge_efficiency

    n = len(month_df)
    pv = month_df[PV_COL].to_numpy(dtype=float)
    load = month_df[LOAD_COL].to_numpy(dtype=float)
    buy_price = month_df["import_tariff_CNY_per_kWh"].to_numpy(dtype=float)
    sell_price = month_df["export_price_CNY_per_kWh"].to_numpy(dtype=float)

    model = pulp.LpProblem("E4b_Full_Year_Grid_Connected_Dispatch", pulp.LpMinimize)

    pv_to_load = pulp.LpVariable.dicts("pv_to_load", range(n), lowBound=0)
    pv_to_batt = pulp.LpVariable.dicts("pv_to_batt", range(n), lowBound=0, upBound=battery.power_kW)
    pv_export = pulp.LpVariable.dicts("pv_export", range(n), lowBound=0)
    pv_curt = pulp.LpVariable.dicts("pv_curt", range(n), lowBound=0)
    batt_to_load = pulp.LpVariable.dicts("batt_to_load", range(n), lowBound=0, upBound=battery.power_kW)
    grid_to_load = pulp.LpVariable.dicts("grid_to_load", range(n), lowBound=0)
    soc = pulp.LpVariable.dicts(
        "soc",
        range(n),
        lowBound=battery.min_soc_kWh,
        upBound=battery.max_soc_kWh,
    )
    u_ch = pulp.LpVariable.dicts("u_ch", range(n), lowBound=0, upBound=1, cat="Binary")
    u_dis = pulp.LpVariable.dicts("u_dis", range(n), lowBound=0, upBound=1, cat="Binary")
    u_imp = pulp.LpVariable.dicts("u_imp", range(n), lowBound=0, upBound=1, cat="Binary")
    u_exp = pulp.LpVariable.dicts("u_exp", range(n), lowBound=0, upBound=1, cat="Binary")

    for t in range(n):
        model += pv_to_load[t] + pv_to_batt[t] + pv_export[t] + pv_curt[t] == pv[t], f"pv_balance_{t}"
        model += pv_to_load[t] + batt_to_load[t] + grid_to_load[t] == load[t], f"load_balance_{t}"
        model += pv_to_batt[t] <= battery.power_kW * u_ch[t], f"charge_limit_{t}"
        model += batt_to_load[t] <= battery.power_kW * u_dis[t], f"discharge_limit_{t}"
        model += u_ch[t] + u_dis[t] <= 1, f"charge_discharge_mutex_{t}"
        model += grid_to_load[t] <= max(float(load[t]), 1e-6) * u_imp[t], f"grid_import_limit_{t}"
        model += pv_export[t] <= max(float(pv[t]), 1e-6) * u_exp[t], f"grid_export_limit_{t}"
        model += u_imp[t] + u_exp[t] <= 1, f"grid_import_export_mutex_{t}"

        soc_prev = initial_soc_kWh if t == 0 else soc[t - 1]
        model += (
            soc[t]
            == soc_prev + eta_ch * pv_to_batt[t] * DT_HOURS - (batt_to_load[t] / eta_dis) * DT_HOURS
        ), f"soc_dynamics_{t}"

    model += pulp.lpSum(
        (
            buy_price[t] * grid_to_load[t]
            - sell_price[t] * pv_export[t]
            + config.curtailment_penalty * pv_curt[t]
            + config.cycle_penalty * (pv_to_batt[t] + batt_to_load[t])
        )
        * DT_HOURS
        for t in range(n)
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

    out = month_df.copy().reset_index(drop=True)
    out["milp_pv_to_load_kW"] = [pulp.value(pv_to_load[t]) or 0.0 for t in range(n)]
    out["milp_pv_to_battery_kW"] = [pulp.value(pv_to_batt[t]) or 0.0 for t in range(n)]
    out["milp_pv_export_kW"] = [pulp.value(pv_export[t]) or 0.0 for t in range(n)]
    out["milp_pv_curtailment_kW"] = [pulp.value(pv_curt[t]) or 0.0 for t in range(n)]
    out["milp_battery_to_load_kW"] = [pulp.value(batt_to_load[t]) or 0.0 for t in range(n)]
    out["milp_grid_import_kW"] = [pulp.value(grid_to_load[t]) or 0.0 for t in range(n)]
    out["milp_battery_soc_kWh"] = [pulp.value(soc[t]) or 0.0 for t in range(n)]
    out["milp_charge_binary"] = [pulp.value(u_ch[t]) or 0.0 for t in range(n)]
    out["milp_discharge_binary"] = [pulp.value(u_dis[t]) or 0.0 for t in range(n)]
    out["milp_grid_import_binary"] = [pulp.value(u_imp[t]) or 0.0 for t in range(n)]
    out["milp_grid_export_binary"] = [pulp.value(u_exp[t]) or 0.0 for t in range(n)]
    out["milp_grid_import_kWh"] = out["milp_grid_import_kW"] * DT_HOURS
    out["milp_grid_export_kWh"] = out["milp_pv_export_kW"] * DT_HOURS
    out["milp_import_cost_CNY"] = out["milp_grid_import_kWh"] * out["import_tariff_CNY_per_kWh"]
    out["milp_export_revenue_CNY"] = out["milp_grid_export_kWh"] * out["export_price_CNY_per_kWh"]
    out["milp_net_grid_cost_CNY"] = out["milp_import_cost_CNY"] - out["milp_export_revenue_CNY"]

    summary = summarize_dispatch(out, status, elapsed_s)
    return out, summary


def summarize_dispatch(out: pd.DataFrame, status: str, solve_time_s: float) -> dict:
    return {
        "solver_status": status,
        "solve_time_s": solve_time_s,
        "time_steps": int(len(out)),
        "load_kWh": float(out[LOAD_COL].sum() * DT_HOURS),
        "pv_kWh": float(out[PV_COL].sum() * DT_HOURS),
        "grid_import_kWh": float(out["milp_grid_import_kWh"].sum()),
        "grid_export_kWh": float(out["milp_grid_export_kWh"].sum()),
        "curtailment_kWh": float(out["milp_pv_curtailment_kW"].sum() * DT_HOURS),
        "battery_charge_kWh_ac": float(out["milp_pv_to_battery_kW"].sum() * DT_HOURS),
        "battery_discharge_kWh_ac": float(out["milp_battery_to_load_kW"].sum() * DT_HOURS),
        "import_cost_CNY": float(out["milp_import_cost_CNY"].sum()),
        "export_revenue_CNY": float(out["milp_export_revenue_CNY"].sum()),
        "net_grid_cost_CNY": float(out["milp_net_grid_cost_CNY"].sum()),
        "min_soc_kWh": float(out["milp_battery_soc_kWh"].min()),
        "max_soc_kWh": float(out["milp_battery_soc_kWh"].max()),
        "end_soc_kWh": float(out["milp_battery_soc_kWh"].iloc[-1]),
        "max_grid_import_kW": float(out["milp_grid_import_kW"].max()),
        "max_grid_export_kW": float(out["milp_pv_export_kW"].max()),
    }


def write_progress_svg(rows: list[dict], active_month: int | None = None) -> None:
    width = 860
    height = 430
    completed = sum(1 for row in rows if row["status"] == "done")
    total = len(rows)
    pct = completed / total if total else 0
    row_svg = []
    for idx, row in enumerate(rows):
        col = idx % 4
        line = idx // 4
        x = 70 + col * 190
        y = 120 + line * 72
        color = "#5aa469" if row["status"] == "done" else "#d99000" if row["month"] == active_month else "#c9c3b8"
        label = f'M{row["month"]:02d}: {row["status"]}'
        if row.get("cost") is not None:
            label += f' | {row["cost"]:.0f} CNY'
        row_svg.append(
            f'<circle cx="{x}" cy="{y}" r="9" fill="{color}" />'
            f'<text x="{x + 18}" y="{y + 5}" font-size="13" fill="#263238">{label}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="48" y="38" font-size="22" font-weight="700" fill="#263238">E4-b full-year grid-connected MILP progress</text>
  <text x="48" y="61" font-size="13" fill="#687076">Monthly MILP with SOC carried forward; fixed battery 215 kWh / 50 kW</text>
  <rect x="48" y="78" width="560" height="12" rx="6" fill="#e4ded4" />
  <rect x="48" y="78" width="{560 * pct:.1f}" height="12" rx="6" fill="#5aa469" />
  <text x="630" y="90" font-size="13" fill="#263238">{completed}/{total} months complete</text>
  {''.join(row_svg)}
</svg>
"""
    PROGRESS_SVG.write_text(svg, encoding="utf-8")


def scale(value: float, d0: float, d1: float, r0: float, r1: float) -> float:
    if d1 == d0:
        return r0
    return r0 + (value - d0) / (d1 - d0) * (r1 - r0)


def write_soc_svg(out: pd.DataFrame) -> None:
    width = 1180
    height = 440
    ml, mr, mt, mb = 74, 34, 56, 54
    cw, ch = width - ml - mr, height - mt - mb
    n = len(out)
    y_max = BatteryParams().capacity_kWh
    points = []
    for i, row in enumerate(out.itertuples()):
        x = scale(i, 0, n - 1, ml, ml + cw)
        y = scale(row.milp_battery_soc_kWh, 0, y_max, mt + ch, mt)
        points.append(f"{x:.1f},{y:.1f}")
    month_ticks = out.groupby(out["datetime"].dt.month).head(1)
    x_labels = []
    x_grid = []
    for row in month_ticks.itertuples():
        i = row.Index
        x = scale(i, 0, n - 1, ml, ml + cw)
        x_grid.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ch}" stroke="#e6e0d6" />')
        x_labels.append(f'<text x="{x:.1f}" y="{height - 20}" text-anchor="middle" font-size="11" fill="#263238">{row.datetime:%b}</text>')
    y_ticks = [0, 50, 100, 150, 200]
    y_grid = []
    y_labels = []
    for val in y_ticks:
        y = scale(val, 0, y_max, mt + ch, mt)
        y_grid.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + cw}" y2="{y:.1f}" stroke="#e6e0d6" />')
        y_labels.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#263238">{val}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">E4-b MILP full-year battery SOC</text>
  <text x="{ml}" y="52" font-size="13" fill="#687076">Grid-connected economic dispatch; fixed battery 215 kWh / 50 kW</text>
  {''.join(x_grid)}
  {''.join(y_grid)}
  <polyline points="{' '.join(points)}" fill="none" stroke="#2f6f73" stroke-width="1.7" />
  <line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#333" />
  <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#333" />
  {''.join(x_labels)}
  {''.join(y_labels)}
  <text x="20" y="{mt + ch / 2:.1f}" transform="rotate(-90 20 {mt + ch / 2:.1f})" text-anchor="middle" font-size="13" fill="#263238">SOC (kWh)</text>
</svg>
"""
    SOC_SVG.write_text(svg, encoding="utf-8")


def write_monthly_svg(monthly: pd.DataFrame) -> None:
    width = 1120
    height = 520
    ml, mr, mt, mb = 76, 46, 62, 68
    cw, ch = width - ml - mr, height - mt - mb
    y_max = max(monthly["grid_import_kWh"].max(), monthly["grid_export_kWh"].max()) * 1.2
    cost_min = min(0.0, monthly["net_grid_cost_CNY"].min())
    cost_max = max(monthly["import_cost_CNY"].max(), monthly["export_revenue_CNY"].max(), monthly["net_grid_cost_CNY"].max()) * 1.2
    group_w = cw / len(monthly)
    bar_w = group_w * 0.18
    bars = []
    labels = []
    cost_points = []
    for idx, row in enumerate(monthly.itertuples()):
        cx = ml + group_w * (idx + 0.5)
        imp_y = scale(row.grid_import_kWh, 0, y_max, mt + ch, mt)
        exp_y = scale(row.grid_export_kWh, 0, y_max, mt + ch, mt)
        bars.append(f'<rect x="{cx - bar_w * 1.2:.1f}" y="{imp_y:.1f}" width="{bar_w:.1f}" height="{mt + ch - imp_y:.1f}" fill="#c44e52" opacity="0.82" />')
        bars.append(f'<rect x="{cx + bar_w * 0.2:.1f}" y="{exp_y:.1f}" width="{bar_w:.1f}" height="{mt + ch - exp_y:.1f}" fill="#5aa469" opacity="0.82" />')
        labels.append(f'<text x="{cx:.1f}" y="{height - 28}" text-anchor="middle" font-size="12" fill="#263238">{int(row.month)}</text>')
        cost_y = scale(row.net_grid_cost_CNY, cost_min, cost_max, mt + ch, mt)
        cost_points.append(f"{cx:.1f},{cost_y:.1f}")
    y_ticks = [y_max * i / 5 for i in range(6)]
    grid = []
    ylabels = []
    for val in y_ticks:
        y = scale(val, 0, y_max, mt + ch, mt)
        grid.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + cw}" y2="{y:.1f}" stroke="#e6e0d6" />')
        ylabels.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#263238">{val:.0f}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">E4-b monthly grid import/export and net cost</text>
  <text x="{ml}" y="54" font-size="13" fill="#687076">Bars: energy (kWh); line: monthly net grid cost (CNY, scaled)</text>
  {''.join(grid)}
  {''.join(bars)}
  <polyline points="{' '.join(cost_points)}" fill="none" stroke="#334e68" stroke-width="2.4" />
  <line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#333" />
  <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#333" />
  {''.join(labels)}
  {''.join(ylabels)}
  <text x="{ml + cw / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="13" fill="#263238">Month</text>
  <text x="20" y="{mt + ch / 2:.1f}" transform="rotate(-90 20 {mt + ch / 2:.1f})" text-anchor="middle" font-size="13" fill="#263238">Energy (kWh)</text>
  <rect x="{width - 250}" y="80" width="16" height="12" fill="#c44e52" opacity="0.82" />
  <text x="{width - 226}" y="91" font-size="12" fill="#263238">Grid import</text>
  <rect x="{width - 250}" y="104" width="16" height="12" fill="#5aa469" opacity="0.82" />
  <text x="{width - 226}" y="115" font-size="12" fill="#263238">Grid export</text>
  <line x1="{width - 250}" y1="132" x2="{width - 226}" y2="132" stroke="#334e68" stroke-width="3" />
  <text x="{width - 218}" y="136" font-size="12" fill="#263238">Net cost</text>
</svg>
"""
    MONTHLY_SVG.write_text(svg, encoding="utf-8")


def write_compare_svg(summary: dict, rbc_summary: dict | None) -> None:
    width = 760
    height = 430
    ml, mr, mt, mb = 88, 40, 64, 70
    cw, ch = width - ml - mr, height - mt - mb
    categories = ["grid_import_kWh", "grid_export_kWh", "net_grid_cost_CNY"]
    labels = ["Import kWh", "Export kWh", "Net cost CNY"]
    milp_vals = [summary.get(k, 0.0) for k in categories]
    rbc_vals = [rbc_summary.get(k, 0.0) if rbc_summary else 0.0 for k in categories]
    max_val = max(max(abs(v) for v in milp_vals + rbc_vals), 1.0) * 1.2
    group_w = cw / len(categories)
    bar_w = group_w * 0.22
    zero_y = mt + ch
    bars = []
    xlabels = []
    for idx, label in enumerate(labels):
        cx = ml + group_w * (idx + 0.5)
        for offset, value, color, name in [
            (-0.7, rbc_vals[idx], "#8b95a1", "RBC"),
            (0.2, milp_vals[idx], "#5aa469", "MILP"),
        ]:
            y = scale(abs(value), 0, max_val, mt + ch, mt)
            bars.append(f'<rect x="{cx + offset * bar_w:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{zero_y - y:.1f}" fill="{color}" opacity="0.88" />')
        xlabels.append(f'<text x="{cx:.1f}" y="{height - 34}" text-anchor="middle" font-size="12" fill="#263238">{label}</text>')
    y_ticks = [max_val * i / 5 for i in range(6)]
    grid = []
    ylabels = []
    for val in y_ticks:
        y = scale(val, 0, max_val, mt + ch, mt)
        grid.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + cw}" y2="{y:.1f}" stroke="#e6e0d6" />')
        ylabels.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#263238">{val:.0f}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">E4-b grid-connected annual comparison</text>
  <text x="{ml}" y="54" font-size="13" fill="#687076">Existing RBC grid-economics proxy vs MILP economic dispatch</text>
  {''.join(grid)}
  {''.join(bars)}
  <line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#333" />
  <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#333" />
  {''.join(ylabels)}
  {''.join(xlabels)}
  <rect x="{width - 180}" y="86" width="16" height="12" fill="#8b95a1" opacity="0.88" />
  <text x="{width - 156}" y="97" font-size="12" fill="#263238">RBC</text>
  <rect x="{width - 180}" y="110" width="16" height="12" fill="#5aa469" opacity="0.88" />
  <text x="{width - 156}" y="121" font-size="12" fill="#263238">MILP</text>
</svg>
"""
    COMPARE_SVG.write_text(svg, encoding="utf-8")


def summarize_monthly(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["month"] = out["datetime"].dt.month
    rows = []
    for month, grp in out.groupby("month"):
        row = summarize_dispatch(grp, status="combined", solve_time_s=0.0)
        row["month"] = int(month)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month")


def load_rbc_summary() -> dict | None:
    if not RBC_ECONOMICS_CSV.exists():
        return None
    row = pd.read_csv(RBC_ECONOMICS_CSV).iloc[0].to_dict()
    return {
        "grid_import_kWh": float(row.get("grid_import_kWh", 0.0)),
        "grid_export_kWh": float(row.get("grid_export_kWh", 0.0)),
        "import_cost_CNY": float(row.get("import_cost_CNY", 0.0)),
        "export_revenue_CNY": float(row.get("export_revenue_CNY", 0.0)),
        "net_grid_cost_CNY": float(row.get("net_grid_cost_CNY", 0.0)),
    }


def write_summary_md(summary: dict, monthly: pd.DataFrame, rbc: dict | None) -> None:
    lines = [
        "# E4-b Full-Year Grid-Connected MILP Summary",
        "",
        "Battery is fixed at `215 kWh / 50 kW`; PV uses `pv_floating_disturbed_raw_kW` only.",
        "Grid charging is disabled; the battery charges only from PV.",
        "Simultaneous grid import and export are disabled in each 15-minute step.",
        "",
        "## Annual MILP Result",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Grid import | {summary['grid_import_kWh']:.1f} kWh |",
        f"| Grid export | {summary['grid_export_kWh']:.1f} kWh |",
        f"| Curtailment | {summary['curtailment_kWh']:.1f} kWh |",
        f"| Import cost | {summary['import_cost_CNY']:.1f} CNY |",
        f"| Export revenue | {summary['export_revenue_CNY']:.1f} CNY |",
        f"| Net grid cost | {summary['net_grid_cost_CNY']:.1f} CNY |",
        f"| Min SOC | {summary['min_soc_kWh']:.1f} kWh |",
        f"| End SOC | {summary['end_soc_kWh']:.1f} kWh |",
        "",
    ]
    if rbc:
        lines.extend(
            [
                "## Existing RBC Economics Proxy",
                "",
                "| Metric | RBC | MILP | Difference (MILP - RBC) |",
                "| --- | ---: | ---: | ---: |",
                f"| Grid import | {rbc['grid_import_kWh']:.1f} kWh | {summary['grid_import_kWh']:.1f} kWh | {summary['grid_import_kWh'] - rbc['grid_import_kWh']:.1f} kWh |",
                f"| Grid export | {rbc['grid_export_kWh']:.1f} kWh | {summary['grid_export_kWh']:.1f} kWh | {summary['grid_export_kWh'] - rbc['grid_export_kWh']:.1f} kWh |",
                f"| Net grid cost | {rbc['net_grid_cost_CNY']:.1f} CNY | {summary['net_grid_cost_CNY']:.1f} CNY | {summary['net_grid_cost_CNY'] - rbc['net_grid_cost_CNY']:.1f} CNY |",
                "",
            ]
        )
    lines.extend(
        [
            "## Figures",
            "",
            "- `plots/milp_e4b_full_year_soc.svg`",
            "- `plots/milp_e4b_monthly_import_export_cost.svg`",
            "- `plots/milp_e4b_rbc_vs_milp_grid_cost.svg`",
            "- `plots/milp_e4b_progress.svg`",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def run_e4b(config: MilpGridConfig, only_months: list[int] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DISPATCH_INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {DISPATCH_INPUT_CSV}")

    print_step("Loading full-year dispatch input")
    df = pd.read_csv(DISPATCH_INPUT_CSV, parse_dates=["datetime"])
    df = add_tariff_columns(df)
    df["month"] = df["datetime"].dt.month
    months = sorted(df["month"].unique())
    if only_months:
        months = [m for m in months if m in only_months]
    if not months:
        raise ValueError("No months selected")

    progress_rows = [{"month": int(m), "status": "pending", "cost": None} for m in months]
    write_progress_svg(progress_rows)

    battery = BatteryParams()
    initial_soc = battery.initial_soc_kWh
    solved = []
    month_summaries = []

    for idx, month in enumerate(months):
        progress_rows[idx]["status"] = "solving"
        write_progress_svg(progress_rows, active_month=int(month))

        month_df = df[df["month"] == month].copy().reset_index(drop=True)
        print_step(f"Solving month {int(month):02d}: {len(month_df)} steps, initial SOC={initial_soc:.1f} kWh")
        out, summary = solve_grid_milp(month_df, initial_soc, config)
        summary["month"] = int(month)
        month_summaries.append(summary)
        solved.append(out)
        initial_soc = summary["end_soc_kWh"]

        progress_rows[idx]["status"] = "done"
        progress_rows[idx]["cost"] = summary["net_grid_cost_CNY"]
        write_progress_svg(progress_rows)
        print_step(
            f"Month {int(month):02d} complete: status={summary['solver_status']}, "
            f"import={summary['grid_import_kWh']:.1f} kWh, export={summary['grid_export_kWh']:.1f} kWh, "
            f"net_cost={summary['net_grid_cost_CNY']:.1f} CNY, end_SOC={summary['end_soc_kWh']:.1f} kWh, "
            f"solve_time={summary['solve_time_s']:.1f}s"
        )

    full = pd.concat(solved, ignore_index=True)
    full.to_csv(DETAIL_CSV, index=False)

    summary = summarize_dispatch(full, status="combined", solve_time_s=float(sum(m["solve_time_s"] for m in month_summaries)))
    summary["solver_status"] = "monthly_combined"
    summary["months_solved"] = len(months)
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)

    monthly = pd.DataFrame(month_summaries).sort_values("month")
    monthly.to_csv(MONTHLY_CSV, index=False)

    rbc = load_rbc_summary()
    write_soc_svg(full)
    write_monthly_svg(monthly)
    write_compare_svg(summary, rbc)
    write_summary_md(summary, monthly, rbc)

    print_step(f"Saved detail CSV: {DETAIL_CSV.relative_to(ROOT)}")
    print_step(f"Saved summary CSV: {SUMMARY_CSV.relative_to(ROOT)}")
    print_step(f"Saved monthly CSV: {MONTHLY_CSV.relative_to(ROOT)}")
    print_step(f"Saved summary MD: {SUMMARY_MD.relative_to(ROOT)}")
    print_step(f"Saved figures: {SOC_SVG.relative_to(ROOT)}, {MONTHLY_SVG.relative_to(ROOT)}, {COMPARE_SVG.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E4-b full-year grid-connected MILP dispatch.")
    parser.add_argument("--solver", choices=["gurobi", "cbc"], default="gurobi")
    parser.add_argument("--only-months", nargs="*", type=int, choices=list(range(1, 13)), help="Only solve selected months.")
    parser.add_argument("--solver-log", action="store_true", help="Show CBC solver log.")
    parser.add_argument("--time-limit", type=int, default=None, help="Optional solver time limit per month in seconds.")
    parser.add_argument("--mip-gap", type=float, default=None, help="Optional relative MIP gap, e.g. 0.001.")
    parser.add_argument("--curtailment-penalty", type=float, default=0.001)
    parser.add_argument("--cycle-penalty", type=float, default=0.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MilpGridConfig(
        solver=args.solver,
        curtailment_penalty=args.curtailment_penalty,
        cycle_penalty=args.cycle_penalty,
        solver_time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        solver_log=args.solver_log,
    )
    run_e4b(config, only_months=args.only_months)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)

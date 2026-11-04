"""
E5-b grid-connected receding-horizon MPC for the full 2023 raw disturbed PV year.

Scope:
- PV input: pv_floating_disturbed_raw_kW only
- Load input: load_kW
- Battery: fixed 215 kWh / 50 kW / round-trip efficiency 0.92
- Mode: grid-connected
- Grid charging is disabled; the battery charges only from PV.
- Default horizon: 24 h = 96 steps
- Default control step: 1 step = 15 minutes

The horizon optimization reuses the E4-b grid-connected MILP model and executes
only the first control interval before rolling forward. This mirrors the E4-b
economic objective while changing the information assumption from month-ahead
offline optimization to finite-horizon online control.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dispatch_rbc_15min import BatteryParams, DT_HOURS
from milp_e4b_full_year_grid_fixed_battery import (
    DISPATCH_INPUT_CSV,
    LOAD_COL,
    PV_COL,
    ROOT,
    add_tariff_columns,
    load_rbc_summary,
    scale,
    solve_grid_milp,
    MilpGridConfig,
)


OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

MILP_SUMMARY_CSV = OUT_DIR / "milp_e4b_full_year_grid_summary.csv"

DETAIL_CSV = OUT_DIR / "mpc_e5b_full_year_grid_dispatch_2023.csv"
SUMMARY_CSV = OUT_DIR / "mpc_e5b_full_year_grid_summary.csv"
SUMMARY_MD = OUT_DIR / "mpc_e5b_full_year_grid_summary.md"
MONTHLY_CSV = OUT_DIR / "mpc_e5b_full_year_grid_monthly_summary.csv"

PROGRESS_SVG = PLOTS_DIR / "mpc_e5b_progress.svg"
SOC_SVG = PLOTS_DIR / "mpc_e5b_full_year_soc.svg"
MONTHLY_SVG = PLOTS_DIR / "mpc_e5b_monthly_import_export_cost.svg"
COMPARE_SVG = PLOTS_DIR / "mpc_e5b_rbc_milp_mpc_grid_cost.svg"


@dataclass(frozen=True)
class MpcGridConfig:
    solver: str = "gurobi"
    horizon_steps: int = 96
    control_step: int = 1
    curtailment_penalty: float = 0.001
    cycle_penalty: float = 0.0001
    solver_time_limit_s: int | None = None
    mip_gap: float | None = None
    solver_log: bool = False


def print_step(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_soc(value: float) -> float:
    battery = BatteryParams()
    clipped = min(max(float(value), battery.min_soc_kWh), battery.max_soc_kWh)
    return round(clipped, 6)


def write_progress_svg(total_steps: int, executed_steps: int, active_month: int | None = None) -> None:
    width = 820
    height = 250
    pct = executed_steps / total_steps if total_steps else 0.0
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="48" y="38" font-size="22" font-weight="700" fill="#263238">E5-b full-year grid-connected MPC progress</text>
  <text x="48" y="61" font-size="13" fill="#687076">Rolling-horizon MPC; fixed battery 215 kWh / 50 kW; active month {active_month or "-"}</text>
  <rect x="48" y="86" width="600" height="14" rx="7" fill="#e4ded4" />
  <rect x="48" y="86" width="{600 * pct:.1f}" height="14" rx="7" fill="#5aa469" />
  <text x="670" y="99" font-size="13" fill="#263238">{executed_steps}/{total_steps} steps</text>
  <text x="48" y="138" font-size="13" fill="#263238">Completion: {pct * 100:.1f}%</text>
</svg>
"""
    PROGRESS_SVG.write_text(svg, encoding="utf-8")


def summarize_dispatch(out: pd.DataFrame, status: str, solve_time_s: float) -> dict:
    return {
        "solver_status": status,
        "solve_time_s": solve_time_s,
        "time_steps": int(len(out)),
        "load_kWh": float(out[LOAD_COL].sum() * DT_HOURS),
        "pv_kWh": float(out[PV_COL].sum() * DT_HOURS),
        "grid_import_kWh": float(out["mpc_grid_import_kWh"].sum()),
        "grid_export_kWh": float(out["mpc_grid_export_kWh"].sum()),
        "curtailment_kWh": float(out["mpc_pv_curtailment_kW"].sum() * DT_HOURS),
        "battery_charge_kWh_ac": float(out["mpc_pv_to_battery_kW"].sum() * DT_HOURS),
        "battery_discharge_kWh_ac": float(out["mpc_battery_to_load_kW"].sum() * DT_HOURS),
        "import_cost_CNY": float(out["mpc_import_cost_CNY"].sum()),
        "export_revenue_CNY": float(out["mpc_export_revenue_CNY"].sum()),
        "net_grid_cost_CNY": float(out["mpc_net_grid_cost_CNY"].sum()),
        "min_soc_kWh": float(out["mpc_battery_soc_kWh"].min()),
        "max_soc_kWh": float(out["mpc_battery_soc_kWh"].max()),
        "end_soc_kWh": float(out["mpc_battery_soc_kWh"].iloc[-1]),
        "max_grid_import_kW": float(out["mpc_grid_import_kW"].max()),
        "max_grid_export_kW": float(out["mpc_pv_export_kW"].max()),
    }


def rename_milp_to_mpc(out: pd.DataFrame) -> pd.DataFrame:
    rename = {col: col.replace("milp_", "mpc_") for col in out.columns if col.startswith("milp_")}
    return out.rename(columns=rename)


def run_e5b(
    config: MpcGridConfig,
    only_months: list[int] | None = None,
    max_steps: int | None = None,
    resume: bool = False,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if config.control_step < 1:
        raise ValueError("--control-step must be >= 1")
    if config.horizon_steps < config.control_step:
        raise ValueError("--horizon-steps must be >= --control-step")

    print_step("Loading full-year dispatch input")
    df = pd.read_csv(DISPATCH_INPUT_CSV, parse_dates=["datetime"])
    df = add_tariff_columns(df)
    df["month"] = df["datetime"].dt.month
    if only_months:
        df = df[df["month"].isin(only_months)].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows selected")
    if max_steps is not None:
        df = df.iloc[:max_steps].copy().reset_index(drop=True)

    battery = BatteryParams()
    soc_now = normalize_soc(battery.initial_soc_kWh)
    grid_config = MilpGridConfig(
        solver=config.solver,
        curtailment_penalty=config.curtailment_penalty,
        cycle_penalty=config.cycle_penalty,
        solver_time_limit_s=config.solver_time_limit_s,
        mip_gap=config.mip_gap,
        solver_log=config.solver_log,
    )

    total_steps = len(df)
    write_progress_svg(total_steps, 0)
    solved_chunks = []
    statuses: list[str] = []
    total_solve_time = 0.0
    k = 0

    if resume and DETAIL_CSV.exists():
        partial = pd.read_csv(DETAIL_CSV, parse_dates=["datetime"])
        if len(partial) > len(df):
            raise ValueError(f"Cannot resume: existing {DETAIL_CSV} has more rows than selected input")
        if len(partial):
            expected = df.iloc[: len(partial)]["datetime"].reset_index(drop=True)
            actual = partial["datetime"].reset_index(drop=True)
            if not expected.equals(actual):
                raise ValueError("Cannot resume: existing partial CSV datetime sequence does not match selected input")
            solved_chunks.append(partial)
            k = len(partial)
            soc_now = normalize_soc(float(partial["mpc_battery_soc_kWh"].iloc[-1]))
            statuses.extend(str(x) for x in partial.get("mpc_horizon_status", pd.Series(dtype=str)).dropna().unique())
            print_step(f"Resuming from {DETAIL_CSV.relative_to(ROOT)} at step {k + 1}; SOC={soc_now:.6f} kWh")

    while k < total_steps:
        horizon = df.iloc[k : min(total_steps, k + config.horizon_steps)].copy().reset_index(drop=True)
        active_month = int(df.iloc[k]["month"])
        print_step(
            f"MPC solve at step {k + 1}/{total_steps}; month={active_month:02d}; "
            f"horizon={len(horizon)}; SOC={soc_now:.1f} kWh"
        )
        soc_now = normalize_soc(soc_now)
        horizon_out, summary = solve_grid_milp(horizon, soc_now, grid_config)
        statuses.append(str(summary["solver_status"]))
        total_solve_time += float(summary["solve_time_s"])
        if summary["solver_status"] != "Optimal":
            raise RuntimeError(f"MPC horizon solve failed at step {k + 1} with status={summary['solver_status']}")

        execute_n = min(config.control_step, total_steps - k, len(horizon_out))
        executed = rename_milp_to_mpc(horizon_out.iloc[:execute_n].copy())
        executed["mpc_horizon_status"] = summary["solver_status"]
        executed["mpc_horizon_solve_time_s"] = summary["solve_time_s"]
        solved_chunks.append(executed)

        soc_now = normalize_soc(float(executed["mpc_battery_soc_kWh"].iloc[-1]))
        k += execute_n

        if k % 96 == 0 or k >= total_steps:
            write_progress_svg(total_steps, k, active_month=active_month)
            partial = pd.concat(solved_chunks, ignore_index=True)
            partial.to_csv(DETAIL_CSV, index=False)
            print_step(f"  Executed {k}/{total_steps} steps; partial CSV saved; SOC={soc_now:.1f} kWh")

    full = pd.concat(solved_chunks, ignore_index=True)
    full.to_csv(DETAIL_CSV, index=False)

    status = "all_optimal" if set(statuses) == {"Optimal"} else ";".join(sorted(set(statuses)))
    summary = summarize_dispatch(full, status=status, solve_time_s=total_solve_time)
    summary["horizon_steps"] = config.horizon_steps
    summary["horizon_hours"] = config.horizon_steps * DT_HOURS
    summary["control_step"] = config.control_step
    summary["control_step_hours"] = config.control_step * DT_HOURS
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)

    monthly = summarize_monthly(full)
    monthly.to_csv(MONTHLY_CSV, index=False)

    rbc = load_rbc_summary()
    milp = load_milp_summary()
    write_soc_svg(full)
    write_monthly_svg(monthly)
    write_compare_svg(summary, rbc, milp)
    write_summary_md(summary, monthly, rbc, milp)

    print_step(f"Saved detail CSV: {DETAIL_CSV.relative_to(ROOT)}")
    print_step(f"Saved summary CSV: {SUMMARY_CSV.relative_to(ROOT)}")
    print_step(f"Saved monthly CSV: {MONTHLY_CSV.relative_to(ROOT)}")
    print_step(f"Saved summary MD: {SUMMARY_MD.relative_to(ROOT)}")
    print_step(f"Saved figures: {SOC_SVG.relative_to(ROOT)}, {MONTHLY_SVG.relative_to(ROOT)}, {COMPARE_SVG.relative_to(ROOT)}")


def summarize_monthly(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, grp in out.groupby(out["datetime"].dt.month):
        row = summarize_dispatch(grp, status="combined", solve_time_s=0.0)
        row["month"] = int(month)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month")


def load_milp_summary() -> dict | None:
    if not MILP_SUMMARY_CSV.exists():
        return None
    row = pd.read_csv(MILP_SUMMARY_CSV).iloc[0].to_dict()
    return {
        "grid_import_kWh": float(row.get("grid_import_kWh", 0.0)),
        "grid_export_kWh": float(row.get("grid_export_kWh", 0.0)),
        "import_cost_CNY": float(row.get("import_cost_CNY", 0.0)),
        "export_revenue_CNY": float(row.get("export_revenue_CNY", 0.0)),
        "net_grid_cost_CNY": float(row.get("net_grid_cost_CNY", 0.0)),
    }


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
        y = scale(row.mpc_battery_soc_kWh, 0, y_max, mt + ch, mt)
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
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">E5-b MPC full-year battery SOC</text>
  <text x="{ml}" y="52" font-size="13" fill="#687076">Grid-connected rolling economic dispatch; fixed battery 215 kWh / 50 kW</text>
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
    if cost_max == cost_min:
        cost_max = cost_min + 1.0
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
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">E5-b monthly grid import/export and net cost</text>
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


def write_compare_svg(summary: dict, rbc: dict | None, milp: dict | None) -> None:
    width = 860
    height = 440
    ml, mr, mt, mb = 88, 40, 64, 72
    cw, ch = width - ml - mr, height - mt - mb
    categories = ["grid_import_kWh", "grid_export_kWh", "net_grid_cost_CNY"]
    labels = ["Import kWh", "Export kWh", "Net cost CNY"]
    series = [
        ("RBC", rbc, "#8b95a1"),
        ("MILP", milp, "#5aa469"),
        ("MPC", summary, "#2f6f73"),
    ]
    vals = []
    for _, data, _ in series:
        vals.extend([(data or {}).get(k, 0.0) for k in categories])
    max_val = max(max(abs(v) for v in vals), 1.0) * 1.2
    group_w = cw / len(categories)
    bar_w = group_w * 0.16
    bars = []
    xlabels = []
    for idx, label in enumerate(labels):
        cx = ml + group_w * (idx + 0.5)
        for j, (_, data, color) in enumerate(series):
            value = (data or {}).get(categories[idx], 0.0)
            y = scale(abs(value), 0, max_val, mt + ch, mt)
            x = cx + (j - 1.35) * bar_w
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{mt + ch - y:.1f}" fill="{color}" opacity="0.88" />')
        xlabels.append(f'<text x="{cx:.1f}" y="{height - 34}" text-anchor="middle" font-size="12" fill="#263238">{label}</text>')
    y_ticks = [max_val * i / 5 for i in range(6)]
    grid = []
    ylabels = []
    for val in y_ticks:
        y = scale(val, 0, max_val, mt + ch, mt)
        grid.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + cw}" y2="{y:.1f}" stroke="#e6e0d6" />')
        ylabels.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#263238">{val:.0f}</text>')
    legend = []
    for i, (name, _, color) in enumerate(series):
        legend.append(f'<rect x="{width - 190}" y="{86 + i * 24}" width="16" height="12" fill="{color}" opacity="0.88" />')
        legend.append(f'<text x="{width - 166}" y="{97 + i * 24}" font-size="12" fill="#263238">{name}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{ml}" y="34" font-size="22" font-weight="700" fill="#263238">E5-b grid-connected annual comparison</text>
  <text x="{ml}" y="54" font-size="13" fill="#687076">RBC proxy, offline MILP, and rolling-horizon MPC</text>
  {''.join(grid)}
  {''.join(bars)}
  <line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#333" />
  <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#333" />
  {''.join(ylabels)}
  {''.join(xlabels)}
  {''.join(legend)}
</svg>
"""
    COMPARE_SVG.write_text(svg, encoding="utf-8")


def write_summary_md(summary: dict, monthly: pd.DataFrame, rbc: dict | None, milp: dict | None) -> None:
    lines = [
        "# E5-b Full-Year Grid-Connected MPC Summary",
        "",
        "Battery is fixed at `215 kWh / 50 kW`; PV uses `pv_floating_disturbed_raw_kW` only.",
        "Grid charging is disabled; the battery charges only from PV.",
        "",
        "## Annual MPC Result",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Horizon | {summary['horizon_hours']:.1f} h |",
        f"| Control step | {summary['control_step_hours']:.2f} h |",
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
    if rbc or milp:
        lines.extend(["## Strategy Comparison", "", "| Metric | RBC | MILP | MPC |", "| --- | ---: | ---: | ---: |"])
        for key, label, unit in [
            ("grid_import_kWh", "Grid import", "kWh"),
            ("grid_export_kWh", "Grid export", "kWh"),
            ("net_grid_cost_CNY", "Net grid cost", "CNY"),
        ]:
            rbc_val = (rbc or {}).get(key, math.nan)
            milp_val = (milp or {}).get(key, math.nan)
            mpc_val = summary.get(key, math.nan)
            lines.append(f"| {label} | {rbc_val:.1f} {unit} | {milp_val:.1f} {unit} | {mpc_val:.1f} {unit} |")
        lines.append("")
    lines.extend(
        [
            "## Figures",
            "",
            "- `plots/mpc_e5b_full_year_soc.svg`",
            "- `plots/mpc_e5b_monthly_import_export_cost.svg`",
            "- `plots/mpc_e5b_rbc_milp_mpc_grid_cost.svg`",
            "- `plots/mpc_e5b_progress.svg`",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E5-b full-year grid-connected rolling-horizon MPC.")
    parser.add_argument("--solver", choices=["gurobi", "cbc"], default="gurobi")
    parser.add_argument("--only-months", nargs="*", type=int, choices=list(range(1, 13)), help="Only run selected months.")
    parser.add_argument("--horizon-steps", type=int, default=96, help="MPC horizon in 15-minute steps. Default 96 = 24 h.")
    parser.add_argument("--control-step", type=int, default=1, help="How many optimized steps to execute before re-solving. Default 1 = 15 min.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for testing, e.g. --max-steps 96.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output/mpc_e5b_full_year_grid_dispatch_2023.csv if present.")
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument("--time-limit", type=int, default=None, help="Optional solver time limit per MPC horizon in seconds.")
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--curtailment-penalty", type=float, default=0.001)
    parser.add_argument("--cycle-penalty", type=float, default=0.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MpcGridConfig(
        solver=args.solver,
        horizon_steps=args.horizon_steps,
        control_step=args.control_step,
        curtailment_penalty=args.curtailment_penalty,
        cycle_penalty=args.cycle_penalty,
        solver_time_limit_s=args.time_limit,
        mip_gap=args.mip_gap,
        solver_log=args.solver_log,
    )
    run_e5b(config, only_months=args.only_months, max_steps=args.max_steps, resume=args.resume)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)

"""
Rule-based battery dispatch for the 15-minute disturbed PV scenario.

The controller is intentionally simple:
- PV surplus charges the battery first; remaining surplus is curtailed.
- PV deficit is supplied by the battery first; remaining deficit is unmet load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

DISPATCH_INPUT_CSV = OUT_DIR / "dispatch_input_15min_disturbed_2023.csv"
RBC_OUTPUT_CSV = OUT_DIR / "dispatch_rbc_15min_raw_battery215_2023.csv"
RBC_SUMMARY_CSV = OUT_DIR / "dispatch_rbc_15min_raw_battery215_summary.csv"

DT_HOURS = 0.25


@dataclass(frozen=True)
class BatteryParams:
    capacity_kWh: float = 215.0
    power_kW: float = 50.0
    roundtrip_efficiency: float = 0.92
    initial_soc_fraction: float = 0.50
    min_soc_fraction: float = 0.00
    max_soc_fraction: float = 1.00

    @property
    def charge_efficiency(self) -> float:
        return float(np.sqrt(self.roundtrip_efficiency))

    @property
    def discharge_efficiency(self) -> float:
        return float(np.sqrt(self.roundtrip_efficiency))

    @property
    def min_soc_kWh(self) -> float:
        return self.capacity_kWh * self.min_soc_fraction

    @property
    def max_soc_kWh(self) -> float:
        return self.capacity_kWh * self.max_soc_fraction

    @property
    def initial_soc_kWh(self) -> float:
        return self.capacity_kWh * self.initial_soc_fraction


def _max_continuous_hours(power_kw: pd.Series, threshold: float = 1e-6) -> float:
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


def run_rbc(
    df: pd.DataFrame,
    battery: BatteryParams,
    pv_col: str = "pv_floating_disturbed_raw_kW",
) -> pd.DataFrame:
    required = {"datetime", "load_kW", pv_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dispatch input missing columns: {sorted(missing)}")

    load = df["load_kW"].to_numpy(dtype=float)
    pv = df[pv_col].to_numpy(dtype=float)
    n = len(df)

    soc = np.empty(n, dtype=float)
    charge = np.zeros(n, dtype=float)
    discharge = np.zeros(n, dtype=float)
    curtailment = np.zeros(n, dtype=float)
    unmet = np.zeros(n, dtype=float)

    soc_now = battery.initial_soc_kWh

    for i in range(n):
        net = pv[i] - load[i]

        if net >= 0.0:
            max_charge_by_power = battery.power_kW
            max_charge_by_capacity = (battery.max_soc_kWh - soc_now) / (
                battery.charge_efficiency * DT_HOURS
            )
            charge_kw = max(0.0, min(net, max_charge_by_power, max_charge_by_capacity))
            soc_now += charge_kw * battery.charge_efficiency * DT_HOURS
            charge[i] = charge_kw
            curtailment[i] = max(0.0, net - charge_kw)
        else:
            deficit = -net
            max_discharge_by_power = battery.power_kW
            max_discharge_by_soc = (soc_now - battery.min_soc_kWh) * battery.discharge_efficiency / DT_HOURS
            discharge_kw = max(0.0, min(deficit, max_discharge_by_power, max_discharge_by_soc))
            soc_now -= discharge_kw / battery.discharge_efficiency * DT_HOURS
            discharge[i] = discharge_kw
            unmet[i] = max(0.0, deficit - discharge_kw)

        soc_now = min(max(soc_now, battery.min_soc_kWh), battery.max_soc_kWh)
        soc[i] = soc_now

    out = df.copy()
    out["pv_kW"] = pv
    out["battery_soc_kWh"] = soc
    out["battery_soc_fraction"] = soc / battery.capacity_kWh
    out["battery_charge_kW"] = charge
    out["battery_discharge_kW"] = discharge
    out["curtailment_kW"] = curtailment
    out["unmet_load_kW"] = unmet
    out["served_load_kW"] = load - unmet
    return out


def summarize_dispatch(out: pd.DataFrame, battery: BatteryParams) -> pd.DataFrame:
    load_kwh = float(out["load_kW"].sum() * DT_HOURS)
    pv_kwh = float(out["pv_kW"].sum() * DT_HOURS)
    served_kwh = float(out["served_load_kW"].sum() * DT_HOURS)
    unmet_kwh = float(out["unmet_load_kW"].sum() * DT_HOURS)
    curtail_kwh = float(out["curtailment_kW"].sum() * DT_HOURS)
    charge_kwh = float(out["battery_charge_kW"].sum() * DT_HOURS)
    discharge_kwh = float(out["battery_discharge_kW"].sum() * DT_HOURS)
    direct_pv_used_kwh = float(np.minimum(out["load_kW"], out["pv_kW"]).sum() * DT_HOURS)
    pv_used_kwh = pv_kwh - curtail_kwh

    summary = {
        "battery_capacity_kWh": battery.capacity_kWh,
        "battery_power_kW": battery.power_kW,
        "roundtrip_efficiency": battery.roundtrip_efficiency,
        "initial_soc_fraction": battery.initial_soc_fraction,
        "load_kWh": load_kwh,
        "pv_kWh": pv_kwh,
        "served_load_kWh": served_kwh,
        "unmet_load_kWh": unmet_kwh,
        "curtailment_kWh": curtail_kwh,
        "battery_charge_kWh_ac": charge_kwh,
        "battery_discharge_kWh_ac": discharge_kwh,
        "direct_pv_used_kWh": direct_pv_used_kwh,
        "pv_used_kWh": pv_used_kwh,
        "load_served_fraction": served_kwh / load_kwh if load_kwh else np.nan,
        "self_sufficiency_fraction": served_kwh / load_kwh if load_kwh else np.nan,
        "pv_self_consumption_fraction": pv_used_kwh / pv_kwh if pv_kwh else np.nan,
        "curtailment_fraction": curtail_kwh / pv_kwh if pv_kwh else np.nan,
        "unmet_load_fraction": unmet_kwh / load_kwh if load_kwh else np.nan,
        "min_soc_kWh": float(out["battery_soc_kWh"].min()),
        "max_soc_kWh": float(out["battery_soc_kWh"].max()),
        "end_soc_kWh": float(out["battery_soc_kWh"].iloc[-1]),
        "max_continuous_unmet_hours": _max_continuous_hours(out["unmet_load_kW"]),
    }
    return pd.DataFrame([summary])


def plot_dispatch(out: pd.DataFrame, summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Plotting skipped: matplotlib is not installed in this Python environment.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = out.copy()
    plot_df["date"] = plot_df["datetime"].dt.date
    plot_df["month"] = plot_df["datetime"].dt.month

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(plot_df["datetime"], plot_df["battery_soc_kWh"], color="#2f6f73", linewidth=0.7)
    ax.set_title("Battery SOC - RBC 15-min raw disturbed PV (215 kWh)")
    ax.set_xlabel("Date")
    ax.set_ylabel("SOC (kWh)")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "dispatch_rbc_15min_raw_battery215_soc.png", dpi=150)
    plt.close(fig)

    target_day = plot_df.groupby("date")["unmet_load_kW"].sum().idxmax()
    day = plot_df[plot_df["date"] == target_day]
    hour = day["datetime"].dt.hour + day["datetime"].dt.minute / 60.0

    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    ax2.plot(hour, day["load_kW"], label="Load", color="#334e68", linewidth=1.8)
    ax2.plot(hour, day["pv_kW"], label="PV raw disturbed", color="#d99000", linewidth=1.8)
    ax2.bar(hour, day["battery_charge_kW"], width=0.08, label="Charge", color="#5aa469", alpha=0.65)
    ax2.bar(hour, -day["battery_discharge_kW"], width=0.08, label="Discharge", color="#c44e52", alpha=0.65)
    ax2.set_title(f"Worst unmet-load day dispatch ({target_day})")
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("kW")
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 4))
    ax2.grid(True, alpha=0.25)
    ax2.legend(ncols=4, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "dispatch_rbc_15min_raw_battery215_worst_day.png", dpi=150)
    plt.close(fig2)

    monthly = plot_df.groupby("month")[["load_kW", "pv_kW", "curtailment_kW", "unmet_load_kW"]].sum() * DT_HOURS
    x = np.arange(len(monthly.index))
    width = 0.22
    fig3, ax3 = plt.subplots(figsize=(10, 4.8))
    ax3.bar(x - width, monthly["load_kW"], width=width, label="Load", color="#334e68")
    ax3.bar(x, monthly["pv_kW"], width=width, label="PV", color="#d99000")
    ax3.bar(x + width, monthly["curtailment_kW"], width=width, label="Curtailment", color="#8e6c8a")
    ax3.plot(x, monthly["unmet_load_kW"], label="Unmet load", color="#c44e52", marker="o", linewidth=1.6)
    ax3.set_title("Monthly energy summary - RBC 15-min raw disturbed PV")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("kWh")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(m) for m in monthly.index])
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(ncols=4, fontsize=8)
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "dispatch_rbc_15min_raw_battery215_monthly.png", dpi=150)
    plt.close(fig3)


def main() -> None:
    if not DISPATCH_INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {DISPATCH_INPUT_CSV}; run build_pv_disturbance_15min.py first.")

    battery = BatteryParams()
    df = pd.read_csv(DISPATCH_INPUT_CSV, parse_dates=["datetime"])
    out = run_rbc(df, battery)
    summary = summarize_dispatch(out, battery)
    plot_dispatch(out, summary)

    out.to_csv(RBC_OUTPUT_CSV, index=False)
    summary.to_csv(RBC_SUMMARY_CSV, index=False)

    row = summary.iloc[0]
    print(f"Battery: {battery.capacity_kWh:.0f} kWh / {battery.power_kW:.0f} kW")
    print(f"Load: {row['load_kWh']:,.1f} kWh")
    print(f"PV: {row['pv_kWh']:,.1f} kWh")
    print(f"Unmet load: {row['unmet_load_kWh']:,.1f} kWh ({row['unmet_load_fraction'] * 100:.2f}%)")
    print(f"Curtailment: {row['curtailment_kWh']:,.1f} kWh ({row['curtailment_fraction'] * 100:.2f}% of PV)")
    print(f"PV self-consumption: {row['pv_self_consumption_fraction'] * 100:.2f}%")
    print(f"Min SOC: {row['min_soc_kWh']:,.1f} kWh")
    print(f"Max continuous unmet duration: {row['max_continuous_unmet_hours']:.2f} h")
    print(f"Saved: {RBC_OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Saved: {RBC_SUMMARY_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

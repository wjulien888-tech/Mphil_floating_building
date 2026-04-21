"""
未调度场景：将全年逐日光伏发电与逐日负荷画在同一张图里，并可视化逐日差值 (PV - Load)。

输入：
  output/pv_generation_8760.csv（datetime, pv_kW）
  output/load_8760.csv（datetime, load_kW）

输出：
  plots/pv_load_daily_compare.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"
PV_CSV = OUT_DIR / "pv_generation_8760.csv"
LOAD_CSV = OUT_DIR / "load_8760.csv"
FIG_PATH = PLOTS_DIR / "pv_load_daily_compare.png"
WEEKLY_FIG_PATH = PLOTS_DIR / "pv_load_weekly_hourly_overlay.png"
WEEK_START = "2023-07-17"
WEEK_END = "2023-07-23"


def main() -> None:
    if not PV_CSV.exists():
        raise FileNotFoundError(f"缺少 {PV_CSV}，请先运行 pv_generation.py")
    if not LOAD_CSV.exists():
        raise FileNotFoundError(f"缺少 {LOAD_CSV}，请先运行 load_generation.py")

    pv = pd.read_csv(PV_CSV, parse_dates=["datetime"])
    load = pd.read_csv(LOAD_CSV, parse_dates=["datetime"])

    if len(pv) != len(load):
        raise ValueError(f"行数不一致: PV {len(pv)} vs load {len(load)}")
    if not pv["datetime"].equals(load["datetime"]):
        raise ValueError("datetime 列与 load 不对齐，请检查两个 CSV 是否同一套时间轴")

    m = pd.DataFrame(
        {
            "datetime": pv["datetime"],
            "pv_kW": pv["pv_kW"].to_numpy(dtype=float),
            "load_kW": load["load_kW"].to_numpy(dtype=float),
        }
    )
    m["date"] = m["datetime"].dt.normalize()
    daily = m.groupby("date", as_index=False).agg(pv_kWh=("pv_kW", "sum"), load_kWh=("load_kW", "sum"))
    daily["net_kWh"] = daily["pv_kWh"] - daily["load_kWh"]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(14, 7.5),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    d = daily["date"]
    y_pv = daily["pv_kWh"].to_numpy()
    y_ld = daily["load_kWh"].to_numpy()

    ax_top.plot(d, y_pv, color="#ff7f0e", lw=1.1, label="Daily PV (kWh/day)")
    ax_top.plot(d, y_ld, color="#1f77b4", lw=1.1, label="Daily load (kWh/day)")

    # 差值填色：绿 = 光伏多于负荷；红 = 负荷多于光伏
    ax_top.fill_between(
        d,
        y_ld,
        y_pv,
        where=y_pv >= y_ld,
        interpolate=True,
        color="#2ca02c",
        alpha=0.22,
        label="PV surplus (shaded)",
    )
    ax_top.fill_between(
        d,
        y_pv,
        y_ld,
        where=y_pv < y_ld,
        interpolate=True,
        color="#d62728",
        alpha=0.22,
        label="Load deficit (shaded)",
    )

    ax_top.set_ylabel("kWh/day")
    ax_top.set_title("Full-year daily PV vs load (no dispatch)")
    ax_top.legend(loc="upper right", fontsize=8, ncol=2)
    ax_top.grid(True, alpha=0.25)

    net = daily["net_kWh"].to_numpy()
    colors = np.where(net >= 0.0, "#2ca02c", "#d62728")
    ax_bot.bar(d, net, width=1.0, color=colors, edgecolor="none", alpha=0.85)
    ax_bot.axhline(0.0, color="0.35", lw=0.8)
    ax_bot.set_ylabel("kWh/day")
    ax_bot.set_xlabel("Date")
    ax_bot.set_title("Daily net energy: PV minus load (no dispatch)")
    ax_bot.grid(True, axis="y", alpha=0.25)

    for ax in (ax_top, ax_bot):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax_bot.get_xticklabels(), rotation=0, ha="center")

    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)

    # 小时级周叠加图：展示夜间负荷与连续阴雨导致的多日缺口更直观。
    week_start = pd.Timestamp(WEEK_START)
    week_end = pd.Timestamp(WEEK_END) + pd.Timedelta(days=1)
    week = m[(m["datetime"] >= week_start) & (m["datetime"] < week_end)].copy()
    if week.empty:
        raise ValueError(f"周窗口无数据: {WEEK_START} ~ {WEEK_END}")

    fig_w, ax_w = plt.subplots(figsize=(14, 4.8), layout="constrained")
    ax_w.plot(
        week["datetime"],
        week["pv_kW"],
        color="#ff7f0e",
        lw=1.2,
        label="PV (kW, hourly)",
    )
    ax_w.plot(
        week["datetime"],
        week["load_kW"],
        color="#1f77b4",
        lw=1.2,
        label="Load (kW, hourly)",
    )

    y_pv_h = week["pv_kW"].to_numpy()
    y_ld_h = week["load_kW"].to_numpy()
    x_h = week["datetime"]
    ax_w.fill_between(
        x_h,
        y_ld_h,
        y_pv_h,
        where=y_pv_h >= y_ld_h,
        interpolate=True,
        color="#2ca02c",
        alpha=0.20,
        label="Hourly surplus",
    )
    ax_w.fill_between(
        x_h,
        y_pv_h,
        y_ld_h,
        where=y_pv_h < y_ld_h,
        interpolate=True,
        color="#d62728",
        alpha=0.20,
        label="Hourly deficit",
    )

    ax_w.set_title(f"Hourly PV vs load overlay ({WEEK_START} to {WEEK_END})")
    ax_w.set_ylabel("kW")
    ax_w.set_xlabel("Datetime")
    ax_w.grid(True, alpha=0.25)
    ax_w.legend(loc="upper right", ncol=2, fontsize=8)
    ax_w.xaxis.set_major_locator(mdates.DayLocator())
    ax_w.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax_w.get_xticklabels(), rotation=0, ha="center")
    fig_w.savefig(WEEKLY_FIG_PATH, dpi=150)
    plt.close(fig_w)

    n_surplus = int((net > 0).sum())
    n_deficit = int((net < 0).sum())
    print(f"Saved: {FIG_PATH}")
    print(
        f"Annual PV: {y_pv.sum():,.1f} kWh | Annual load: {y_ld.sum():,.1f} kWh | "
        f"Net (PV-load): {net.sum():,.1f} kWh"
    )
    print(f"Days with PV > load: {n_surplus} | Days with PV < load: {n_deficit}")
    print(f"Saved: {WEEKLY_FIG_PATH}")


if __name__ == "__main__":
    main()

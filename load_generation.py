"""
任务 2（重构版）：2023 全年逐时建筑负荷 (kW)，与 output/pv_generation_8760.csv 时间戳逐行对齐。

负荷分项建模（三段式）：
    load_kW = base_kW + hvac_kW + aux_kW

- base_kW  : 基础负荷（照明、普通插座、机房等），全年都有；日程=ASHRAE 工作日/周末
- hvac_kW  : 空调负荷，仅空调季 4-10 月；月强度 + 日内白天强度调制
- aux_kW   : 附属负荷（水泵、室外照明等），傍晚夜间较重

标定口径（与设计院资料一致）：
- 全年总电量  = 60,461 kWh （PDF 参考值）
- HVAC 占比   ≈ 60%
- 基础占比    ≈ 30%
- 附属占比    ≈ 10%

节假日：
- holidays 中国法定假日 + 国务院办公厅 2023 年放假调休日期取并集
- 国务院补班日 → 强制使用工作日形状
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"
PV_CSV = OUT_DIR / "pv_generation_8760.csv"
LOAD_CSV = OUT_DIR / "load_8760.csv"

# --- 目标年电量（与设计院 PDF 参考一致） ---
TARGET_ANNUAL_KWH = 60_461.0
SHARE_HVAC = 0.60
SHARE_BASE = 0.30
SHARE_AUX = 0.10

TARGET_HVAC_KWH = TARGET_ANNUAL_KWH * SHARE_HVAC
TARGET_BASE_KWH = TARGET_ANNUAL_KWH * SHARE_BASE
TARGET_AUX_KWH = TARGET_ANNUAL_KWH * SHARE_AUX

# --- 南沙 HVAC 季节性 ---
# 11-3 月不开空调；4-10 月开空调（用户口径）。月份调制用于描述“刚开机/末季过渡”。
HVAC_MONTH_SHAPE = {
    1: 0.00, 2: 0.00, 3: 0.00, 11: 0.00, 12: 0.00,
    4: 0.30, 10: 0.40,
    5: 0.70, 9: 0.85,
    6: 0.95,
    7: 1.00, 8: 1.00,
}

# --- 日内形状（仅表示曲线形状，绝对量级由最终缩放决定） ---
# 基础负荷（与原 ASHRAE 90.1 工作日/周末系数一致）
SHAPE_BASE_WEEKDAY = np.array(
    [
        0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
        0.50, 0.70, 1.00, 1.00, 1.00, 0.80,
        1.00, 1.00, 1.00, 1.00, 0.80, 0.60,
        0.50, 0.40, 0.35, 0.30, 0.30, 0.30,
    ],
    dtype=float,
)
SHAPE_BASE_WEEKEND = np.array(
    [
        0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
        0.25, 0.30, 0.40, 0.40, 0.40, 0.35,
        0.40, 0.40, 0.40, 0.35, 0.30, 0.25,
        0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
    ],
    dtype=float,
)

# HVAC 日内形状：白天强，夜间为 0
SHAPE_HVAC_WEEKDAY = np.array(
    [
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.10, 0.30, 0.60, 0.80, 0.90, 0.95,
        1.00, 1.00, 1.00, 0.95, 0.85, 0.60,
        0.30, 0.10, 0.00, 0.00, 0.00, 0.00,
    ],
    dtype=float,
)
SHAPE_HVAC_WEEKEND = np.array(
    [
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.10, 0.20, 0.30, 0.40, 0.45,
        0.50, 0.50, 0.45, 0.40, 0.30, 0.20,
        0.10, 0.00, 0.00, 0.00, 0.00, 0.00,
    ],
    dtype=float,
)

# 附属负荷日内形状：傍晚/夜间为主（室外照明 + 水泵轮换）
SHAPE_AUX_ALLDAY = np.array(
    [
        0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
        0.60, 0.40, 0.30, 0.30, 0.30, 0.30,
        0.30, 0.30, 0.30, 0.30, 0.40, 0.50,
        0.70, 0.90, 1.00, 1.00, 0.80, 0.60,
    ],
    dtype=float,
)

# --- 节假日表（沿用之前版本） ---
OFFICIAL_REST_2023: frozenset[date] = frozenset(
    {
        date(2022, 12, 31),
        date(2023, 1, 1),
        date(2023, 1, 2),
        *{date(2023, 1, d) for d in range(21, 28)},
        date(2023, 4, 5),
        *{date(2023, 4, d) for d in range(29, 31)},
        date(2023, 5, 1),
        date(2023, 5, 2),
        date(2023, 5, 3),
        date(2023, 6, 22),
        date(2023, 6, 23),
        date(2023, 6, 24),
        date(2023, 9, 29),
        date(2023, 9, 30),
        *{date(2023, 10, d) for d in range(1, 7)},
    }
)

MAKEUP_WORKDAYS_2023: frozenset[date] = frozenset(
    {
        date(2023, 1, 28),
        date(2023, 1, 29),
        date(2023, 4, 23),
        date(2023, 5, 6),
        date(2023, 6, 25),
        date(2023, 10, 7),
        date(2023, 10, 8),
    }
)


def _legal_holiday_dates_2023() -> set[date]:
    cn = holidays.country_holidays("CN", years=[2023])
    return {d for d in cn.keys() if isinstance(d, date) and d.year == 2023}


def _is_weekend_schedule(d: date, legal_cn: set[date]) -> bool:
    """返回 True 表示该日应使用 '周末/节假日' 形状（否则使用工作日形状）。"""
    if d in MAKEUP_WORKDAYS_2023:
        return False
    if d in OFFICIAL_REST_2023 or d in legal_cn or d.weekday() >= 5:
        return True
    return False


def _raw_profiles(datetimes: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (base_shape, hvac_shape, aux_shape)：三条未缩放的形状序列。"""
    n = len(datetimes)
    base = np.empty(n, dtype=float)
    hvac = np.empty(n, dtype=float)
    aux = np.empty(n, dtype=float)

    legal_cn = _legal_holiday_dates_2023()
    dt = pd.to_datetime(datetimes)
    hours = dt.dt.hour.to_numpy()
    months = dt.dt.month.to_numpy()
    dates = dt.dt.date.to_numpy()

    for i in range(n):
        h = int(hours[i])
        m = int(months[i])
        d = dates[i]
        is_we = _is_weekend_schedule(d, legal_cn)

        base[i] = SHAPE_BASE_WEEKEND[h] if is_we else SHAPE_BASE_WEEKDAY[h]
        month_k = HVAC_MONTH_SHAPE.get(m, 0.0)
        hvac[i] = month_k * (SHAPE_HVAC_WEEKEND[h] if is_we else SHAPE_HVAC_WEEKDAY[h])
        aux[i] = SHAPE_AUX_ALLDAY[h]

    return base, hvac, aux


def build_load(datetimes: pd.Series) -> pd.DataFrame:
    """生成分项负荷并校准到目标年电量与占比。"""
    base_shape, hvac_shape, aux_shape = _raw_profiles(datetimes)

    # 1 小时步长：sum(kW) == 年 kWh
    def scale_to(target_kwh: float, shape: np.ndarray) -> tuple[np.ndarray, float]:
        total = float(shape.sum())
        if total <= 0.0:
            return np.zeros_like(shape), 0.0
        factor = target_kwh / total
        return shape * factor, factor

    base_kw, _ = scale_to(TARGET_BASE_KWH, base_shape)
    hvac_kw, _ = scale_to(TARGET_HVAC_KWH, hvac_shape)
    aux_kw, _ = scale_to(TARGET_AUX_KWH, aux_shape)

    total_kw = base_kw + hvac_kw + aux_kw

    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(datetimes).to_numpy(),
            "base_kW": base_kw,
            "hvac_kW": hvac_kw,
            "aux_kW": aux_kw,
            "load_kW": total_kw,
        }
    )


# ---------------- 绘图 ----------------
_MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_daily_monthly(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date", as_index=False)["load_kW"].sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["load_kW"], lw=0.8, color="C2")
    ax.set_title("Daily load energy (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "load_daily_kwh.png", dpi=150)
    plt.close(fig)

    df["month"] = df["datetime"].dt.month
    monthly = df.groupby("month", as_index=False)[["base_kW", "hvac_kW", "aux_kW"]].sum()

    fig2, ax2 = plt.subplots(figsize=(9, 4.2))
    ax2.bar(monthly["month"], monthly["base_kW"], label="Base", color="#1f77b4")
    ax2.bar(
        monthly["month"],
        monthly["hvac_kW"],
        bottom=monthly["base_kW"],
        label="HVAC",
        color="#d62728",
    )
    ax2.bar(
        monthly["month"],
        monthly["aux_kW"],
        bottom=monthly["base_kW"] + monthly["hvac_kW"],
        label="Auxiliary",
        color="#9467bd",
    )
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(_MONTHS_EN)
    ax2.set_title("Monthly load energy by component (kWh/month)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("kWh")
    ax2.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "load_monthly_kwh.png", dpi=150)
    plt.close(fig2)


def plot_daily_full_year(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = df["datetime"].dt.normalize()
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    daily = df.groupby("date", as_index=False)["load_kW"].sum()

    month_of_day = pd.to_datetime(daily["date"]).dt.month.to_numpy()
    cmap = plt.get_cmap("tab20")
    colors = [cmap((m - 1) % 20) for m in month_of_day]

    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.bar(daily["date"], daily["load_kW"], width=1.0, color=colors, edgecolor="none")
    ax.set_title("Daily load energy for the full year (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")
    ax.margins(x=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "load_daily_full_year_bar.png", dpi=150)
    plt.close(fig)

    grid = np.full((12, 31), np.nan, dtype=float)
    by_md = df.groupby(["month", "day"])["load_kW"].sum()
    for (m, d), v in by_md.items():
        grid[m - 1, d - 1] = v

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    im = ax2.imshow(grid, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    ax2.set_yticks(range(12))
    ax2.set_yticklabels(_MONTHS_EN)
    ax2.set_xticks(range(0, 31, 2))
    ax2.set_xticklabels([str(i + 1) for i in range(0, 31, 2)])
    ax2.set_xlabel("Day of month")
    ax2.set_title("Daily load calendar heatmap (kWh/day)")
    cbar = fig2.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label("kWh/day")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "load_daily_full_year_heatmap.png", dpi=150)
    plt.close(fig2)


def plot_typical_day_stack(df: pd.DataFrame) -> None:
    """画两个典型日的分项叠加图：冬季代表日 (1/18, 无空调) 与夏季代表日 (7/19, 满空调)。"""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = df["datetime"].dt.date

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, target_date, title in (
        (axes[0], date(2023, 1, 18), "Typical winter day (2023-01-18)"),
        (axes[1], date(2023, 7, 19), "Typical summer day (2023-07-19)"),
    ):
        day = df[df["date"] == target_date].copy()
        hours = pd.to_datetime(day["datetime"]).dt.hour.to_numpy()
        ax.stackplot(
            hours,
            day["base_kW"].to_numpy(),
            day["hvac_kW"].to_numpy(),
            day["aux_kW"].to_numpy(),
            labels=["Base", "HVAC", "Auxiliary"],
            colors=["#1f77b4", "#d62728", "#9467bd"],
        )
        ax.set_title(title)
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("kW")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "load_typical_day_stack.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not PV_CSV.exists():
        raise FileNotFoundError(f"未找到 {PV_CSV}，请先运行 pv_generation.py")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pv = pd.read_csv(PV_CSV, parse_dates=["datetime"])
    if len(pv) != 8760:
        print(f"警告：期望 8760 行，实际 {len(pv)} 行")

    out = build_load(pv["datetime"])
    out.to_csv(LOAD_CSV, index=False)

    annual = float(out["load_kW"].sum())
    e_base = float(out["base_kW"].sum())
    e_hvac = float(out["hvac_kW"].sum())
    e_aux = float(out["aux_kW"].sum())

    print(f"年总负荷:     {annual:,.1f} kWh   (目标 {TARGET_ANNUAL_KWH:,.1f} kWh)")
    print(f"  - 基础:    {e_base:,.1f} kWh ({e_base / annual * 100:.1f} %, 目标 {SHARE_BASE * 100:.0f} %)")
    print(f"  - HVAC:    {e_hvac:,.1f} kWh ({e_hvac / annual * 100:.1f} %, 目标 {SHARE_HVAC * 100:.0f} %)")
    print(f"  - 附属:    {e_aux:,.1f} kWh ({e_aux / annual * 100:.1f} %, 目标 {SHARE_AUX * 100:.0f} %)")
    print(
        "说明：分项建模（基础 + HVAC + 附属），按目标年电量与分项占比各自缩放；"
        "HVAC 仅 4-10 月、白天为主；节假日按周末形状，国务院补班按工作日形状。"
    )

    plot_daily_monthly(out)
    plot_daily_full_year(out)
    plot_typical_day_stack(out)

    print(f"已保存: {LOAD_CSV}")
    print(
        "图表: "
        f"{PLOTS_DIR / 'load_daily_kwh.png'}, "
        f"{PLOTS_DIR / 'load_monthly_kwh.png'}, "
        f"{PLOTS_DIR / 'load_daily_full_year_bar.png'}, "
        f"{PLOTS_DIR / 'load_daily_full_year_heatmap.png'}, "
        f"{PLOTS_DIR / 'load_typical_day_stack.png'}"
    )


if __name__ == "__main__":
    main()

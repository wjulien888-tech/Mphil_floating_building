"""
Task 1: PVGIS -> hourly PV generation (kW) for year 2023 (8760 h).
Reads CSVs from data/, writes output/pv_generation_8760.csv and plots to plots/.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Paths (project layout) ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

# --- PVGIS ---
PVGIS_SKIPROWS = 8  # metadata lines before header row with 'time'
TIME_COL = "time"

# --- Building BIPV (from PDF) ---
# Sphere: 290 m² total, 4 orientations equal -> 72.5 m² each
SPHERE_M2_PER_ORIENT = 290.0 / 4.0
OPAQUE_FRAC = 250.0 / 290.0  # HJT on opaque part of sphere
TRANS_FRAC = 40.0 / 290.0  # CdTe on transparent part
EFF_HJT = 0.21
EFF_CDTE = 0.12

# Per orientation: split sphere area into opaque / transparent patches
SPHERE_OPAQUE_M2 = SPHERE_M2_PER_ORIENT * OPAQUE_FRAC
SPHERE_TRANS_M2 = SPHERE_M2_PER_ORIENT * TRANS_FRAC

# Horizontal (PDF): use pvgis_deck.csv; deck HJT, glass balustrade CdTe
DECK_M2 = 67.6
BALU_M2 = 62.4

# Water surface microclimate (apply to combined hourly power)
ALBEDO_FACTOR = 1.05
COOLING_FACTOR = 1.012

DESIGN_ANNUAL_KWH = 55_580

TILT_FILES = {
    "south": "pvgis_south_45.csv",
    "east": "pvgis_east_45.csv",
    "west": "pvgis_west_45.csv",
    "north": "pvgis_north_45.csv",
}
DECK_FILE = "pvgis_deck.csv"
TOP_FILE = "pvgis_top.csv"


def load_pvgis_csv(path: Path) -> pd.DataFrame:
    """Load PVGIS hourly file: skip metadata, parse time column."""
    df = pd.read_csv(path, sep=",", skiprows=PVGIS_SKIPROWS)
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing '{TIME_COL}' in {path}")
    # Footer lines after data (PVGIS column descriptions) are not valid timestamps
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="%Y%m%d:%H%M", errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def temp_factor_c(t_c: pd.Series | np.ndarray) -> np.ndarray:
    """Linear temperature correction: 1 - 0.004 * (T - 25)."""
    t = np.asarray(t_c, dtype=float)
    return 1.0 + (-0.004) * (t - 25.0)


def power_kw_one_surface(
    g_w_m2: np.ndarray,
    t_c: np.ndarray,
    area_m2: float,
    efficiency: float,
) -> np.ndarray:
    """Irradiance (W/m2) -> DC power (kW) for one homogeneous surface."""
    g = np.asarray(g_w_m2, dtype=float)
    tf = temp_factor_c(t_c)
    power = g * area_m2 * efficiency * tf / 1000.0
    return np.maximum(0.0, power)


def total_irradiance_w_m2(df: pd.DataFrame) -> np.ndarray:
    return (
        df["Gb(i)"].to_numpy(dtype=float)
        + df["Gd(i)"].to_numpy(dtype=float)
        + df["Gr(i)"].to_numpy(dtype=float)
    )


def build_hourly_pv() -> pd.DataFrame:
    # Load tilted orientations
    frames: dict[str, pd.DataFrame] = {}
    for name, fname in TILT_FILES.items():
        p = DATA_DIR / fname
        if not p.exists():
            raise FileNotFoundError(p)
        frames[name] = load_pvgis_csv(p)

    # Align time index across all tilt files
    idx = frames["south"][TIME_COL]
    for k, d in frames.items():
        if not d[TIME_COL].equals(idx):
            raise ValueError(f"Time index mismatch for {k}")

    deck_path = DATA_DIR / DECK_FILE
    top_path = DATA_DIR / TOP_FILE
    if not deck_path.exists():
        raise FileNotFoundError(deck_path)
    deck = load_pvgis_csv(deck_path)

    # Include pvgis_top: PDF deck + top are both horizontal; if identical, irradiance is counted once.
    if top_path.exists():
        top = load_pvgis_csv(top_path)
        if not top[TIME_COL].equals(idx):
            raise ValueError("Time index mismatch for pvgis_top")
        if file_md5(deck_path) != file_md5(top_path):
            raise ValueError(
                "pvgis_top.csv differs from pvgis_deck.csv; review horizontal area split manually."
            )
        print(
            "Note: pvgis_top.csv and pvgis_deck.csv are identical (same MD5). "
            "Horizontal irradiance is applied once to deck + balustrade areas (no double counting)."
        )
    else:
        print("Note: pvgis_top.csv not found; using pvgis_deck.csv only for horizontal surfaces.")

    if not deck[TIME_COL].equals(idx):
        raise ValueError("Time index mismatch for pvgis_deck")

    t2m = deck["T2m"].to_numpy(dtype=float)

    powers: list[np.ndarray] = []

    # Sphere: four orientations, each with opaque HJT + transparent CdTe
    for name in ("south", "east", "west", "north"):
        df = frames[name]
        g = total_irradiance_w_m2(df)
        tcol = df["T2m"].to_numpy(dtype=float)
        powers.append(power_kw_one_surface(g, tcol, SPHERE_OPAQUE_M2, EFF_HJT))
        powers.append(power_kw_one_surface(g, tcol, SPHERE_TRANS_M2, EFF_CDTE))

    # Horizontal: same G for deck (HJT) and balustrade (CdTe)
    g_h = total_irradiance_w_m2(deck)
    powers.append(power_kw_one_surface(g_h, t2m, DECK_M2, EFF_HJT))
    powers.append(power_kw_one_surface(g_h, t2m, BALU_M2, EFF_CDTE))

    raw_sum = np.sum(powers, axis=0)
    pv_kw = raw_sum * ALBEDO_FACTOR * COOLING_FACTOR

    out = pd.DataFrame({TIME_COL: idx, "pv_kW": pv_kw})
    return out


def annual_energy_kwh(series_kw: pd.Series) -> float:
    """8760 hourly average kW -> kWh (1 h timestep)."""
    return float(series_kw.sum())


def plot_daily_monthly(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    dt_col = "datetime" if "datetime" in df.columns else TIME_COL
    df["date"] = df[dt_col].dt.date
    daily = df.groupby("date", as_index=False)["pv_kW"].sum()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["pv_kW"], lw=0.8, color="C1")
    ax.set_title("Daily PV energy (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pv_daily_kwh.png", dpi=150)
    plt.close(fig)

    df["month"] = df[dt_col].dt.month
    monthly = df.groupby("month", as_index=False)["pv_kW"].sum()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(monthly["month"], monthly["pv_kW"], color="C0")
    ax2.set_xticks(range(1, 13))
    ax2.set_title("Monthly PV energy (kWh/month)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("kWh")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "pv_monthly_kwh.png", dpi=150)
    plt.close(fig2)


def plot_daily_full_year(df: pd.DataFrame) -> None:
    """Full-year daily PV views: a 365-day bar chart and a month-by-day calendar heatmap."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    dt_col = "datetime" if "datetime" in df.columns else TIME_COL
    df["date"] = df[dt_col].dt.normalize()
    df["month"] = df[dt_col].dt.month
    df["day"] = df[dt_col].dt.day
    daily = df.groupby("date", as_index=False)["pv_kW"].sum()

    # 1) 365-day bar chart colored by month for quick scanning.
    month_of_day = pd.to_datetime(daily["date"]).dt.month.to_numpy()
    cmap = plt.get_cmap("tab20")
    colors = [cmap((m - 1) % 20) for m in month_of_day]

    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.bar(daily["date"], daily["pv_kW"], width=1.0, color=colors, edgecolor="none")
    ax.set_title("Daily PV energy for the full year (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")
    ax.margins(x=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pv_daily_full_year_bar.png", dpi=150)
    plt.close(fig)

    # 2) Calendar heatmap: rows = month 1..12, columns = day 1..31.
    grid = np.full((12, 31), np.nan, dtype=float)
    by_md = df.groupby(["month", "day"])["pv_kW"].sum()
    for (m, d), v in by_md.items():
        grid[m - 1, d - 1] = v

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    im = ax2.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax2.set_yticks(range(12))
    ax2.set_yticklabels([
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ])
    ax2.set_xticks(range(0, 31, 2))
    ax2.set_xticklabels([str(i + 1) for i in range(0, 31, 2)])
    ax2.set_xlabel("Day of month")
    ax2.set_title("Daily PV energy calendar heatmap (kWh/day)")
    cbar = fig2.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label("kWh/day")
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "pv_daily_full_year_heatmap.png", dpi=150)
    plt.close(fig2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_hourly_pv()
    if len(df) != 8760:
        print(f"Warning: expected 8760 rows, got {len(df)}")

    out_path = OUT_DIR / "pv_generation_8760.csv"
    df_out = df.rename(columns={TIME_COL: "datetime"})
    df_out.to_csv(out_path, index=False)

    e_year = annual_energy_kwh(df["pv_kW"])
    print(f"Annual PV generation (sum of hourly kW): {e_year:,.1f} kWh")
    print(f"Design estimate (PDF):                  {DESIGN_ANNUAL_KWH:,.1f} kWh")
    print(f"Difference vs design:                   {e_year - DESIGN_ANNUAL_KWH:,.1f} kWh")

    df_plot = df.rename(columns={TIME_COL: "datetime"})
    plot_daily_monthly(df_plot)
    plot_daily_full_year(df_plot)
    print(f"Saved: {out_path}")
    print(
        "Plots: "
        f"{PLOTS_DIR / 'pv_daily_kwh.png'}, "
        f"{PLOTS_DIR / 'pv_monthly_kwh.png'}, "
        f"{PLOTS_DIR / 'pv_daily_full_year_bar.png'}, "
        f"{PLOTS_DIR / 'pv_daily_full_year_heatmap.png'}"
    )


if __name__ == "__main__":
    main()

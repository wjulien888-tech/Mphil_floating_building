"""
Generate a 15-minute synthetic building load profile for 2023.

This version uses measured HKUST Clear Water Bay weather data as the weather
driver, then calibrates end-use components to the design annual electricity
consumption. It does not claim measured building load; it creates a transparent
synthetic load profile for PV/weather-aligned dispatch studies.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HKUST_ROOT = ROOT / "HKUST_clea_bay_PV+wind_data" / "Dataset" / "Time series dataset"
MET_ROOT = HKUST_ROOT / "Meteorological dataset"
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

LOAD_15MIN_CSV = OUT_DIR / "load_15min_2023.csv"
WEATHER_15MIN_CSV = OUT_DIR / "hkust_weather_15min_2023.csv"

DT_HOURS = 0.25
YEAR = 2023

# Annual design electricity from the design-institute reference used by the
# current hourly load model.
TARGET_ANNUAL_KWH = 60_461.0

# End-use shares. HVAC remains aligned with the existing design narrative
# (~60%); the remaining 40% is split into always-on/base, plug, lighting, and
# auxiliary equipment for a more explainable 15-minute profile.
END_USE_TARGETS = {
    "base_kW": TARGET_ANNUAL_KWH * 0.12,
    "plug_kW": TARGET_ANNUAL_KWH * 0.13,
    "lighting_kW": TARGET_ANNUAL_KWH * 0.10,
    "hvac_kW": TARGET_ANNUAL_KWH * 0.60,
    "aux_kW": TARGET_ANNUAL_KWH * 0.05,
}

COOLING_MONTHS = {4, 5, 6, 7, 8, 9, 10}
COOLING_BASE_TEMP_C = 24.0
HUMIDITY_BASE_RH = 65.0
IRRADIANCE_BASE_W_M2 = 250.0
MAX_WEATHER_FFILL_STEPS = 4
MIN_WEATHER_OBSERVATIONS = 12
WEATHER_SOURCES = {
    "temperature_C": ("Temperature", "Temp (Degree Celsius)"),
    "relative_humidity_pct": ("Relative Humidity", "RH (%)"),
    "irradiance_W_m2": ("Irradiance", "Irradiance (W/m2)"),
    "rainfall_mm": ("Rainfall", "Rainfall(mm)"),
}
WEATHER_PLACEHOLDERS = {
    "temperature_C": 24.0, "relative_humidity_pct": 65.0,
    "irradiance_W_m2": 0.0, "rainfall_mm": 0.0,
}


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
    return set()


def _is_weekend_schedule(d: date, legal_cn: set[date]) -> bool:
    if d in MAKEUP_WORKDAYS_2023:
        return False
    if d in OFFICIAL_REST_2023 or d in legal_cn or d.weekday() >= 5:
        return True
    return False


def _read_weather_csv(path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", "N/A", ""])
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=["Time"]).set_index("Time")[[value_col]].sort_index()


def aggregate_weather_15min(
    sources: dict[str, pd.Series],
    full_index: pd.DatetimeIndex,
    max_ffill_steps: int = MAX_WEATHER_FFILL_STEPS,
    min_observations: int = MIN_WEATHER_OBSERVATIONS,
) -> pd.DataFrame:
    """Past-only fill with observation counts and explicit invalid placeholders.

    A row denotes [timestamp, timestamp + 15 minutes), available at its end.
    Counts are unique observed minutes. Rain totals are never forward-filled:
    repeating a preceding accumulation would invent a rain measurement.
    """
    out = pd.DataFrame(index=full_index)
    for col, source in sources.items():
        source = source.replace([np.inf, -np.inf], np.nan)
        source = source.groupby(level=0).mean().sort_index()
        grouped = source.resample("15min")
        counts = grouped.count().reindex(full_index, fill_value=0).astype(int)
        values = (grouped.sum(min_count=1) if col == "rainfall_mm" else grouped.mean())
        values = values.reindex(full_index).where(counts >= min_observations)
        if col in {"irradiance_W_m2", "rainfall_mm"}:
            values = values.clip(lower=0)
        if col == "relative_humidity_pct":
            values = values.clip(0, 100)
        missing = values.isna()
        gap_age = missing.groupby((~missing).cumsum()).cumsum().astype(int)
        fill_limit = 0 if col == "rainfall_mm" else max_ffill_steps
        available = values.ffill(limit=fill_limit) if fill_limit else values
        out[col] = available.fillna(WEATHER_PLACEHOLDERS[col])
        out[f"{col}_observation_count"] = counts
        out[f"{col}_raw_missing"] = (counts == 0).astype(int)
        out[f"{col}_partial"] = ((counts > 0) & (counts < 15)).astype(int)
        out[f"{col}_insufficient_coverage"] = missing.astype(int)
        out[f"{col}_gap_age_steps"] = gap_age
        out[f"{col}_filled"] = (missing & available.notna()).astype(int)
        out[f"{col}_placeholder"] = available.isna().astype(int)
        out[f"{col}_valid"] = available.notna().astype(int)
    out["weather_history_valid"] = out[[f"{c}_valid" for c in sources]].all(axis=1).astype(int)
    out.index.name = "datetime"
    return out.reset_index()


def build_weather_15min(year: int = YEAR) -> pd.DataFrame:
    """Aggregate one year, preserving source quality instead of hiding gaps."""
    sources = {
        name: _read_weather_csv(MET_ROOT / folder / f"{folder}_{year}.csv", value)[value]
        for name, (folder, value) in WEATHER_SOURCES.items()
    }
    full_index = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", inclusive="left")
    return aggregate_weather_15min(sources, full_index)


def _weekday_operation(hour_float: np.ndarray) -> np.ndarray:
    points_h = np.array([0, 6, 7.5, 9, 12, 13, 17.5, 19.5, 22, 24], dtype=float)
    points_v = np.array([0.08, 0.08, 0.35, 1.00, 0.85, 1.00, 0.90, 0.35, 0.12, 0.08], dtype=float)
    return np.interp(hour_float, points_h, points_v)


def _weekend_operation(hour_float: np.ndarray) -> np.ndarray:
    points_h = np.array([0, 7, 10, 16, 20, 24], dtype=float)
    points_v = np.array([0.06, 0.06, 0.22, 0.25, 0.12, 0.06], dtype=float)
    return np.interp(hour_float, points_h, points_v)


def _occupancy_and_operation(datetimes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    legal_cn = _legal_holiday_dates_2023()
    dt = pd.to_datetime(datetimes)
    hour_float = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60.0
    dates = dt.dt.date.to_numpy()

    occupancy = np.empty(len(dt), dtype=float)
    operation = np.empty(len(dt), dtype=float)
    for i, d in enumerate(dates):
        if _is_weekend_schedule(d, legal_cn):
            operation[i] = _weekend_operation(np.array([hour_float[i]]))[0]
            occupancy[i] = operation[i] * 0.75
        else:
            operation[i] = _weekday_operation(np.array([hour_float[i]]))[0]
            occupancy[i] = operation[i]
    return occupancy, operation


def _scale_to_energy(raw_kw: np.ndarray, target_kwh: float) -> np.ndarray:
    raw_energy = float(np.nansum(raw_kw) * DT_HOURS)
    if raw_energy <= 0.0:
        return np.zeros_like(raw_kw)
    return raw_kw * (target_kwh / raw_energy)


def load_raw_shapes(weather: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(weather["datetime"])
    occupancy, operation = _occupancy_and_operation(dt)

    temp_c = weather["temperature_C"].to_numpy(dtype=float)
    rh = weather["relative_humidity_pct"].to_numpy(dtype=float)
    irr = weather["irradiance_W_m2"].to_numpy(dtype=float)
    months = dt.dt.month.to_numpy()

    daylight = np.clip(irr / 600.0, 0.0, 1.0)
    cooling_season = np.isin(months, list(COOLING_MONTHS)).astype(float)
    cooling_degree = np.maximum(0.0, temp_c - COOLING_BASE_TEMP_C)
    humidity_term = np.maximum(0.0, rh - HUMIDITY_BASE_RH) / 20.0
    solar_term = np.maximum(0.0, irr - IRRADIANCE_BASE_W_M2) / 500.0

    # Raw shapes are dimensionless before fixed historical calibration.
    base_raw = 0.55 + 0.25 * operation + 0.10 * occupancy
    plug_raw = 0.06 + np.power(np.clip(occupancy, 0.0, 1.0), 1.25)
    lighting_raw = 0.04 + operation * (0.25 + 0.75 * (1.0 - daylight))
    hvac_raw = cooling_season * (0.25 + 0.75 * operation) * (
        cooling_degree + 0.75 * humidity_term + 0.60 * solar_term
    )

    hour_float = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60.0
    night_lighting = ((hour_float >= 18.5) | (hour_float <= 6.0)).astype(float)
    pump_cycle = 0.5 + 0.5 * np.sin(2.0 * np.pi * (hour_float - 5.0) / 12.0)
    aux_raw = 0.35 + 0.25 * pump_cycle + 0.40 * night_lighting

    out = pd.DataFrame(
        {
            "datetime": dt,
            "base_kW": base_raw,
            "plug_kW": plug_raw,
            "lighting_kW": lighting_raw,
            "hvac_kW": hvac_raw,
            "aux_kW": aux_raw,
            "temperature_C": temp_c,
            "relative_humidity_pct": rh,
            "irradiance_W_m2": irr,
            "rainfall_mm": weather["rainfall_mm"].to_numpy(dtype=float),
            "occupancy_factor": occupancy,
            "operation_factor": operation,
        }
    )
    return out


def fit_load_coefficients(reference_weather: pd.DataFrame) -> dict[str, float]:
    """Fit on an independent, pre-2023 reference; do not normalize the test year."""
    raw = load_raw_shapes(reference_weather)
    valid = reference_weather["weather_history_valid"].astype(bool)
    if not valid.any():
        raise ValueError("No valid reference weather for load calibration")
    coefficients = {}
    for name, annual_target in END_USE_TARGETS.items():
        annualized_raw = float(raw.loc[valid, name].mean() * 8760.0)
        if annualized_raw <= 0:
            raise ValueError(f"Invalid historical load calibration for {name}")
        coefficients[name] = annual_target / annualized_raw
    return coefficients


def build_load_15min(
    weather: pd.DataFrame, coefficients: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply coefficients calibrated on 2022, with no 2023 annual rescaling.

    This remains a synthetic building scenario, not measured building demand.
    """
    if coefficients is None:
        coefficients = fit_load_coefficients(build_weather_15min(YEAR - 1))
    out = load_raw_shapes(weather)
    for name in END_USE_TARGETS:
        out[name] *= coefficients[name]
    out["load_kW"] = out[list(END_USE_TARGETS)].sum(axis=1)
    for col in weather:
        if col not in out:
            out[col] = weather[col].to_numpy()
    out["load_scenario_valid"] = weather["weather_history_valid"].to_numpy()
    return out


def _energy_kwh(df: pd.DataFrame, col: str) -> float:
    return float(df[col].sum() * DT_HOURS)


def plot_15min_outputs(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Plotting skipped: matplotlib is not installed in this Python environment.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["date"] = plot_df["datetime"].dt.date
    plot_df["month"] = plot_df["datetime"].dt.month

    daily = plot_df.groupby("date", as_index=False)["load_kW"].sum()
    daily["load_kWh"] = daily["load_kW"] * DT_HOURS

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily["date"], daily["load_kWh"], color="#334e68", linewidth=0.9)
    ax.set_title("Daily 15-min synthetic load energy (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "load_15min_daily_kwh.png", dpi=150)
    plt.close(fig)

    end_uses = ["base_kW", "plug_kW", "lighting_kW", "hvac_kW", "aux_kW"]
    monthly = plot_df.groupby("month")[end_uses].sum() * DT_HOURS
    fig2, ax2 = plt.subplots(figsize=(9, 4.6))
    bottom = np.zeros(len(monthly), dtype=float)
    colors = ["#4c78a8", "#59a14f", "#f2cf5b", "#e45756", "#8e6c8a"]
    labels = ["Base", "Plug", "Lighting", "HVAC", "Aux"]
    for col, color, label in zip(end_uses, colors, labels):
        values = monthly[col].to_numpy()
        ax2.bar(monthly.index, values, bottom=bottom, color=color, label=label)
        bottom += values
    ax2.set_xticks(range(1, 13))
    ax2.set_title("Monthly 15-min synthetic load by end use")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("kWh")
    ax2.legend(ncols=3, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "load_15min_monthly_enduse.png", dpi=150)
    plt.close(fig2)

    fig3, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), sharey=True)
    for ax, target_day, title in (
        (axes[0], date(2023, 1, 18), "Winter day (2023-01-18)"),
        (axes[1], date(2023, 7, 19), "Summer day (2023-07-19)"),
    ):
        day = plot_df[plot_df["date"] == target_day]
        x = day["datetime"].dt.hour + day["datetime"].dt.minute / 60.0
        ax.stackplot(
            x,
            *[day[col].to_numpy() for col in end_uses],
            labels=labels,
            colors=colors,
            alpha=0.95,
        )
        ax.set_title(title)
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("kW")
    axes[0].legend(loc="upper left", fontsize=8)
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "load_15min_typical_day_stack.png", dpi=150)
    plt.close(fig3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    weather = build_weather_15min()
    weather.to_csv(WEATHER_15MIN_CSV, index=False)

    load = build_load_15min(weather)
    load.to_csv(LOAD_15MIN_CSV, index=False)
    plot_15min_outputs(load)

    print(f"Rows: {len(load):,} (expected 35,040 for 15-min non-leap year)")
    print(f"Annual load: {_energy_kwh(load, 'load_kW'):,.1f} kWh (target {TARGET_ANNUAL_KWH:,.1f})")
    for col in ["base_kW", "plug_kW", "lighting_kW", "hvac_kW", "aux_kW"]:
        energy = _energy_kwh(load, col)
        print(f"  - {col}: {energy:,.1f} kWh ({energy / TARGET_ANNUAL_KWH * 100:.1f}%)")
    print(f"Saved: {LOAD_15MIN_CSV.relative_to(ROOT)}")
    print(f"Saved: {WEATHER_15MIN_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

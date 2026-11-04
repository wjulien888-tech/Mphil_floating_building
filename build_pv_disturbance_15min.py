"""
Build 15-minute PV disturbance factors from HKUST measured PV data and apply
them to the floating-building PVGIS baseline.

Research convention used here:
- PVGIS determines the floating-building absolute PV energy.
- HKUST measured PV only provides empirical 15-minute disturbance factors.
- Training uses raw weather-disturbed PV without any annual normalization.
- Baselines use only preceding days, initialized from December 2022.
- Annual preservation is an explicitly requested retrospective diagnostic only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"

HKUST_PV_15MIN_CSV = OUT_DIR / "hkust_pv_measured_15min_2023.csv"
PVGIS_HOURLY_CSV = OUT_DIR / "pv_generation_8760.csv"
LOAD_15MIN_CSV = OUT_DIR / "load_15min_2023.csv"

DISTURBANCE_CSV = OUT_DIR / "pv_disturbance_15min_2023.csv"
PV_DISTURBED_CSV = OUT_DIR / "pv_floating_pvgis_disturbed_15min_2023.csv"
DISPATCH_DISTURBED_CSV = OUT_DIR / "dispatch_input_15min_disturbed_2023.csv"

YEAR = 2023
DT_HOURS = 0.25

# PVGIS timestamps in the existing file are hourly center timestamps. The current
# Guangzhou/Hong Kong workflow uses local time, so the hourly PVGIS sequence is
# circularly shifted by UTC+8 before expanding to 15 minutes.
PVGIS_UTC_TO_LOCAL_SHIFT_HOURS = 8

# Same-slot high percentile over strictly preceding days. No centered windows,
# interpolation from future days, or peak taken over the complete target year.
CLEAR_BASELINE_WINDOW_DAYS = 31
CLEAR_BASELINE_QUANTILE = 0.90
MAX_DISTURBANCE_FACTOR = 1.20
MIN_BASELINE_FRACTION_OF_PEAK = 0.02


def _full_15min_index() -> pd.DatetimeIndex:
    return pd.date_range(f"{YEAR}-01-01 00:00", f"{YEAR}-12-31 23:45", freq="15min")


def build_pvgis_15min_baseline() -> pd.DataFrame:
    """Convert hourly center-stamped PVGIS output to local 15-minute step-hold."""
    pvgis = pd.read_csv(PVGIS_HOURLY_CSV, parse_dates=["datetime"])
    if len(pvgis) != 8760:
        raise ValueError(f"Expected 8760 PVGIS rows, found {len(pvgis)}")

    hourly_values = pvgis["pv_kW"].to_numpy(dtype=float)
    local_hourly_values = np.roll(hourly_values, PVGIS_UTC_TO_LOCAL_SHIFT_HOURS)

    hourly_index = pd.date_range(f"{YEAR}-01-01 00:00", f"{YEAR}-12-31 23:00", freq="h")
    hourly = pd.Series(local_hourly_values, index=hourly_index, name="pvgis_floating_baseline_kW")

    baseline_15min = hourly.reindex(_full_15min_index(), method="ffill").to_frame()
    baseline_15min.index.name = "datetime"
    return baseline_15min.reset_index()


def build_disturbance_factors(
    hkust: pd.DataFrame | None = None,
    output_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    if hkust is None:
        from align_hkust_pv_15min import build_measured_pv_15min
        hkust = build_measured_pv_15min(start="2022-12-01", end="2024-01-01")
    if output_index is None:
        output_index = _full_15min_index()
    required = {"datetime", "hkust_measured_pv_kW", "raw_missing_site_count"}
    missing = required.difference(hkust.columns)
    if missing:
        raise ValueError(f"{HKUST_PV_15MIN_CSV.name} missing columns: {sorted(missing)}")

    hkust = hkust.copy()
    hkust["datetime"] = pd.to_datetime(hkust["datetime"])
    hkust = hkust.set_index("datetime").sort_index()
    if hkust.index.has_duplicates:
        raise ValueError("Duplicate HKUST PV timestamps")
    full_history = pd.date_range(hkust.index.min().normalize(), hkust.index.max(), freq="15min")
    hkust = hkust.reindex(full_history)
    measured = hkust["hkust_measured_pv_kW"].astype(float).clip(lower=0.0)
    observed_valid = hkust["raw_missing_site_count"].eq(0) & measured.notna()
    historical_observed = measured.where(observed_valid)

    tmp = historical_observed.to_frame("hkust_measured_pv_kW")
    tmp["date"] = tmp.index.normalize()
    tmp["slot"] = tmp.index.hour * 4 + tmp.index.minute // 15
    daily_slots = tmp.pivot(index="date", columns="slot", values="hkust_measured_pv_kW")

    prior_days = daily_slots.shift(1)
    clear_like = prior_days.rolling(
        window=CLEAR_BASELINE_WINDOW_DAYS,
        min_periods=7,
    ).quantile(CLEAR_BASELINE_QUANTILE)

    clear_long = (
        clear_like.stack()
        .rename("hkust_clear_like_baseline_kW")
        .reset_index()
        .rename(columns={"level_1": "slot"})
    )
    clear_long["datetime"] = clear_long["date"] + pd.to_timedelta(clear_long["slot"] * 15, unit="min")
    clear_long = clear_long.set_index("datetime")["hkust_clear_like_baseline_kW"].reindex(full_history)

    prior_peak = historical_observed.expanding(min_periods=1).max().shift(1)
    min_baseline = prior_peak * MIN_BASELINE_FRACTION_OF_PEAK
    raw_factor = measured / clear_long
    baseline_available = clear_long.notna() & prior_peak.gt(0)
    ratio_usable = baseline_available & clear_long.ge(min_baseline) & clear_long.gt(0)
    disturbance = raw_factor.where(ratio_usable, 1.0).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    disturbance = disturbance.clip(lower=0.0, upper=MAX_DISTURBANCE_FACTOR)

    out = pd.DataFrame(
        {
            "hkust_measured_pv_kW": measured,
            "hkust_clear_like_baseline_kW": clear_long.fillna(0),
            "disturbance_factor": disturbance,
            "raw_missing_site_count": hkust["raw_missing_site_count"].fillna(8).astype(int),
            "clear_baseline_min_kW": min_baseline,
            "clear_baseline_valid": baseline_available.astype(int),
            "pv_target_valid": (observed_valid & baseline_available).astype(int),
            "pv_history_valid": (
                hkust.get("pv_history_valid", observed_valid).fillna(0).astype(bool)
                & baseline_available
            ).astype(int),
        }
    )
    out.index.name = "datetime"
    return out.reindex(output_index).rename_axis("datetime").reset_index()


def apply_disturbance(
    pvgis_15min: pd.DataFrame,
    disturbance: pd.DataFrame,
    retrospective_annual_normalization: bool = False,
) -> pd.DataFrame:
    merged = pvgis_15min.merge(
        disturbance,
        on="datetime",
        how="left",
        validate="one_to_one",
    )
    merged["disturbance_factor"] = merged["disturbance_factor"].fillna(1.0)
    merged["pv_floating_disturbed_raw_kW"] = (
        merged["pvgis_floating_baseline_kW"] * merged["disturbance_factor"]
    )
    merged["pv_floating_disturbed_raw_kWh"] = merged["pv_floating_disturbed_raw_kW"] * DT_HOURS
    if not retrospective_annual_normalization:
        return merged

    baseline_energy = float(merged["pvgis_floating_baseline_kW"].sum() * DT_HOURS)
    raw_energy = float(merged["pv_floating_disturbed_raw_kW"].sum() * DT_HOURS)
    if raw_energy <= 0.0:
        raise ValueError("Raw disturbed PV annual energy is zero.")

    annual_normalization = baseline_energy / raw_energy
    merged["annual_energy_normalization"] = annual_normalization
    merged["pv_floating_disturbed_annual_preserved_kW"] = (
        merged["pv_floating_disturbed_raw_kW"] * annual_normalization
    )
    merged["pv_floating_disturbed_annual_preserved_kWh"] = (
        merged["pv_floating_disturbed_annual_preserved_kW"] * DT_HOURS
    )
    return merged


def build_dispatch_input(disturbed_pv: pd.DataFrame) -> pd.DataFrame:
    load = pd.read_csv(LOAD_15MIN_CSV, parse_dates=["datetime"])
    dispatch = load.merge(
        disturbed_pv,
        on="datetime",
        how="left",
        validate="one_to_one",
    )
    dispatch["net_load_minus_raw_disturbed_pv_kW"] = (
        dispatch["load_kW"] - dispatch["pv_floating_disturbed_raw_kW"]
    )
    dispatch["history_valid"] = (
        dispatch["weather_history_valid"].astype(bool) & dispatch["pv_history_valid"].astype(bool)
    ).astype(int)
    return dispatch


def summarize(disturbance: pd.DataFrame, disturbed_pv: pd.DataFrame, dispatch: pd.DataFrame) -> None:
    baseline_energy = float(disturbed_pv["pvgis_floating_baseline_kW"].sum() * DT_HOURS)
    raw_energy = float(disturbed_pv["pv_floating_disturbed_raw_kW"].sum() * DT_HOURS)
    factor = disturbance["disturbance_factor"]
    daylight = disturbed_pv["pvgis_floating_baseline_kW"] > 0.01

    print(f"Rows: {len(disturbed_pv):,} (expected 35,040)")
    print(f"PVGIS baseline annual energy: {baseline_energy:,.1f} kWh")
    print(f"Raw disturbed annual energy: {raw_energy:,.1f} kWh")
    print("No annual normalization: all training PV transformations are causal.")
    print(f"Disturbance factor mean: {factor.mean():.3f}")
    print(f"Disturbance factor 5th/50th/95th pct: {factor.quantile(0.05):.3f}, {factor.quantile(0.50):.3f}, {factor.quantile(0.95):.3f}")
    print(f"Daylight intervals after PVGIS local shift: {int(daylight.sum()):,}")
    print(f"Raw disturbed net load range: {dispatch['net_load_minus_raw_disturbed_pv_kW'].min():,.1f} to {dispatch['net_load_minus_raw_disturbed_pv_kW'].max():,.1f} kW")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pvgis_15min = build_pvgis_15min_baseline()
    disturbance = build_disturbance_factors()
    disturbed_pv = apply_disturbance(pvgis_15min, disturbance)
    dispatch = build_dispatch_input(disturbed_pv)

    disturbance.to_csv(DISTURBANCE_CSV, index=False)
    disturbed_pv.to_csv(PV_DISTURBED_CSV, index=False)
    dispatch.to_csv(DISPATCH_DISTURBED_CSV, index=False)

    summarize(disturbance, disturbed_pv, dispatch)
    print(f"Saved: {DISTURBANCE_CSV.relative_to(ROOT)}")
    print(f"Saved: {PV_DISTURBED_CSV.relative_to(ROOT)}")
    print(f"Saved: {DISPATCH_DISTURBED_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

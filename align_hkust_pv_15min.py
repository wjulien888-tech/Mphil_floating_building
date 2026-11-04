"""
Align measured HKUST PV site-level data to the 2023 15-minute dispatch timeline.

The measured PV is not used as the floating-building PV capacity directly. It is
kept as an empirical Clear Water Bay PV profile that can later be normalized into
disturbance factors or used for scenario construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "HKUST_clea_bay_PV+wind_data"
PV_SITE_DIR = (
    DATASET_ROOT
    / "Dataset"
    / "Time series dataset"
    / "PV generation dataset"
    / "PV stations with panel level optimizer"
    / "Site level dataset"
)
OUT_DIR = ROOT / "output"

LOAD_15MIN_CSV = OUT_DIR / "load_15min_2023.csv"
WEATHER_15MIN_CSV = OUT_DIR / "hkust_weather_15min_2023.csv"
PV_15MIN_CSV = OUT_DIR / "hkust_pv_measured_15min_2023.csv"
DISPATCH_INPUT_CSV = OUT_DIR / "dispatch_input_15min_2023.csv"

YEAR = 2023
DT_HOURS = 0.25

# Low-missing, long-coverage site-level files identified by inspect_hkust_data.py.
DEFAULT_SITES = [
    "UG Hall4.csv",
    "SQ567.csv",
    "UG Hall7.csv",
    "SQ1.csv",
    "SQ4.csv",
    "SQ2.csv",
    "LSK North.csv",
    "SQ3.csv",
]


def _site_column_name(filename: str) -> str:
    stem = Path(filename).stem
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in stem)
    clean = "_".join(part for part in clean.split("_") if part)
    return f"pv_{clean}_kW"


def _read_site_power(path: Path, start: str, end: str) -> pd.Series:
    df = pd.read_csv(path, na_values=["NA", "N/A", ""])
    required = {"Time", "power(W)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["power(W)"] = pd.to_numeric(df["power(W)"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")
    df = df[(df["Time"] >= start) & (df["Time"] < end)]
    if df["Time"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate timestamps")
    series = df.set_index("Time")["power(W)"] / 1000.0
    series.name = _site_column_name(path.name)
    return series.clip(lower=0.0)


def clean_measured_sites(pv: pd.DataFrame, max_ffill_steps: int = 4) -> pd.DataFrame:
    """Causal history imputation; imputed observations are never valid targets."""
    pv = pv.replace([np.inf, -np.inf], np.nan)
    raw_missing = pv.isna()
    available = pv.ffill(limit=max_ffill_steps) if max_ffill_steps else pv.copy()
    out = available.fillna(0.0).copy()  # finite placeholders, explicitly invalid
    out.index.name = "datetime"
    out["hkust_measured_pv_kW"] = out.sum(axis=1)
    out["hkust_measured_pv_kWh"] = out["hkust_measured_pv_kW"] * DT_HOURS
    out["selected_site_count"] = len(pv.columns)
    out["raw_missing_site_count"] = raw_missing.sum(axis=1).astype(int)
    out["pv_imputed_site_count"] = (raw_missing & available.notna()).sum(axis=1).astype(int)
    out["pv_placeholder_site_count"] = available.isna().sum(axis=1).astype(int)
    out["pv_observed_valid"] = (~raw_missing.any(axis=1)).astype(int)
    out["pv_history_valid"] = available.notna().all(axis=1).astype(int)
    return out.reset_index()


def build_measured_pv_15min(
    sites: list[str] | None = None,
    start: str = f"{YEAR}-01-01",
    end: str = f"{YEAR + 1}-01-01",
) -> pd.DataFrame:
    selected_sites = sites or DEFAULT_SITES
    full_index = pd.date_range(start, end, freq="15min", inclusive="left")
    site_frames: list[pd.Series] = []

    for site in selected_sites:
        path = PV_SITE_DIR / site
        if not path.exists():
            raise FileNotFoundError(path)
        site_frames.append(_read_site_power(path, start, end).reindex(full_index))

    pv = pd.concat(site_frames, axis=1)
    out = clean_measured_sites(pv)
    out["selected_sites"] = ";".join(selected_sites)
    return out


def build_dispatch_input(pv: pd.DataFrame) -> pd.DataFrame:
    if not LOAD_15MIN_CSV.exists():
        raise FileNotFoundError(f"Missing {LOAD_15MIN_CSV}; run load_generation_15min.py first.")
    if not WEATHER_15MIN_CSV.exists():
        raise FileNotFoundError(f"Missing {WEATHER_15MIN_CSV}; run load_generation_15min.py first.")

    load = pd.read_csv(LOAD_15MIN_CSV, parse_dates=["datetime"])
    weather = pd.read_csv(WEATHER_15MIN_CSV, parse_dates=["datetime"])
    pv = pv.copy()
    pv["datetime"] = pd.to_datetime(pv["datetime"])

    dispatch = load.merge(
        pv[["datetime", "hkust_measured_pv_kW", "hkust_measured_pv_kWh", "raw_missing_site_count"]],
        on="datetime",
        how="left",
        validate="one_to_one",
    )
    # The load file already contains the weather columns; this check keeps the
    # standalone weather output synchronized with the dispatch input timeline.
    if len(dispatch) != len(weather):
        raise ValueError("Dispatch and weather row counts differ.")
    dispatch["net_load_minus_hkust_pv_kW"] = dispatch["load_kW"] - dispatch["hkust_measured_pv_kW"]
    return dispatch


def summarize(pv: pd.DataFrame, dispatch: pd.DataFrame) -> None:
    annual_pv = float(pv["hkust_measured_pv_kWh"].sum())
    peak_pv = float(pv["hkust_measured_pv_kW"].max())
    annual_load = float(dispatch["load_kW"].sum() * DT_HOURS)
    min_net = float(dispatch["net_load_minus_hkust_pv_kW"].min())
    max_net = float(dispatch["net_load_minus_hkust_pv_kW"].max())
    missing_intervals = int((pv["raw_missing_site_count"] > 0).sum())

    print(f"Rows: {len(pv):,} (expected 35,040)")
    print(f"Selected sites: {pv['selected_site_count'].iloc[0]}")
    print(f"Measured HKUST PV energy: {annual_pv:,.1f} kWh")
    print(f"Measured HKUST PV peak: {peak_pv:,.1f} kW")
    print(f"Synthetic load energy: {annual_load:,.1f} kWh")
    print(f"Net load range using measured HKUST PV: {min_net:,.1f} to {max_net:,.1f} kW")
    print(f"Intervals with at least one raw missing site value: {missing_intervals:,}")
    print("Note: measured HKUST PV is kept only as an empirical disturbance source, not as floating-building PV capacity.")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pv = build_measured_pv_15min()
    pv.to_csv(PV_15MIN_CSV, index=False)

    dispatch = build_dispatch_input(pv)
    dispatch.to_csv(DISPATCH_INPUT_CSV, index=False)

    summarize(pv, dispatch)
    print(f"Saved: {PV_15MIN_CSV.relative_to(ROOT)}")
    print(f"Saved: {DISPATCH_INPUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Build, audit and atomically freeze reproducible M0 data. Refuses overwrite.

Run with external/ol-ems-hkust/.venv/bin/python prepare_m0_data.py.
Raw inputs and prior outputs remain unchanged. A new version needs a new path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from align_hkust_pv_15min import DEFAULT_SITES, PV_SITE_DIR, build_measured_pv_15min
from build_pv_disturbance_15min import (
    build_disturbance_factors, build_pvgis_15min_baseline, apply_disturbance,
    CLEAR_BASELINE_WINDOW_DAYS, CLEAR_BASELINE_QUANTILE,
    MIN_BASELINE_FRACTION_OF_PEAK, MAX_DISTURBANCE_FACTOR, PVGIS_HOURLY_CSV,
)
from build_tiered_load_scenarios import ALLOCATIONS, END_USES, TIERS, validate_allocations
from load_generation_15min import (
    build_weather_15min, build_load_15min, fit_load_coefficients, MET_ROOT,
    WEATHER_SOURCES, MAX_WEATHER_FFILL_STEPS, MIN_WEATHER_OBSERVATIONS,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output/frozen/hkust_m0_v1_20260909"
SPLITS = {"train": ("2023-01-01", "2023-09-01"),
          "val": ("2023-09-01", "2023-11-01"),
          "test": ("2023-11-01", "2024-01-01")}
SCENARIO_IMPORT_PRICES = {"valley": .29526875, "flat": .73196875,
                          "peak": 1.22496875, "sharp_peak": 1.52436875}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def gap_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for variable in WEATHER_SOURCES:
        mask = frame[f"{variable}_insufficient_coverage"].astype(bool)
        groups = mask.ne(mask.shift()).cumsum()
        for _, part in frame.loc[mask].groupby(groups[mask]):
            records.append({"variable": variable, "start": str(part.datetime.iloc[0]),
                            "end_exclusive": str(part.datetime.iloc[-1] + pd.Timedelta(minutes=15)),
                            "steps": len(part), "hours": len(part) * .25,
                            "zero_observation_steps": int(part[f"{variable}_raw_missing"].sum())})
    return records


def split_quality(dispatch: pd.DataFrame) -> dict:
    history = dispatch.history_valid.to_numpy(dtype=bool)
    target = dispatch.pv_target_valid.to_numpy(dtype=bool)
    bad_history = np.concatenate([[0], np.cumsum(~history)])
    bad_target = np.concatenate([[0], np.cumsum(~target)])
    anchors = np.arange(672, len(dispatch) - 96 + 1)
    usable = (bad_history[anchors] - bad_history[anchors - 672] == 0)
    usable &= (bad_target[anchors + 96] - bad_target[anchors] == 0)
    result = {}
    for name, (start, end) in SPLITS.items():
        rowmask = dispatch.datetime.ge(start) & dispatch.datetime.lt(end)
        assigned = dispatch.datetime.iloc[anchors].ge(start).to_numpy()
        assigned &= dispatch.datetime.iloc[anchors + 95].lt(end).to_numpy()
        result[name] = {"start_inclusive": start, "end_exclusive": end,
                        "rows": int(rowmask.sum()),
                        "history_valid_rows": int(dispatch.loc[rowmask, "history_valid"].sum()),
                        "pv_target_valid_rows": int(dispatch.loc[rowmask, "pv_target_valid"].sum()),
                        "candidate_windows_stride_1": int(assigned.sum()),
                        "valid_windows_stride_1": int((assigned & usable).sum()),
                        "filtered_windows_stride_1": int((assigned & ~usable).sum())}
    return result


def freeze(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Frozen data is write-once; choose another --output: {output_dir}")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
                    "-p", "test_causal_preprocessing.py", "-v"], cwd=ROOT, check=True)
    print("Building weather and fixed 2022 load calibration...", flush=True)
    reference_weather = build_weather_15min(2022)
    coefficients = fit_load_coefficients(reference_weather)
    weather = build_weather_15min(2023)
    load = build_load_15min(weather, coefficients)
    print("Building causal PV with December 2022 initialization...", flush=True)
    measured = build_measured_pv_15min(start="2022-12-01", end="2024-01-01")
    disturbance = build_disturbance_factors(measured)
    pv = apply_disturbance(build_pvgis_15min_baseline(), disturbance)
    dispatch = load.merge(pv, on="datetime", validate="one_to_one")
    dispatch["history_valid"] = (dispatch.weather_history_valid.astype(bool) & dispatch.pv_history_valid.astype(bool)).astype(int)
    dispatch["net_load_minus_raw_disturbed_pv_kW"] = dispatch.load_kW - dispatch.pv_floating_disturbed_raw_kW
    hours = dispatch.datetime.dt.hour + dispatch.datetime.dt.minute / 60.0
    period = pd.Series("flat", index=dispatch.index)
    period.loc[hours.lt(8)] = "valley"
    period.loc[(hours.ge(10) & hours.lt(12)) | (hours.ge(14) & hours.lt(19))] = "peak"
    summer = dispatch.datetime.dt.month.between(7, 9)
    period.loc[summer & ((hours.ge(11) & hours.lt(12)) | (hours.ge(15) & hours.lt(17)))] = "sharp_peak"
    dispatch["tariff_period"] = period
    dispatch["import_price"] = period.map(SCENARIO_IMPORT_PRICES)
    dispatch["export_price"] = .36
    dispatch["grid_available"] = 1
    validate_allocations()
    tiers = dispatch[["datetime", "load_kW"]].copy()
    for scenario, allocations in ALLOCATIONS.items():
        for tier in TIERS:
            tiers[f"{scenario}_{tier}_kW"] = sum(load[end_use] * allocations[end_use][tier] for end_use in END_USES)
        reconstructed = tiers[[f"{scenario}_{tier}_kW" for tier in TIERS]].sum(axis=1)
        if not np.allclose(reconstructed, dispatch.load_kW, rtol=0, atol=1e-8):
            raise ValueError(f"Tier reconstruction failed: {scenario}")
    expected_time = pd.date_range("2023-01-01", "2024-01-01", freq="15min", inclusive="left")
    if not np.array_equal(dispatch.datetime.to_numpy(), expected_time.to_numpy()):
        raise ValueError("Dispatch timeline is incomplete")
    if not np.isfinite(dispatch.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Non-finite dispatch data")
    quality = split_quality(dispatch)
    if any(stats["valid_windows_stride_1"] == 0 for stats in quality.values()):
        raise ValueError(f"No usable windows in a split: {quality}")
    gaps = gap_records(weather)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    outputs = {"dispatch.csv": dispatch, "load_tiers.csv": tiers, "weather.csv": weather,
               "pv_disturbance.csv": disturbance, "weather_gaps.csv": pd.DataFrame(gaps),
               "measured_pv_with_initialization.csv": measured}
    for name, frame in outputs.items():
        frame.to_csv(staging / name, index=False)
    input_paths = [PV_SITE_DIR / name for name in DEFAULT_SITES]
    input_paths += [MET_ROOT / folder / f"{folder}_{year}.csv" for folder, _ in WEATHER_SOURCES.values() for year in (2022, 2023)]
    input_paths += [PVGIS_HOURLY_CSV, ROOT / "HKUST_clea_bay_PV+wind_data/README.md"]
    code_paths = [ROOT / name for name in ("align_hkust_pv_15min.py", "build_pv_disturbance_15min.py",
                 "load_generation_15min.py", "build_tiered_load_scenarios.py", "prepare_m0_data.py",
                 "tests/test_causal_preprocessing.py", "pv_generation.py")]
    manifest = {
        "schema_version": 1, "dataset_version": output_dir.name, "frozen_date": "2026-09-09",
        "timezone_convention": "Asia/Shanghai, naive local timestamps; 15-minute rows denote completed intervals",
        "frequency": "15min", "rows": len(dispatch), "splits": quality,
        "inputs": [file_record(path) for path in input_paths],
        "code": [file_record(path) for path in code_paths],
        "outputs": [{"path": name, "size_bytes": (staging / name).stat().st_size,
                     "sha256": sha256_file(staging / name)} for name in outputs],
        "parameters": {"lookback_steps": 672, "forecast_horizon_steps": 96,
            "weather_min_observed_minutes_per_bin": MIN_WEATHER_OBSERVATIONS,
            "weather_max_causal_ffill_steps": MAX_WEATHER_FFILL_STEPS, "rainfall_ffill_steps": 0,
            "pv_max_causal_ffill_steps": 4, "pv_initialization_start": "2022-12-01",
            "clear_baseline_preceding_days": CLEAR_BASELINE_WINDOW_DAYS,
            "clear_baseline_min_valid_days": 7, "clear_baseline_quantile": CLEAR_BASELINE_QUANTILE,
            "minimum_baseline_fraction_of_prior_peak": MIN_BASELINE_FRACTION_OF_PEAK,
            "maximum_disturbance_factor": MAX_DISTURBANCE_FACTOR,
            "pvgis_fixed_climatology_local_shift_hours": 8,
            "load_calibration_period": ["2022-01-01", "2023-01-01"],
            "load_coefficients": coefficients, "tier_allocations": ALLOCATIONS},
        "quality": {"raw_pv_missing_rows_2023": int(dispatch.raw_missing_site_count.gt(0).sum()),
            "pv_target_invalid_rows": int(dispatch.pv_target_valid.eq(0).sum()),
            "history_invalid_rows": int(dispatch.history_valid.eq(0).sum()),
            "weather_invalid_rows": int(weather.weather_history_valid.eq(0).sum()),
            "max_weather_gap_hours": max((gap["hours"] for gap in gaps), default=0),
            "weather_by_column": {col: {
                "zero_observation_rows": int(weather[f"{col}_raw_missing"].sum()),
                "partial_observation_rows": int(weather[f"{col}_partial"].sum()),
                "filled_rows": int(weather[f"{col}_filled"].sum()),
                "invalid_rows": int(weather[f"{col}_valid"].eq(0).sum()),
            } for col in WEATHER_SOURCES},
            "load_reference_valid_rows": int(reference_weather.weather_history_valid.sum())},
        "energy_scenario_kWh": {"load_2023": float(dispatch.load_kW.sum() * .25),
            "pvgis_reference": float(dispatch.pvgis_floating_baseline_kW.sum() * .25),
            "constructed_pv_raw_including_flagged_placeholders": float(dispatch.pv_floating_disturbed_raw_kW.sum() * .25)},
        "rules": [
            "No future 2023 observation changes an earlier 2023 transformed PV, weather or load row.",
            "PV baseline: same slot over preceding 31 days, excluding current day; only fully observed site sums enter baseline.",
            "Peak threshold: expanding maximum of observed site sum through the preceding interval; never full-year peak.",
            "No backward fill, centered baseline, future interpolation or annual PV energy normalization in training data.",
            "PV targets require complete raw site observations and available causal baseline; imputed targets are rejected.",
            "Weather bins need 12/15 observed minutes. Temperature/RH/irradiance may carry forward at most four bins; rainfall may not.",
            "Finite placeholders remain explicitly invalid and cannot be included in scaler fit or sampled history/targets.",
            "Accept a window only if all 672 history rows have history_valid=1 and all 96 target rows have pv_target_valid=1.",
            "Fit the model scaler only on training-period rows with history_valid=1; keep target windows wholly inside their split.",
            "Load coefficients fitted once using 2022 weather; 2023 load is not normalized using future annual weather.",
            "Calendar schedules, PVGIS reference and tier allocations are fixed scenario assumptions, not future measured inputs."
        ],
        "provenance": {
            "measured_source": "HKUST Clear Water Bay rooftop PV/weather, local dataset README; https://doi.org/10.5061/dryad.m37pvmd99",
            "training_target": "Constructed floating-building PV: fixed PVGIS profile multiplied by causally derived HKUST disturbance; not measured floating-building power.",
            "pvgis": "Existing output/pv_generation_8760.csv is frozen as an independent engineering reference; source script and file hashes included.",
            "load": "Synthetic end-use scenario calibrated against 2022; not metered demand; 2023 total need not equal 60461 kWh.",
            "tariff": {"definition": "Explicit research scenario, not measured HKUST bills: valley 00-08; peak 10-12 and 14-19; sharp peak Jul-Sep 11-12 and 15-17; otherwise flat. No trigger from same-day future temperature.",
                       "import_CNY_per_kWh": SCENARIO_IMPORT_PRICES, "export_CNY_per_kWh": .36},
            "grid": "All-grid-connected M0 reference, grid_available=1. Outage scenarios remain separate work.",
            "source_selection": "Eight sites retained from existing project screening; site selection was not re-optimized in this freeze."
        },
        "verification": {"causal_regression_tests": "6 passed before generation", "timeline_and_finite_values": "passed", "tier_reconstruction": "passed"},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output_dir.exists():
        raise FileExistsError(f"Output appeared during generation: {output_dir}; staged data remains at {staging}")
    staging.rename(output_dir)
    print(json.dumps({"frozen_path": str(output_dir), "quality": manifest["quality"], "splits": quality,
                      "energy_scenario_kWh": manifest["energy_scenario_kWh"]}, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    freeze(parser.parse_args().output)

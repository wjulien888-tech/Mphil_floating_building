"""Freeze the missing 2021--2022 dispatch inputs for M1/M2, without training.

The synthetic load formula, fixed 2022 calibration coefficients, tier shares
and research tariff are inherited from M0. Earlier years use their own weather
and calendar. All 2023 load, tier and tariff values are copied from the original
freeze. PV comes unchanged from the selected multi-year PV freeze.

This script never imports a model, solves MPC, launches tests or overwrites a
freeze. The frozen coefficients are scenario parameters calibrated within the
training period, not a claim that their 2022 calibration was known in 2021.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import load_generation_15min as load_model
from build_tiered_load_scenarios import END_USES, TIERS

DEFAULT_REFERENCE = ROOT / "output/frozen/hkust_m0_v1_20260909"
DEFAULT_PV = ROOT / "output/frozen/pv_only_2021_2023_multihorizon96_v1_20260921/lookback_96/pv_timeseries.csv"
DEFAULT_OUTPUT = ROOT / "output/frozen/task_dispatch_2021_2023_v1_20261003"
START, END = "2021-07-01", "2024-01-01"
WEATHER_DRIVERS = ("temperature_C", "relative_humidity_pct", "irradiance_W_m2")
LOAD_COLUMNS = ["datetime", *END_USES, "occupancy_factor", "operation_factor", "load_kW", "load_scenario_valid"]
DISPATCH_COLUMNS = [*LOAD_COLUMNS, "tariff_period", "import_price", "export_price", "grid_available"]
CALENDAR_SOURCES = {
    "2021": "https://app.www.gov.cn/govdata/gov/202011/25/465322/article.html",
    "2022": "https://app.www.gov.cn/govdata/gov/202110/25/477428/article.html",
    "2023": "https://app.www.gov.cn/govdata/gov/202212/08/495070/article.html",
}


def _days(start: str, end: str) -> set[date]:
    return set(pd.date_range(start, end, freq="D").date)


# State Council annual rest/makeup schedules; same mainland research-site
# convention as the existing Nansha load model, not Hong Kong office holidays.
REST_DATES = {
    2021: set().union(*[_days(a, b) for a, b in (
        ("2021-01-01", "2021-01-03"), ("2021-02-11", "2021-02-17"),
        ("2021-04-03", "2021-04-05"), ("2021-05-01", "2021-05-05"),
        ("2021-06-12", "2021-06-14"), ("2021-09-19", "2021-09-21"),
        ("2021-10-01", "2021-10-07"))]),
    2022: set().union(*[_days(a, b) for a, b in (
        ("2022-01-01", "2022-01-03"), ("2022-01-31", "2022-02-06"),
        ("2022-04-03", "2022-04-05"), ("2022-04-30", "2022-05-04"),
        ("2022-06-03", "2022-06-05"), ("2022-09-10", "2022-09-12"),
        ("2022-10-01", "2022-10-07"), ("2022-12-31", "2022-12-31"))]),
    2023: set(load_model.OFFICIAL_REST_2023),
}
MAKEUP_DATES = {
    2021: {date.fromisoformat(s) for s in (
        "2021-02-07", "2021-02-20", "2021-04-25", "2021-05-08",
        "2021-09-18", "2021-09-26", "2021-10-09")},
    2022: {date.fromisoformat(s) for s in (
        "2022-01-29", "2022-01-30", "2022-04-02", "2022-04-24",
        "2022-05-07", "2022-10-08", "2022-10-09")},
    2023: set(load_model.MAKEUP_WORKDAYS_2023),
}


def uses_rest_schedule(day: date) -> bool:
    if day.year not in REST_DATES:
        raise ValueError(f"No verified calendar for year {day.year}")
    if day in MAKEUP_DATES[day.year]:
        return False
    return day in REST_DATES[day.year] or day.weekday() >= 5


def occupancy_and_operation(times: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    dt = pd.to_datetime(times)
    hours = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60.0
    rest = np.array([uses_rest_schedule(day) for day in dt.dt.date], dtype=bool)
    operation = np.where(rest, load_model._weekend_operation(hours), load_model._weekday_operation(hours))
    return operation * np.where(rest, 0.75, 1.0), operation


def build_load(weather: pd.DataFrame, coefficients: dict[str, float]) -> pd.DataFrame:
    """Original five-end-use formula, with the date's own annual calendar.

    The formula matches load_generation_15min.load_raw_shapes; only its
    hard-coded 2023 calendar is replaced. No annual energy normalization.
    Rainfall has never been an input to any of these five formulas.
    """
    if set(coefficients) != set(END_USES) or not all(np.isfinite(v) and v > 0 for v in coefficients.values()):
        raise ValueError("Invalid frozen load coefficients")
    dt = pd.to_datetime(weather["datetime"])
    occupancy, operation = occupancy_and_operation(dt)
    temp = weather["temperature_C"].to_numpy(dtype=float)
    rh = weather["relative_humidity_pct"].to_numpy(dtype=float)
    irr = weather["irradiance_W_m2"].to_numpy(dtype=float)
    daylight = np.clip(irr / 600.0, 0.0, 1.0)
    cooling = np.isin(dt.dt.month.to_numpy(), list(load_model.COOLING_MONTHS)).astype(float)
    cooling_degree = np.maximum(0.0, temp - load_model.COOLING_BASE_TEMP_C)
    humidity_term = np.maximum(0.0, rh - load_model.HUMIDITY_BASE_RH) / 20.0
    solar_term = np.maximum(0.0, irr - load_model.IRRADIANCE_BASE_W_M2) / 500.0
    hour = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60.0
    night = ((hour >= 18.5) | (hour <= 6.0)).astype(float)
    pump = 0.5 + 0.5 * np.sin(2.0 * np.pi * (hour - 5.0) / 12.0)
    raw = {
        "base_kW": 0.55 + 0.25 * operation + 0.10 * occupancy,
        "plug_kW": 0.06 + np.power(np.clip(occupancy, 0.0, 1.0), 1.25),
        "lighting_kW": 0.04 + operation * (0.25 + 0.75 * (1.0 - daylight)),
        "hvac_kW": cooling * (0.25 + 0.75 * operation) * (cooling_degree + 0.75 * humidity_term + 0.60 * solar_term),
        "aux_kW": 0.35 + 0.25 * pump + 0.40 * night,
    }
    out = pd.DataFrame({"datetime": dt, **{col: values * coefficients[col] for col, values in raw.items()}})
    out["occupancy_factor"], out["operation_factor"] = occupancy, operation
    out["load_kW"] = out[list(END_USES)].sum(axis=1)
    out["load_scenario_valid"] = weather["load_weather_valid"].to_numpy(dtype=int)
    return out[LOAD_COLUMNS]


def build_weather(year: int, met_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    sources, paths = {}, []
    for name in WEATHER_DRIVERS:
        folder, value = load_model.WEATHER_SOURCES[name]
        path = met_root / folder / f"{folder}_{year}.csv"
        sources[name] = load_model._read_weather_csv(path, value)[value]
        paths.append(path)
    axis = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", inclusive="left")
    frame = load_model.aggregate_weather_15min(sources, axis)
    return frame.rename(columns={"weather_history_valid": "load_weather_valid"}), paths


def add_tariff(load: pd.DataFrame, tariff: dict[str, Any]) -> pd.DataFrame:
    out = load.copy()
    dt = pd.to_datetime(out["datetime"])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    period = pd.Series("flat", index=out.index)
    period.loc[hour.lt(8)] = "valley"
    period.loc[(hour.ge(10) & hour.lt(12)) | (hour.ge(14) & hour.lt(19))] = "peak"
    summer = dt.dt.month.between(7, 9)
    period.loc[summer & ((hour.ge(11) & hour.lt(12)) | (hour.ge(15) & hour.lt(17)))] = "sharp_peak"
    out["tariff_period"] = period
    out["import_price"] = period.map(tariff["import_CNY_per_kWh"])
    out["export_price"] = float(tariff["export_CNY_per_kWh"])
    out["grid_available"] = 1
    return out


def build_tiers(load: pd.DataFrame, allocations: dict[str, Any]) -> pd.DataFrame:
    tiers = load[["datetime", "load_kW"]].copy()
    for scenario, uses in allocations.items():
        if set(uses) != set(END_USES):
            raise ValueError(f"Invalid frozen tier end uses: {scenario}")
        for end_use, fractions in uses.items():
            if (set(fractions) != set(TIERS) or not all(0 <= v <= 1 for v in fractions.values())
                    or not np.isclose(sum(fractions.values()), 1.0, rtol=0, atol=1e-12)):
                raise ValueError(f"Invalid frozen tier allocation: {scenario}/{end_use}")
        for tier in TIERS:
            tiers[f"{scenario}_{tier}_kW"] = sum(load[col] * uses[col][tier] for col in END_USES)
    return tiers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    return {"path": label, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"], float_precision="round_trip")
    axis = pd.DatetimeIndex(frame["datetime"])
    if axis.tz is not None or axis.hasnans or axis.has_duplicates or not axis.is_monotonic_increasing:
        raise ValueError(f"Expected unique ordered local timestamps: {path}")
    return frame


def verify_2023_preserved(dispatch: pd.DataFrame, tiers: pd.DataFrame,
                          reference_dispatch: pd.DataFrame, reference_tiers: pd.DataFrame) -> None:
    for name, actual, expected in (("dispatch", dispatch, reference_dispatch[DISPATCH_COLUMNS]),
                                  ("tiers", tiers, reference_tiers)):
        observed = actual.loc[actual.datetime.ge("2023-01-01"), expected.columns].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(observed, expected.reset_index(drop=True), check_exact=True, check_dtype=False)
        except AssertionError as error:
            raise ValueError(f"2023 frozen {name} changed") from error


def freeze(output: Path = DEFAULT_OUTPUT, reference: Path = DEFAULT_REFERENCE,
           pv_path: Path = DEFAULT_PV, met_root: Path = load_model.MET_ROOT) -> dict[str, Any]:
    output, reference, pv_path = output.resolve(), reference.resolve(), pv_path.resolve()
    if output.exists():
        raise FileExistsError(f"Frozen data cannot be overwritten; choose another --output: {output}")
    source_manifest = reference / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    # Refuse modified inputs that merely reuse the reference folder name.
    for filename in ("dispatch.csv", "load_tiers.csv", "weather.csv"):
        record = next((r for r in manifest["outputs"] if r["path"] == filename), None)
        if record is None or _sha256(reference / filename) != record["sha256"]:
            raise ValueError(f"Reference frozen hash mismatch: {filename}")
    reference_dispatch = _read(reference / "dispatch.csv")
    reference_tiers = _read(reference / "load_tiers.csv")
    reference_weather = _read(reference / "weather.csv")
    reference_axis = pd.date_range("2023-01-01", END, freq="15min", inclusive="left")
    for label, frame in (("dispatch", reference_dispatch), ("tiers", reference_tiers), ("weather", reference_weather)):
        if not pd.DatetimeIndex(frame.datetime).equals(reference_axis):
            raise ValueError(f"Incomplete 2023 reference {label}")
    coefficients = manifest["parameters"]["load_coefficients"]
    allocations = manifest["parameters"]["tier_allocations"]
    tariff = manifest["provenance"]["tariff"]
    dispatch_parts, tier_parts, weather_parts, source_paths = [], [], [], []
    for year in (2021, 2022):
        print(f"Preparing {year}: original load formula with {year} weather/calendar...", flush=True)
        weather, paths = build_weather(year, met_root)
        weather = weather.loc[weather.datetime.ge(START)].reset_index(drop=True)
        load = build_load(weather, coefficients)
        dispatch_parts.append(add_tariff(load, tariff))
        tier_parts.append(build_tiers(load, allocations))
        weather["load_scenario_valid"] = load.load_scenario_valid.to_numpy()
        weather_parts.append(weather)
        source_paths.extend(paths)
    dispatch_parts.append(reference_dispatch[DISPATCH_COLUMNS].copy())
    tier_parts.append(reference_tiers.copy())
    weather_columns = ["datetime", *[c for c in reference_weather if any(c == driver or c.startswith(driver + "_") for driver in WEATHER_DRIVERS)]]
    weather_2023 = reference_weather[weather_columns].copy()
    weather_2023["load_weather_valid"] = weather_2023[[f"{name}_valid" for name in WEATHER_DRIVERS]].all(axis=1).astype(int)
    weather_2023["load_scenario_valid"] = reference_dispatch.load_scenario_valid.to_numpy()
    weather_parts.append(weather_2023)
    dispatch = pd.concat(dispatch_parts, ignore_index=True)
    tiers = pd.concat(tier_parts, ignore_index=True)
    weather = pd.concat(weather_parts, ignore_index=True)
    axis = pd.date_range(START, END, freq="15min", inclusive="left")
    for label, frame in (("dispatch", dispatch), ("tiers", tiers), ("weather", weather)):
        if not pd.DatetimeIndex(frame.datetime).equals(axis):
            raise ValueError(f"Incomplete multi-year {label} time axis")
        if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
            raise ValueError(f"Non-finite values in {label}")
    pv_source = _read(pv_path)
    pv = pv_source.loc[pv_source.datetime.ge(START) & pv_source.datetime.lt(END)].reset_index(drop=True)
    if not pd.DatetimeIndex(pv.datetime).equals(axis):
        raise ValueError("Multi-year PV source does not cover the required complete time axis")
    needed = ["datetime", "pv_kW", "history_valid", "pv_target_valid", "split"]
    if not set(needed).issubset(pv):
        raise ValueError(f"PV source must have {needed}")
    # Keep PV validity separate from load validity. The training selector checks
    # the original PV file and load_scenario_valid independently.
    dispatch = dispatch.merge(pv[needed].rename(columns={"history_valid": "pv_history_valid", "split": "pv_split"}),
                              on="datetime", validate="one_to_one", sort=False)
    verify_2023_preserved(dispatch, tiers, reference_dispatch, reference_tiers)
    for scenario in allocations:
        cols = [f"{scenario}_{tier}_kW" for tier in TIERS]
        if not np.allclose(tiers[cols].sum(axis=1), dispatch.load_kW, rtol=0, atol=1e-8):
            raise ValueError(f"Tier reconstruction failed: {scenario}")
    source_paths += [source_manifest, reference / "dispatch.csv", reference / "load_tiers.csv", reference / "weather.csv", pv_path]
    pv_manifest = pv_path.parent.parent / "manifest.json"
    if pv_manifest.is_file():
        source_paths.append(pv_manifest)
    quality = {}
    for year, part in dispatch.groupby(dispatch.datetime.dt.year):
        quality[str(year)] = {"start": str(part.datetime.iloc[0]), "last_row": str(part.datetime.iloc[-1]),
                              "rows": len(part), "load_invalid_rows": int(part.load_scenario_valid.ne(1).sum()),
                              "pv_target_invalid_rows": int(part.pv_target_valid.ne(1).sum()),
                              "load_kWh_all_rows_including_invalid_placeholders": float(part.load_kW.sum() * 0.25)}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        frames = {"dispatch.csv": dispatch, "load_tiers.csv": tiers, "weather.csv": weather}
        for name, frame in frames.items():
            frame.to_csv(staging / name, index=False)
        # Round-trip checks confirm that writing has not changed held-out data.
        verify_2023_preserved(_read(staging / "dispatch.csv"), _read(staging / "load_tiers.csv"), reference_dispatch, reference_tiers)
        result = {
            "schema_version": 1, "artifact_type": "multiyear_task_dispatch", "dataset_version": output.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "start_inclusive": START, "end_exclusive": END, "frequency": "15min", "rows": len(dispatch),
            "timezone_convention": "Asia/Shanghai naive local timestamps; interval [time,time+15min)",
            "split_windows": {"train": [START, "2023-09-01"], "val": ["2023-09-01", "2023-11-01"], "test": ["2023-11-01", END]},
            "inputs": [_record(path) for path in source_paths],
            "code": [_record(path) for path in (Path(__file__), ROOT / "load_generation_15min.py", ROOT / "build_tiered_load_scenarios.py")],
            "outputs": [{"path": name, "size_bytes": (staging / name).stat().st_size, "sha256": _sha256(staging / name)} for name in frames],
            "parameters": {"load_coefficients": coefficients, "load_calibration_period": manifest["parameters"]["load_calibration_period"],
                           "tier_allocations": allocations, "weather_drivers": list(WEATHER_DRIVERS),
                           "weather_min_observed_minutes_per_bin": load_model.MIN_WEATHER_OBSERVATIONS,
                           "weather_max_causal_ffill_steps": load_model.MAX_WEATHER_FFILL_STEPS,
                           "tariff": tariff},
            "calendar": {"sources": CALENDAR_SOURCES,
                         "rest_dates": {str(y): sorted(d.isoformat() for d in dates) for y, dates in REST_DATES.items()},
                         "makeup_workdays": {str(y): sorted(d.isoformat() for d in dates) for y, dates in MAKEUP_DATES.items()}},
            "quality": quality,
            "weather_quality": {str(year): {name: {"invalid_rows": int(part[f"{name}_valid"].ne(1).sum()),
                                                    "filled_rows": int(part[f"{name}_filled"].sum()),
                                                    "zero_observation_rows": int(part[f"{name}_raw_missing"].sum())}
                                             for name in WEATHER_DRIVERS}
                                for year, part in weather.groupby(weather.datetime.dt.year)},
            "provenance": {
                "load": "Synthetic five-end-use building scenario, not metered demand. 2021/2022 use their own HKUST weather and mainland yearly calendar, with unchanged M0 formula and frozen 2022 calibration coefficients. No copying or shifting of a 2023 load curve; no annual rescaling.",
                "calibration_time_scope": "2022 calibration is inside the full training interval. Those fixed coefficients are used for all years as scenario parameters; no claim that they were available online in 2021. They are not refitted with validation/test observations.",
                "preserved_2023": "All 2023 load components, total load, operation/occupancy, load_scenario_valid, tiers, tariffs and grid flags are preserved exactly from the original freeze, including its original rainfall-dependent quality gate.",
                "rainfall": "Rainfall does not enter any load formula and is not read or imputed by this extension. The raw 2021 rainfall source is XLSX rather than the expected CSV; it is neither replaced with zero nor used to reject otherwise valid temperature/RH/irradiance rows. Rainfall numeric columns are omitted from the new weather audit.",
                "validity": "For 2021/2022, load_scenario_valid requires the three weather drivers valid. load_weather_valid always describes those three drivers; for 2023 load_scenario_valid retains the stricter original all-weather gate. Invalid driver placeholders remain flagged and cannot be used in accepted training episodes.",
                "pv": "Unchanged values and quality flags from the existing multi-year frozen PV source; constructed PVGIS-by-year baseline times causal HKUST disturbance, not measured floating-building generation.",
                "tariff": "The same fixed research tariff schedule and prices apply in every year. These are scenario assumptions, not historical observed electricity bills.",
                "tiers": "Unchanged M0 functional proxy shares, not measured electrical circuit classifications.",
            },
            "verification": {"time_axes_complete": True, "tier_reconstruction": True, "2023_load_tiers_tariff_exact_after_csv_round_trip": True,
                             "training_or_solver_executed": False},
        }
        (staging / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during generation: {output}")
        staging.rename(output)
    except BaseException:
        # Only our own unpublished staging folder can be removed.
        shutil.rmtree(staging)
        raise
    print(json.dumps({"frozen_path": str(output), "quality": quality, "verification": result["verification"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--pv-timeseries", type=Path, default=DEFAULT_PV)
    result.add_argument("--met-root", type=Path, default=load_model.MET_ROOT)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    freeze(args.output, args.reference, args.pv_timeseries, args.met_root)

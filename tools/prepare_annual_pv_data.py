"""Prepare the agreed 18/2/12-month experiment inputs; never train a model.

Existing constructed 2021--2023 PV is preserved. Jan--Feb 2024 uses PVGIS 6
with the original building equations and no HKUST disturbance. March 1 is
stored only as the last MPC horizon's planning tail, not scored as test data.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import numpy as np
import pandas as pd
import pv_generation as physical
import prepare_task_dispatch_multiyear as dispatch_tools
from download_pvgis_2024 import DEFAULT_OUTPUT as RAW, SURFACES, START, END, digest, parameters
from prepare_pv_only_multihorizon import freeze as freeze_forecasts, file_record, write_json

OLD_PV = ROOT / "output/frozen/pv_only_2021_2023_multihorizon96_v1_20260921/lookback_96"
OLD_DISPATCH = ROOT / "output/frozen/task_dispatch_2021_2023_v1_20261003"
REFERENCE = ROOT / "output/frozen/hkust_m0_v1_20260909"
DEFAULT_OUTPUT = ROOT / "output/prepared/annual_18_2_12_20261006"
SPLITS = {"train": ("2021-07-01", "2023-01-01"),
          "val": ("2023-01-01", "2023-03-01"),
          "test": ("2023-03-01", "2024-03-01")}
TAIL_END = "2024-03-02"
PROVENANCE = ("2021--2023: preserved PVGIS 5.3 building baseline times causal HKUST disturbance; "
              "2024 Jan--Feb and planning tail: PVGIS 6 ERA5 building simulation without HKUST disturbance; "
              "not measured floating-building PV. Source segments must be reported separately.")


def verify_files(directory, required):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in required:
        records = [r for r in manifest["outputs"] if r["path"] == name]
        if len(records) != 1 or digest(directory / name) != records[0]["sha256"]:
            raise ValueError(f"Frozen input changed: {directory / name}")
    return manifest


def read_surfaces(raw):
    receipt = json.loads((raw / "manifest.json").read_text())
    expected = pd.date_range(START, END, freq="h")
    tables = {}
    for name in SURFACES:
        records = [r for r in receipt["requests"] if r["surface"] == name]
        if len(records) != 1 or records[0]["parameters"] != parameters(name):
            raise ValueError(f"Missing or mismatched PVGIS request: {name}")
        path = raw / records[0]["file"]
        if digest(path) != records[0]["sha256"]:
            raise ValueError(f"PVGIS raw response modified: {path}")
        table = pd.read_csv(path, parse_dates=["Time"])
        if not pd.DatetimeIndex(table.Time).equals(expected):
            raise ValueError(f"Incomplete or off-grid PVGIS 6 hourly axis: {name}")
        radiation_columns = ("Global Inclined Irradiance Before Reflectivity",
                       "Direct Inclined Irradiance Before Reflectivity",
                       "Diffuse Inclined Irradiance Before Reflectivity",
                       "Ground Reflected Inclined Irradiance Before Reflectivity")
        raw_missing = table[list(radiation_columns)].isna().any(axis=1)
        # Night-only zero is physically supported by the provider's solar flag.
        # Daytime nulls remain invalid; zero below is a storage placeholder only.
        night_zero = raw_missing & table["Sun-Horizon Position"].eq("Below")
        table["night_zero_inferred"] = night_zero.astype(int)
        table["raw_radiation_missing"] = raw_missing.astype(int)
        table.loc[night_zero, list(radiation_columns)] = 0.0
        table["radiation_valid"] = np.isfinite(table[list(radiation_columns)]).all(axis=1).astype(int)
        if np.isinf(table[list(radiation_columns)]).any().any():
            raise ValueError(f"Infinite irradiance: {name}")
        if not np.isfinite(table.Temperature).all():
            raise ValueError(f"Nonfinite temperature: {name}")
        if (table["Global Inclined Irradiance Before Reflectivity"] < 0).any():
            raise ValueError("Negative irradiance")
        if not table.Temperature.between(-50, 60).all():
            raise ValueError("Expected air temperature in Celsius")
        parts = table[[f"{x} Inclined Irradiance Before Reflectivity"
                       for x in ("Direct", "Diffuse", "Ground Reflected")]].sum(axis=1)
        valid = table.radiation_valid.eq(1)
        if not np.allclose(parts[valid], table.loc[valid, "Global Inclined Irradiance Before Reflectivity"], atol=.002, rtol=1e-5):
            raise ValueError("PVGIS irradiance component sum mismatch")
        tables[name] = table
    return tables


def building_hourly(tables):
    """Use irradiance before reflectivity and ambient temperature, not API Power.

    The API Power column describes its default 1-kWp cSi system and is ignored.
    """
    powers = []
    for name, table in tables.items():
        g = table["Global Inclined Irradiance Before Reflectivity"].to_numpy()
        t = table.Temperature.to_numpy()
        areas = ((physical.DECK_M2, physical.EFF_HJT), (physical.BALU_M2, physical.EFF_CDTE)) if name == "deck" else (
            (physical.SPHERE_OPAQUE_M2, physical.EFF_HJT), (physical.SPHERE_TRANS_M2, physical.EFF_CDTE))
        powers.extend(physical.power_kw_one_surface(g, t, area, efficiency) for area, efficiency in areas)
    deck = tables["deck"]
    total = np.sum(powers, axis=0) * physical.ALBEDO_FACTOR * physical.COOLING_FACTOR
    return pd.DataFrame({"datetime_utc": deck.Time,
        "pv_kW": np.nan_to_num(total, nan=0),
        "pv_valid": np.isfinite(total).astype(int),
        "night_zero_inferred": deck.night_zero_inferred,
        "raw_radiation_missing": deck.raw_radiation_missing,
        "temperature_C": deck.Temperature,
        "irradiance_W_m2": deck["Global Inclined Irradiance Before Reflectivity"].fillna(0)})


def quarter_hours(hourly, start="2024-01-01", end=TAIL_END):
    """UTC -> UTC+8, then forward hold each hourly sample for four intervals.

    PVGIS 6 timestamps are returned on the hour. Do not apply the legacy 5.3
    hh:30-centre correction to these timestamps or invent 15-minute variability.
    """
    source = hourly.copy()
    source.index = pd.DatetimeIndex(source.datetime_utc) + pd.Timedelta(hours=8)
    axis = pd.date_range(start, end, freq="15min", inclusive="left")
    out = source.reindex(axis, method="ffill", tolerance=pd.Timedelta(minutes=45))
    if out.isna().any().any():
        raise ValueError("Missing hourly support for extension, including local-year boundary")
    out = out.rename_axis("datetime").reset_index()
    out["source_resolution_minutes"] = 60
    return out


def overlap_audit(hourly):
    old = pd.read_csv(ROOT / "output/frozen/pv_only_2021_2023_96to1_v1_20260920/pvgis_building_hourly.csv",
                      parse_dates=["datetime_utc_center"])
    old = old.loc[old.datetime_utc_center.dt.year.eq(2023)].copy()
    old["hour_key"] = old.datetime_utc_center - pd.Timedelta(minutes=30)
    new = hourly.loc[hourly.datetime_utc.dt.year.eq(2023)]
    joined = old.merge(new[["datetime_utc", "pv_kW"]], left_on="hour_key", right_on="datetime_utc",
                       suffixes=("_v53", "_v6"), validate="one_to_one")
    if len(joined) != 8760:
        raise ValueError("Incomplete 2023 version comparison")
    rows = []
    for month, group in joined.groupby(joined.hour_key.dt.strftime("%Y-%m")):
        a, b = group.pv_kW_v53.to_numpy(), group.pv_kW_v6.to_numpy()
        rows.append(dict(month=month, v53_energy_kWh=float(a.sum()), v6_energy_kWh=float(b.sum()),
                         energy_change_pct=float(100 * (b.sum() / a.sum() - 1)),
                         hourly_rmse_kW=float(np.mean((b-a)**2)**.5)))
    return pd.DataFrame(rows)


def extension_dispatch(extension, reference):
    # Published State Council calendar. This generator is restricted to Jan--Mar 1.
    dispatch_tools.REST_DATES[2024] = {date(2024, 1, 1), *[date(2024, 2, d) for d in range(10, 18)]}
    dispatch_tools.MAKEUP_DATES[2024] = {date(2024, 2, 4), date(2024, 2, 18)}
    if extension.datetime.dt.month.isin(dispatch_tools.load_model.COOLING_MONTHS).any():
        raise ValueError("PVGIS extension lacks RH; cannot generate cooling-season HVAC")
    weather = extension[["datetime", "temperature_C", "irradiance_W_m2"]].copy()
    # RH is multiplied by zero outside cooling months. Retain an explicitly
    # marked unused neutral placeholder, never label it as measured humidity.
    weather["relative_humidity_pct"] = dispatch_tools.load_model.HUMIDITY_BASE_RH
    weather["relative_humidity_pct_valid"] = 0
    weather["relative_humidity_pct_unused_placeholder"] = 1
    weather["load_weather_valid"] = extension.pv_valid.to_numpy()
    weather["data_source"] = "PVGIS6_ERA5_hourly_held; RH_unused_non_cooling"
    load = dispatch_tools.build_load(weather, reference["parameters"]["load_coefficients"])
    if not load.hvac_kW.eq(0).all():
        raise AssertionError("Humidity placeholder affected HVAC")
    dispatch = dispatch_tools.add_tariff(load, reference["provenance"]["tariff"])
    tiers = dispatch_tools.build_tiers(load, reference["parameters"]["tier_allocations"])
    return weather, dispatch, tiers


def prepare(raw=RAW, output=DEFAULT_OUTPUT):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace prepared inputs: {output}")
    old_manifest = verify_files(OLD_PV, ["pv_timeseries.csv"])
    verify_files(OLD_DISPATCH, ["dispatch.csv", "load_tiers.csv", "weather.csv"])
    reference = verify_files(REFERENCE, ["dispatch.csv", "load_tiers.csv"])
    tables = read_surfaces(raw)
    hourly = building_hourly(tables)
    extension = quarter_hours(hourly)
    audit = overlap_audit(hourly)
    old = pd.read_csv(OLD_PV / "pv_timeseries.csv", float_precision="round_trip", parse_dates=["datetime"])
    old_values = old.pv_kW.copy()
    old["data_source"] = "PVGIS53_x_causal_HKUST"
    new = extension[["datetime", "pv_kW"]].copy()
    new["history_valid"] = new["pv_target_valid"] = extension.pv_valid.to_numpy()
    new["data_source"] = "PVGIS6_only"
    combined = pd.concat([old, new], ignore_index=True)
    axis = pd.date_range("2021-01-01", TAIL_END, freq="15min", inclusive="left")
    if not pd.DatetimeIndex(combined.datetime).equals(axis):
        raise ValueError("Combined PV axis must be complete and ordered")
    if not np.array_equal(combined.pv_kW.iloc[:len(old)].to_numpy(), old_values.to_numpy()):
        raise AssertionError("Historical PV changed")
    combined["split"] = "excluded"
    for name, (start, end) in SPLITS.items():
        combined.loc[combined.datetime.ge(start) & combined.datetime.lt(end), "split"] = name
    train = combined.loc[combined.split.eq("train") & combined.history_valid.eq(1), "pv_kW"]
    combined["pv_scaled"] = (combined.pv_kW - train.min()) / (train.max() - train.min())
    combined["forecast_origin_valid"] = 0  # rebuilt by the 96-to-96 freezer
    combined["evaluation_role"] = np.where(combined.datetime.ge("2024-03-01"), "planning_tail_only", combined["split"])
    weather, new_dispatch, new_tiers = extension_dispatch(extension, reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        source = staging / "source"
        source.mkdir()
        combined.to_csv(source / "pv_timeseries.csv", index=False)
        manifest = dict(schema_version=3, dataset_version=output.name, frequency="15min", timezone="Asia/Shanghai",
            splits={k: dict(start_inclusive=v[0], end_exclusive=v[1]) for k, v in SPLITS.items()},
            pv_input_provenance=PROVENANCE,
            provenance={"historical": old_manifest["provenance"], "extension": PROVENANCE},
            planning_tail_end_exclusive=TAIL_END,
            inputs=[file_record(OLD_PV / "manifest.json"), file_record(OLD_PV / "pv_timeseries.csv"),
                    file_record(raw / "manifest.json")],
            outputs=[file_record(source / "pv_timeseries.csv", source)])
        write_json(source / "manifest.json", manifest)
        dispatch_dir = staging / "dispatch"
        dispatch_dir.mkdir()
        for filename, appended in (("dispatch.csv", new_dispatch), ("load_tiers.csv", new_tiers), ("weather.csv", weather)):
            original = pd.read_csv(OLD_DISPATCH / filename, parse_dates=["datetime"], float_precision="round_trip")
            if filename == "dispatch.csv":
                pv_columns = ["datetime", "pv_kW", "history_valid", "pv_target_valid", "split"]
                appended = appended.merge(new[pv_columns[:2]].assign(
                    pv_history_valid=extension.pv_valid.to_numpy(),
                    pv_target_valid=extension.pv_valid.to_numpy(),
                    pv_split=np.where(extension.datetime.lt("2024-03-01"), "test", "excluded")),
                    on="datetime", validate="one_to_one")
            result = pd.concat([original, appended], ignore_index=True)
            if not pd.DatetimeIndex(result.datetime).equals(pd.date_range("2021-07-01", TAIL_END, freq="15min", inclusive="left")):
                raise ValueError(f"Incomplete {filename}")
            pd.testing.assert_frame_equal(result.iloc[:len(original)][original.columns], original, check_dtype=False, check_exact=True)
            if filename == "dispatch.csv":
                # Labels follow the new experiment; historical physical values
                # and quality flags above remain byte-for-byte equivalent.
                result["pv_split"] = "excluded"
                for name, (start, end) in SPLITS.items():
                    result.loc[result.datetime.ge(start) & result.datetime.lt(end), "pv_split"] = name
            result.to_csv(dispatch_dir / filename, index=False)
        write_json(dispatch_dir / "manifest.json", dict(schema_version=1, artifact_type="annual_dispatch_extension",
            source= file_record(OLD_DISPATCH / "manifest.json"),
            parameters=reference["parameters"], provenance=reference["provenance"],
            extension="Original load formula and frozen 2022 coefficients; 2024 calendar; PVGIS6 irradiance and temperature; RH unused outside cooling months",
            calendar_source="https://app.www.gov.cn/govdata/gov/202310/25/508678/article.html",
            outputs=[file_record(p, dispatch_dir) for p in sorted(dispatch_dir.glob("*.csv"))]))
        hourly.to_csv(staging / "pvgis6_building_hourly.csv", index=False)
        extension.to_csv(staging / "pvgis6_extension_15min.csv", index=False)
        hourly.loc[hourly.raw_radiation_missing.eq(1)].to_csv(staging / "pvgis6_missing_radiation_audit.csv", index=False)
        audit.to_csv(staging / "pvgis_version_overlap_2023.csv", index=False)
        monthly = combined.loc[combined.datetime.ge("2021-07-01") & combined.datetime.lt("2024-03-01")].copy()
        monthly["month"] = monthly.datetime.dt.strftime("%Y-%m")
        quality = monthly.groupby(["month", "split", "data_source"]).agg(rows=("pv_kW", "size"),
                    history_valid_rows=("history_valid", "sum"), valid_target_rows=("pv_target_valid", "sum"),
                    pv_sum_kW=("pv_kW", "sum"), max_kW=("pv_kW", "max")).reset_index()
        quality["energy_kWh_including_existing_placeholders"] = quality.pop("pv_sum_kW") * .25
        quality.to_csv(staging / "monthly_data_quality.csv", index=False)
        write_json(staging / "preparation.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
            raw_manifest=file_record(raw / "manifest.json"), historical_pv_unchanged=True,
            source_resolution_minutes=60, model_step_minutes=15, extension_test_rows=5760,
            extension_invalid_daylight_quarters=int(extension.loc[extension.datetime.lt("2024-03-01"), "pv_valid"].eq(0).sum()),
            missing_policy="Provider Below-sun missing irradiance inferred zero; daylight nulls retained invalid with zero storage placeholders. Training/evaluation reject invalid windows. No daylight imputation approved.",
            test_start="2023-03-01", test_end_exclusive="2024-03-01", planning_tail_end_exclusive=TAIL_END,
            interpolation="UTC+8 then forward hold for 4 quarters; no synthetic subhourly noise",
            train_pv_min=float(train.min()), train_pv_max=float(train.max()),
            overlap_energy_change_pct=float(100*(audit.v6_energy_kWh.sum()/audit.v53_energy_kWh.sum()-1)),
            overlap_note="Descriptive audit, not calibration: v5.3 hh:30 mapped to hour start, v6 returned hh:00; different versions and geometry, no correction fitted on held-out data.",
            formal_training_started=False, code=[file_record(Path(__file__)), file_record(ROOT / "pv_generation.py")]))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # Freeze at final paths so provenance does not reference a temporary folder.
    freeze_forecasts(output / "source", output / "forecast", lookbacks=(96,), anchor_lookback=96)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(prepare(args.raw, args.output))

"""Time/source/quality regressions for the annual extension; no training."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "model_variants"),
               str(ROOT / "external/ol-ems-hkust")]
import prepare_annual_pv_data as prep
import _m0_variant_runner as runner
from download_pvgis_2024 import digest, parameters


def test_hourly_hold_uses_previous_utc_hour_across_new_year_and_conserves_energy():
    source = pd.DataFrame({"datetime_utc": pd.date_range("2023-12-31 15:00", periods=3, freq="h"),
                           "pv_kW": [100., 4., 8.]})
    out = prep.quarter_hours(source, "2024-01-01", "2024-01-01 02:00")
    assert out.pv_kW.tolist() == [4.] * 4 + [8.] * 4
    assert out.pv_kW.sum() * .25 == 12.
    assert out.datetime.iloc[0] == pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="Missing hourly support"):
        prep.quarter_hours(source, "2024-01-01", "2024-01-01 03:00")


def test_provider_null_in_daylight_remains_invalid_but_night_is_marked_inferred(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "START", "2024-01-01T00:00:00")
    monkeypatch.setattr(prep, "END", "2024-01-01T02:00:00")
    records = []
    for name in prep.SURFACES:
        frame = pd.DataFrame({"Time": pd.date_range(prep.START, prep.END, freq="h"),
            "Temperature": [20., 20., 20.], "Sun-Horizon Position": ["Above", "Above", "Below"],
            "Global Inclined Irradiance Before Reflectivity": [100., np.nan, np.nan],
            "Direct Inclined Irradiance Before Reflectivity": [60., np.nan, np.nan],
            "Diffuse Inclined Irradiance Before Reflectivity": [35., np.nan, np.nan],
            "Ground Reflected Inclined Irradiance Before Reflectivity": [5., np.nan, np.nan]})
        path = tmp_path / (name + ".csv")
        frame.to_csv(path, index=False)
        records.append(dict(surface=name, file=path.name, sha256=digest(path), parameters=parameters(name)))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(requests=records)))
    tables = prep.read_surfaces(tmp_path)
    for table in tables.values():
        assert table.radiation_valid.tolist() == [1, 0, 1]
        assert table.night_zero_inferred.tolist() == [0, 0, 1]
    hourly = prep.building_hourly(tables)
    assert hourly.pv_valid.tolist() == [1, 0, 1]
    assert hourly.pv_kW.iloc[0] > 0
    assert hourly.pv_kW.iloc[1:].eq(0).all()  # only row 2 is an invalid storage placeholder
    (tmp_path / "deck.csv").write_text("tampered")
    with pytest.raises(ValueError, match="modified"):
        prep.read_surfaces(tmp_path)


def test_new_split_manifest_config_is_read_without_accepting_overlap(tmp_path):
    args = runner.build_parser("GRU").parse_args([])
    args.data_manifest = tmp_path / "manifest.json"
    manifest = dict(splits={k: dict(start_inclusive=v[0], end_exclusive=v[1]) for k,v in prep.SPLITS.items()},
                    pv_input_provenance=prep.PROVENANCE)
    args.data_manifest.write_text(json.dumps(manifest))
    config = runner._runtime_config(args)
    assert config.split_windows["train"].end == "2023-01-01"
    assert config.split_windows["test"].end == "2024-03-01"
    assert config.pv_input_provenance == prep.PROVENANCE
    manifest["splits"]["val"]["start_inclusive"] = "2022-12-01"
    args.data_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonoverlapping"):
        runner._runtime_config(args)


def test_new_winter_load_has_no_dependence_on_unavailable_humidity():
    reference = json.loads((prep.REFERENCE / "manifest.json").read_text())
    extension = pd.DataFrame({"datetime": pd.to_datetime(["2024-02-04 12:00", "2024-02-10 12:00", "2024-02-18 12:00"]),
                              "temperature_C": [20., 20., 20.], "irradiance_W_m2": [400., 400., 400.],
                              "pv_valid": [1, 0, 1]})
    weather, dispatch, tiers = prep.extension_dispatch(extension, reference)
    assert weather.relative_humidity_pct_valid.eq(0).all()
    assert dispatch.hvac_kW.eq(0).all()
    assert dispatch.load_scenario_valid.tolist() == [1, 0, 1]
    assert dispatch.operation_factor.iloc[0] == dispatch.operation_factor.iloc[2]
    assert dispatch.operation_factor.iloc[0] > dispatch.operation_factor.iloc[1]
    np.testing.assert_allclose(tiers[[f"minimum_operation_CL{k}_kW" for k in (1,2,3)]].sum(axis=1), dispatch.load_kW)
    extension["datetime"] = pd.to_datetime(["2024-04-01"] * 3)
    with pytest.raises(ValueError, match="lacks RH"):
        prep.extension_dispatch(extension, reference)

"""Pure preprocessing regressions; no forecaster, solver or training import."""
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import load_generation_15min as original
import prepare_task_dispatch_multiyear as preparation
from build_tiered_load_scenarios import ALLOCATIONS, END_USES


@pytest.mark.parametrize("day,expected", [
    ("2021-09-18", False), ("2021-09-20", True),
    ("2021-09-26", False), ("2021-10-04", True),
    ("2022-01-29", False), ("2022-01-31", True),
    ("2022-10-08", False), ("2022-10-09", False),
    ("2023-01-28", False), ("2023-05-01", True),
])
def test_calendar_uses_each_year_rest_and_makeup_days(day, expected):
    assert preparation.uses_rest_schedule(date.fromisoformat(day)) is expected


def test_formula_matches_original_for_2023_without_rain_as_driver():
    axis = pd.date_range("2023-01-01", "2024-01-01", freq="6h", inclusive="left")
    weather = pd.DataFrame({"datetime": axis, "temperature_C": 28.0,
                            "relative_humidity_pct": 75.0, "irradiance_W_m2": 350.0,
                            "rainfall_mm": np.nan, "load_weather_valid": 1,
                            "weather_history_valid": 1})
    coefficients = {name: 1.5 for name in END_USES}
    expected = original.build_load_15min(weather, coefficients)
    actual = preparation.build_load(weather, coefficients)
    for column in (*END_USES, "load_kW", "occupancy_factor", "operation_factor"):
        np.testing.assert_allclose(actual[column], expected[column], rtol=0, atol=0)
    assert actual.load_scenario_valid.eq(1).all()
    assert "rainfall_mm" not in actual


def test_missing_weather_stays_invalid_without_future_fill():
    minutes = pd.date_range("2021-07-01", periods=8 * 15, freq="min")
    source = pd.Series(30.0, index=minutes)
    source.iloc[15:105] = np.nan
    weather = original.aggregate_weather_15min({name: source for name in preparation.WEATHER_DRIVERS}, minutes[::15])
    weather = weather.rename(columns={"weather_history_valid": "load_weather_valid"})
    load = preparation.build_load(weather, {name: 1.0 for name in END_USES})
    assert load.load_scenario_valid.tolist() == [1, 1, 1, 1, 1, 0, 0, 1]
    assert np.isfinite(load[list(END_USES)].to_numpy()).all()


def test_tier_sums_reconstruct_original_load():
    frame = pd.DataFrame({"datetime": pd.date_range("2022-01-29", periods=4, freq="15min"),
                          **{name: np.arange(4, dtype=float) + i for i, name in enumerate(END_USES)}})
    frame["load_kW"] = frame[list(END_USES)].sum(axis=1)
    tiers = preparation.build_tiers(frame, ALLOCATIONS)
    for scenario in ALLOCATIONS:
        np.testing.assert_allclose(tiers[[f"{scenario}_{tier}_kW" for tier in ("CL1", "CL2", "CL3")]].sum(axis=1), frame.load_kW)


def test_reference_2023_preservation_detects_any_numeric_change():
    axis = pd.date_range("2023-01-01", periods=2, freq="15min")
    dispatch = pd.DataFrame({column: 1.0 for column in preparation.DISPATCH_COLUMNS if column != "datetime"}, index=range(2))
    dispatch["datetime"] = axis
    dispatch["tariff_period"] = "valley"
    dispatch = dispatch[preparation.DISPATCH_COLUMNS]
    tiers = pd.DataFrame({"datetime": axis, "load_kW": [1.0, 1.0]})
    preparation.verify_2023_preserved(dispatch.copy(), tiers.copy(), dispatch, tiers)
    altered = dispatch.copy()
    altered.loc[0, "import_price"] += 1e-12
    with pytest.raises(ValueError, match="2023 frozen dispatch changed"):
        preparation.verify_2023_preserved(altered, tiers, dispatch, tiers)


def test_write_once_refuses_existing_output_before_reading_inputs(tmp_path):
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        preparation.freeze(tmp_path, tmp_path / "missing_reference", tmp_path / "missing_pv")

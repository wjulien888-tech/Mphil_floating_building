"""Behavioural regression checks for the frozen M0 preprocessing path."""

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align_hkust_pv_15min import clean_measured_sites
from build_pv_disturbance_15min import build_disturbance_factors, apply_disturbance
from load_generation_15min import aggregate_weather_15min, build_load_15min


class CausalPreprocessingTests(unittest.TestCase):
    def test_pv_future_observations_do_not_rewrite_past(self):
        index = pd.date_range("2023-07-01", "2023-10-01", freq="15min", inclusive="left")
        daily = np.maximum(0, np.sin((index.hour.to_numpy() - 6) / 12 * np.pi)) * 30
        raw = pd.DataFrame({"site_a": daily, "site_b": daily * 0.5}, index=index)
        cutoff = pd.Timestamp("2023-09-01")
        raw.loc["2023-08-31 12:00", "site_a"] = np.nan
        clean = clean_measured_sites(raw)
        altered = raw.copy()
        altered.loc[cutoff:, :] = 1_000_000
        clean_altered = clean_measured_sites(altered)
        pd.testing.assert_frame_equal(clean.loc[clean.datetime < cutoff], clean_altered.loc[clean_altered.datetime < cutoff])
        first = build_disturbance_factors(clean, index)
        second = build_disturbance_factors(clean_altered, index)
        pd.testing.assert_frame_equal(first.loc[first.datetime < cutoff], second.loc[second.datetime < cutoff])
        gap = first[first.datetime.eq("2023-08-31 12:00")].iloc[0]
        self.assertEqual(gap.pv_target_valid, 0)
        self.assertEqual(gap.pv_history_valid, 1)

    def test_baseline_excludes_current_day_and_requires_history(self):
        index = pd.date_range("2023-01-01", periods=10 * 96, freq="15min")
        raw = pd.DataFrame({"site": np.full(len(index), 10.0)}, index=index)
        raw.loc["2023-01-09 12:00", "site"] = 1000
        result = build_disturbance_factors(clean_measured_sites(raw), index).set_index("datetime")
        self.assertEqual(result.loc["2023-01-09 12:00", "hkust_clear_like_baseline_kW"], 10)
        self.assertTrue(result.iloc[:7 * 96].pv_target_valid.eq(0).all())
        self.assertTrue(result.iloc[7 * 96:].pv_target_valid.eq(1).all())

    def test_weather_long_gaps_are_not_valid_and_rain_is_not_observed_zero(self):
        minute_index = pd.date_range("2023-08-31", "2023-09-02", freq="min", inclusive="left")
        values = pd.Series(25.0, index=minute_index)
        values.loc["2023-08-31 06:00":"2023-08-31 17:59"] = np.nan
        rain = values / 100
        index = pd.date_range("2023-08-31", "2023-09-02", freq="15min", inclusive="left")
        sources = {"temperature_C": values, "rainfall_mm": rain}
        weather = aggregate_weather_15min(sources, index).set_index("datetime")
        self.assertEqual(weather.loc["2023-08-31 06:00", "temperature_C_filled"], 1)
        self.assertEqual(weather.loc["2023-08-31 07:00", "temperature_C_valid"], 0)
        self.assertEqual(weather.loc["2023-08-31 07:00", "temperature_C_observation_count"], 0)
        self.assertEqual(weather.loc["2023-08-31 06:00", "rainfall_mm_valid"], 0)
        self.assertEqual(weather.loc["2023-08-31 06:00", "rainfall_mm_placeholder"], 1)
        self.assertEqual(weather.loc["2023-08-31 18:00", "temperature_C_observation_count"], 15)
        altered = {name: source.copy() for name, source in sources.items()}
        for source in altered.values():
            source.loc["2023-09-01":] = 1000
        changed = aggregate_weather_15min(altered, index).set_index("datetime")
        pd.testing.assert_frame_equal(weather.loc[:"2023-08-31"], changed.loc[:"2023-08-31"])

    def test_weather_partial_coverage_is_recorded(self):
        minutes = pd.date_range("2023-01-01", periods=30, freq="min")
        values = pd.Series(20.0, index=minutes)
        values.iloc[:4] = np.nan
        result = aggregate_weather_15min({"temperature_C": values}, minutes[::15])
        self.assertEqual(result.iloc[0].temperature_C_observation_count, 11)
        self.assertEqual(result.iloc[0].temperature_C_partial, 1)
        self.assertEqual(result.iloc[0].temperature_C_valid, 0)
        self.assertEqual(result.iloc[1].temperature_C_valid, 1)

    def test_fixed_load_coefficients_do_not_use_future_weather(self):
        index = pd.date_range("2023-08-01", "2023-10-01", freq="15min", inclusive="left")
        weather = pd.DataFrame({"datetime": index, "temperature_C": 28.0,
                                "relative_humidity_pct": 70.0, "irradiance_W_m2": 300.0,
                                "rainfall_mm": 0.0, "weather_history_valid": 1})
        coefficients = {name: 1.5 for name in ("base_kW", "plug_kW", "lighting_kW", "hvac_kW", "aux_kW")}
        before = build_load_15min(weather, coefficients)
        weather.loc[weather.datetime.ge("2023-09-01"), "temperature_C"] = 80
        after = build_load_15min(weather, coefficients)
        pd.testing.assert_frame_equal(before.loc[before.datetime.lt("2023-09-01")], after.loc[after.datetime.lt("2023-09-01")])

    def test_training_pv_has_no_retrospective_energy_normalization(self):
        time = pd.date_range("2023-01-01", periods=2, freq="15min")
        baseline = pd.DataFrame({"datetime": time, "pvgis_floating_baseline_kW": [20., 20.]})
        factors = pd.DataFrame({"datetime": time, "disturbance_factor": [.5, .2], "pv_target_valid": [1, 0]})
        result = apply_disturbance(baseline, factors)
        self.assertEqual(result.pv_floating_disturbed_raw_kW.tolist(), [10., 4.])
        self.assertEqual(result.pv_target_valid.tolist(), [1, 0])
        self.assertNotIn("annual_energy_normalization", result)


if __name__ == "__main__":
    unittest.main()

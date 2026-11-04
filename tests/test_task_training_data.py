"""Data-boundary tests: no solver or formal model training is executed."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_variants"))
from _task_training_data import prepare_training_data, validate_checkpoint_splits, write_event_tables


def sources(tmp_path, *, invalid_pv=None, invalid_load=None):
    axis = pd.date_range("2021-06-30", "2023-11-02", freq="15min", inclusive="left")
    split = np.where(axis < pd.Timestamp("2023-09-01"), "train",
                     np.where(axis < pd.Timestamp("2023-11-01"), "val", "test"))
    pv = pd.DataFrame({"datetime": axis, "pv": 2., "history_valid": 1,
                       "pv_target_valid": 1, "split": split})
    # Test values are deliberately unusable and cannot affect selection.
    pv.loc[pv.split == "test", "pv"] = np.nan
    if invalid_pv is not None:
        pv.loc[pv.datetime.eq(pd.Timestamp(invalid_pv)), "pv_target_valid"] = 0
    pv_path = tmp_path / "pv.csv"
    pv.to_csv(pv_path, index=False)
    dispatch = pd.DataFrame({"datetime": axis, "load_kW": 3., "import_price": .3,
                             "export_price": .2, "grid_available": 1., "load_scenario_valid": 1})
    if invalid_load is not None:
        dispatch.loc[dispatch.datetime.eq(pd.Timestamp(invalid_load)), "load_scenario_valid"] = 0
    dispatch_path = tmp_path / "dispatch.csv"
    dispatch.to_csv(dispatch_path, index=False)
    tiers = pd.DataFrame({"datetime": axis, "load_kW": 3., "minimum_operation_CL1_kW": 1.,
                         "minimum_operation_CL2_kW": 1., "minimum_operation_CL3_kW": 1.})
    tiers_path = tmp_path / "tiers.csv"
    tiers.to_csv(tiers_path, index=False)
    config = json.loads((ROOT / "configs/m0_s0_s5_v2.json").read_text())
    config["cases"] = [row for row in config["cases"] if row["name"] in {"S0_grid", "S4_observed_low_outage_24h"}]
    config["s5_grid"]["initial_soc"] = []
    config_path = tmp_path / "scenarios.json"
    config_path.write_text(json.dumps(config))
    dataset = SimpleNamespace(source_frame=pv, lookback_window=96, prediction_horizon=96,
                              pv_min=0., pv_max=8., config=SimpleNamespace(data_path=pv_path, split_windows={
                                  "train": SimpleNamespace(start="2021-07-01", end="2023-09-01"),
                                  "val": SimpleNamespace(start="2023-09-01", end="2023-11-01")}))
    return dataset, config_path, dispatch_path, tiers_path


def test_monthly_episodes_preserve_units_scaler_and_do_not_reach_test(tmp_path):
    bundle = prepare_training_data(*sources(tmp_path))
    assert bundle.metadata["episode_counts"] == {"train": 52, "val": 4}
    assert bundle.metadata["requested_episode_counts"] == {"train": 52, "val": 4}
    assert {e.metadata["month"] for e in bundle.episodes["train"]} == {
        str(month) for month in pd.period_range("2021-07", "2023-08", freq="M")}
    for split, episodes in bundle.episodes.items():
        for episode in episodes:
            assert episode.steps == 288  # 24h before + 24h event + 24h after
            assert len(episode.frame) == episode.steps + 95
            assert len(episode.histories) == episode.steps + 95 + 96
            assert torch.all(episode.histories == .25)  # original scaler, no refit
            assert episode.frame.pv.eq(2.).all()  # physical kW, no PV scaling
            assert episode.case.event_hour == 24
            assert episode.frame.index[0].month == episode.frame.index[-1].month
            assert episode.frame.index[-1] < pd.Timestamp("2023-09-01" if split == "train" else "2023-11-01")
            assert "episode start" in episode.metadata["initial_soc_meaning"]
            assert episode.metadata["pv_repairs"] == 0
    low = next(e for e in bundle.episodes["train"] if e.name == "2023-01_S4_observed_low_outage_24h")
    assert low.metadata["event_start"] == "2023-01-02 12:00:00"  # earliest equal-energy eligible window
    assert (low.frame.grid_available == 0).sum() == 96
    assert bundle.metadata["sources"]["pv_timeseries"]["sha256"]
    json.dumps(bundle.metadata)  # manifest must be directly serializable
    write_event_tables(bundle, tmp_path / "tables")
    table = pd.read_csv(tmp_path / "tables/events.csv")
    coverage = pd.read_csv(tmp_path / "tables/monthly_coverage.csv")
    assert len(table) == 56
    assert coverage.included.sum() == 56
    assert coverage.loc[coverage.split.eq("train"), "month"].nunique() == 26


def test_invalid_target_in_history_excludes_candidate_without_repair(tmp_path):
    bundle = prepare_training_data(*sources(tmp_path, invalid_pv="2023-01-02 12:00"))
    low = next(e for e in bundle.episodes["train"] if e.name == "2023-01_S4_observed_low_outage_24h")
    # The candidate at Jan 4 noon would still include the invalid point in
    # its historical lookback, so selection advances to Jan 5 noon.
    assert low.metadata["event_start"] == "2023-01-05 12:00:00"
    selection = next(s for s in bundle.metadata["selections"] if s["name"] == low.name)
    rejected = [r for r in selection["candidates"] if "invalid_or_missing_pv_history_or_target" in r["rejection_reasons"]]
    assert rejected and all(r["invalid_pv_points"] == 1 for r in rejected)


def test_invalid_fixed_event_uses_nearest_valid_date_and_checks_forecast_tail(tmp_path):
    # S0: event Jan 10 noon, execution Jan 9 noon--Jan 12 noon;
    # Jan 13 morning is in its final planning window, not execution.
    bundle = prepare_training_data(*sources(tmp_path, invalid_load="2023-01-13 09:00"))
    episode = next(e for e in bundle.episodes["train"] if e.name == "2023-01_S0_grid")
    assert episode.metadata["nominal_event_start"] == "2023-01-10 12:00:00"
    assert episode.metadata["resolved_event_start"] == "2023-01-09 12:00:00"
    assert episode.metadata["shift_hours"] == -24
    assert "invalid_load_scenario" in episode.metadata["nominal_rejection_reasons"]
    assert episode.metadata["reason"] == "nominal_invalid_use_nearest_quality_valid_same_clock"
    assert bundle.metadata["excluded_counts"]["train"] == 0


def test_wrong_split_or_horizon_fails_explicitly(tmp_path):
    args = sources(tmp_path)
    args[0].config.split_windows["val"].start = "2023-08-01"
    with pytest.raises(ValueError, match="contiguous and nonoverlapping"):
        prepare_training_data(*args)


def test_no_valid_month_is_excluded_instead_of_searching_test_dates(tmp_path):
    args = sources(tmp_path)
    dispatch = pd.read_csv(args[2], parse_dates=["datetime"])
    dispatch.loc[(dispatch.datetime >= "2023-10-01") & (dispatch.datetime < "2023-11-01"), "load_scenario_valid"] = 0
    dispatch.to_csv(args[2], index=False)
    bundle = prepare_training_data(*args)
    assert not any(e.metadata["month"] == "2023-10" for e in bundle.episodes["val"])
    excluded = next(s for s in bundle.metadata["selections"] if s["name"] == "2023-10_S0_grid")
    assert excluded["status"] == "excluded"
    assert excluded["nominal_event_start"] == "2023-10-10 12:00:00"
    assert all(pd.Timestamp(row["event_start"]) < pd.Timestamp("2023-11-01") for row in excluded["candidates"])
    assert all(e.frame.index[-1] < pd.Timestamp("2023-11-01") for e in bundle.episodes["val"])


@pytest.mark.parametrize("source_index", [2, 3])
def test_2023_only_dispatch_or_tiers_cannot_silently_shorten_training(tmp_path, source_index):
    args = sources(tmp_path)
    path = args[source_index]
    frame = pd.read_csv(path, parse_dates=["datetime"])
    frame.loc[frame.datetime.ge("2023-01-01")].to_csv(path, index=False)
    with pytest.raises(ValueError, match="refusing to silently shorten training"):
        prepare_training_data(*args)


def test_missing_pv_year_is_an_error_even_when_later_months_are_valid(tmp_path):
    args = sources(tmp_path)
    args[0].source_frame = args[0].source_frame.loc[args[0].source_frame.datetime.ge("2023-01-01")]
    with pytest.raises(ValueError, match="PV history and targets must cover"):
        prepare_training_data(*args)


def test_checkpoint_declared_new_dates_are_accepted_but_overlap_rejected():
    config = {"split_windows": {"train": {"start": "2021-07-01", "end": "2023-09-01"},
                                "val": {"start": "2023-09-01", "end": "2023-11-01"}}}
    validate_checkpoint_splits(config)
    config["split_windows"] = {"train": {"start": "2021-07-01", "end": "2023-01-01"},
                               "val": {"start": "2023-01-01", "end": "2023-03-01"},
                               "test": {"start": "2023-03-01", "end": "2024-03-01"}}
    assert str(validate_checkpoint_splits(config)[1]) == "2023-01-01 00:00:00"
    config["split_windows"]["test"]["start"] = "2023-02-01"
    with pytest.raises(ValueError, match="without overlap"):
        validate_checkpoint_splits(config)

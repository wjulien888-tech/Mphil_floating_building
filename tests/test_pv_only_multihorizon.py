"""Data-only contracts for common-origin 1/2/3-day to 1-day PV forecasts."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_pv_only_multihorizon import (
    ANCHOR_LOOKBACK, HORIZON, LOOKBACKS, export_arrays, file_record, freeze,
    make_datasets, read_source, sha256,
)
from configurations.hkust_15min import SplitWindow
from model_variants._pv_only_dataset import PVForecastConfig, PVOnlyDataset


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    timestamps = pd.date_range("2021-07-01", periods=12 * 96, freq="15min")
    splits = {"train": ("2021-07-01", "2021-07-07"),
              "val": ("2021-07-07", "2021-07-10"),
              "test": ("2021-07-10", "2021-07-13")}
    frame = pd.DataFrame({"datetime": timestamps, "pv_kW": np.arange(len(timestamps)) / 10,
                          "history_valid": 1, "pv_target_valid": 1, "split": "excluded",
                          "forecast_origin_valid": 1})
    for name, (start, end) in splits.items():
        frame.loc[frame.datetime.ge(start) & frame.datetime.lt(end), "split"] = name
    frame.to_csv(directory / "pv_timeseries.csv", index=False)
    manifest = {
        "dataset_version": "constructed_pv_test_source",
        "provenance": "Constructed PV from physical baseline and causal measured rooftop disturbance.",
        "splits": {name: {"start_inclusive": start, "end_exclusive": end}
                   for name, (start, end) in splits.items()},
        "outputs": [file_record(directory / "pv_timeseries.csv", directory)],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, frame, manifest


def configs(source):
    directory, _, manifest = source
    splits = {name: SplitWindow(record["start_inclusive"], record["end_exclusive"])
              for name, record in manifest["splits"].items()}
    return {lookback: PVForecastConfig(data_path=directory / "pv_timeseries.csv",
                                     lookback_steps=lookback, prediction_horizon=HORIZON,
                                     anchor_lookback_steps=ANCHOR_LOOKBACK, split_windows=splits)
            for lookback in LOOKBACKS}


def test_identical_origins_targets_and_complete_split_boundaries(source):
    datasets = make_datasets(source[0], source[2])
    raw = source[1].pv_kW.to_numpy(dtype=np.float32)
    for name in ("train", "val", "test"):
        reference = datasets[96].for_split(name)
        for lookback, dataset in datasets.items():
            view = dataset.for_split(name)
            np.testing.assert_array_equal(view._anchors, reference._anchors)
            arrays = export_arrays(view)
            assert arrays["X"].shape == (len(view), lookback, 1)
            assert arrays["y_kw"].shape == (len(view), 96, 1)
            window = view.config.split_windows[name]
            times = pd.to_datetime(arrays["future_timestamps"].ravel(), unit="s")
            assert (times >= window.start).all() and (times < window.end).all()
            for index in (0, len(view) - 1):
                anchor = view._anchors[index]
                item = view[index]
                np.testing.assert_array_equal(item["history"], arrays["X"][index])
                np.testing.assert_array_equal(item["pv_target"], arrays["y_kw"][index])
                np.testing.assert_array_equal(item["future_timestamps"], arrays["future_timestamps"][index])
                np.testing.assert_array_equal(item["persistence"][:, 0], raw[anchor - 96:anchor])
                np.testing.assert_array_equal(item["persistence_last"][:, 0], [raw[anchor - 1]] * 96)
                np.testing.assert_array_equal(item["pv_target"], reference[index]["pv_target"])
    assert datasets[96]._anchors[0] == 288
    assert datasets[96]._anchors[-1] == 480  # target ends at final training row 575
    assert datasets[96].for_split("val")._anchors[0] == 576  # history may precede split


def test_shared_long_history_guard_and_every_future_quality_flag(source):
    directory, frame, _ = source
    frame.loc[100, "history_valid"] = 0  # outside the one-day window at anchor 288
    frame.loc[430, "pv_target_valid"] = 0
    frame.to_csv(directory / "pv_timeseries.csv", index=False)
    for config in configs(source).values():
        dataset = PVOnlyDataset(config)
        anchors = dataset._anchors
        assert not set(range(288, 389)).intersection(anchors)
        assert not set(range(335, 431)).intersection(anchors)
        assert anchors[0] == 431
        assert 480 in anchors


def test_heldout_values_cannot_change_training_scaler_windows_or_targets(source):
    directory, frame, _ = source
    config = configs(source)[192]
    before = PVOnlyDataset(config)
    frame.loc[frame.split.ne("train"), "pv_kW"] += 10000
    frame.to_csv(directory / "pv_timeseries.csv", index=False)
    after = PVOnlyDataset(config)
    assert before.pv_min == after.pv_min and before.pv_max == after.pv_max
    for key, array in export_arrays(before).items():
        np.testing.assert_array_equal(array, export_arrays(after)[key])
    assert export_arrays(after.for_split("val"))["y_scaled"].min() > 1


@pytest.mark.parametrize("field,value", [("lookback_steps", 0), ("prediction_horizon", -1),
                                         ("anchor_lookback_steps", 1.5)])
def test_invalid_window_parameters_rejected(source, field, value):
    with pytest.raises(ValueError, match="positive integer"):
        PVOnlyDataset(replace(configs(source)[96], **{field: value}))


def test_freeze_preserves_source_verifies_hashes_and_refuses_overwrite(source, tmp_path):
    directory, _, _ = source
    source_hash = sha256(directory / "pv_timeseries.csv")
    output = tmp_path / "frozen"
    suite = freeze(directory, output)
    assert suite["common_forecast_origins"] is True
    anchors = {}
    for lookback in LOOKBACKS:
        frozen = output / f"lookback_{lookback}"
        manifest = json.loads((frozen / "manifest.json").read_text())
        assert manifest["schema_version"] == 3
        assert manifest["artifact_type"] == "pv_only_forecast_multistep"
        copied = pd.read_csv(frozen / "pv_timeseries.csv")
        original = pd.read_csv(directory / "pv_timeseries.csv")
        stable_columns = original.columns.drop("forecast_origin_valid")
        pd.testing.assert_frame_equal(copied[stable_columns], original[stable_columns])
        assert manifest["source_pv_timeseries_sha256"] == source_hash
        assert manifest["parameters"]["anchor_lookback_steps"] == 288
        assert manifest["parameters"]["forecast_origin_valid_recomputed"] is True
        index = pd.read_csv(frozen / "sample_index.csv")
        np.testing.assert_array_equal(np.flatnonzero(copied.forecast_origin_valid), index.anchor_index)
        for record in manifest["outputs"]:
            assert file_record(frozen / record["path"], frozen) == record
        for name in ("train", "val", "test"):
            with np.load(frozen / f"{name}.npz") as data:
                np.testing.assert_array_equal(data["anchor_indices"], index[index.split == name].anchor_index)
                if name in anchors:
                    np.testing.assert_array_equal(anchors[name], data["anchor_indices"])
                else:
                    anchors[name] = data["anchor_indices"].copy()
                assert list(data["y_kw"].shape) == manifest["splits"][name]["y_shape"]
    assert sha256(directory / "pv_timeseries.csv") == source_hash
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        freeze(directory, output)


def test_tampered_source_is_rejected_before_any_output_created(source, tmp_path):
    directory, frame, _ = source
    frame.loc[0, "pv_kW"] += 1
    frame.to_csv(directory / "pv_timeseries.csv", index=False)
    with pytest.raises(ValueError, match="frozen manifest"):
        read_source(directory)
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="frozen manifest"):
        freeze(directory, output)
    assert not output.exists()

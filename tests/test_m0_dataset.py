"""Research-critical chronology, quality, scaler and baseline regressions."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "ol-ems-hkust"))
from configurations.hkust_15min import DEFAULT_CONFIG, SplitWindow
from src.dataloader.csv_dataset import CSVDataset


def make_data(tmp_path):
    times = pd.date_range("2023-01-01", periods=15 * 96, freq="15min")
    slot = np.arange(len(times)) % 96
    pv = np.maximum(np.sin((slot - 24) * np.pi / 48), 0.0) * 20
    frame = pd.DataFrame({
        "datetime": times,
        "pv_floating_disturbed_raw_kW": pv,
        "load_kW": 3.0,
        "temperature_C": 20 + np.arange(len(times)) / len(times),
        "relative_humidity_pct": 70.0,
        "irradiance_W_m2": pv * 20,
        "rainfall_mm": 0.0,
        "import_price": 1.0,
        "export_price": 0.3,
        "grid_available": 1.0,
        "history_valid": 1,
        "pv_target_valid": 1,
    })
    tiers = pd.DataFrame({
        "datetime": times,
        "minimum_operation_CL1_kW": 0.5,
        "minimum_operation_CL2_kW": 1.0,
    })
    csv = tmp_path / "dispatch.csv"
    tier_path = tmp_path / "load_tiers.csv"
    tiers.to_csv(tier_path, index=False)
    config = replace(DEFAULT_CONFIG, data_path=csv, load_tiers_path=tier_path,
                     lookback_steps=96, split_windows={
                         "train": SplitWindow("2023-01-01", "2023-01-06"),
                         "val": SplitWindow("2023-01-06", "2023-01-11"),
                         "test": SplitWindow("2023-01-11", "2023-01-16"),
                     })
    frame.to_csv(csv, index=False)
    return frame, config


def test_quality_filters_windows_without_deleting_timestamps(tmp_path):
    frame, config = make_data(tmp_path)
    frame.loc[200, "history_valid"] = 0
    frame.loc[350, "pv_target_valid"] = 0
    frame.loc[600:610, "history_valid"] = 0
    frame.to_csv(config.data_path, index=False)
    dataset = CSVDataset(config=config)
    assert len(dataset.source_frame) == len(frame)
    for name, view in dataset.split_views().items():
        for anchor in view._anchors:
            assert frame.history_valid.iloc[anchor - 96:anchor].all()
            assert frame.pv_target_valid.iloc[anchor:anchor + 96].all()
            assert frame.datetime.iloc[anchor] >= pd.Timestamp(config.split_windows[name].start)
            assert frame.datetime.iloc[anchor + 95] < pd.Timestamp(config.split_windows[name].end)
    q = dataset.validation_summary()["quality"]
    assert q["invalid_history_rows"] == 12
    assert q["invalid_pv_target_rows"] == 1
    assert q["samples_rejected_by_quality"]["train"] > 0
    assert q["samples_rejected_by_quality"]["val"] > 0


def test_scaler_excludes_bad_rows_and_never_fits_heldout(tmp_path):
    frame, config = make_data(tmp_path)
    frame.loc[200, "history_valid"] = 0
    frame.to_csv(config.data_path, index=False)
    original = CSVDataset(config=config)
    for col in config.history_features:
        source = config.source_columns[col]
        frame.loc[200, source] = 1e6
        frame.loc[480:, source] = 2e6
    frame.to_csv(config.data_path, index=False)
    changed = CSVDataset(config=config)
    torch.testing.assert_close(original.scaling_factors_min, changed.scaling_factors_min)
    torch.testing.assert_close(original.scaling_factors_max, changed.scaling_factors_max)
    assert changed.validation_summary()["quality"]["scaler_valid_training_rows"] == 479


def test_quality_flags_required_and_binary(tmp_path):
    frame, config = make_data(tmp_path)
    frame.drop(columns="history_valid").to_csv(config.data_path, index=False)
    with pytest.raises(ValueError, match="history_valid"):
        CSVDataset(config=config)
    frame.loc[0, "history_valid"] = 0.5
    frame.to_csv(config.data_path, index=False)
    with pytest.raises(ValueError, match="only 0 or 1"):
        CSVDataset(config=config)


def test_forecast_view_and_previous_day_use_only_past_values(tmp_path):
    frame, config = make_data(tmp_path)
    dataset = CSVDataset(config=config).for_split("val")
    anchor = int(dataset._anchors[0])
    fast = dataset.forecast_view()[0]
    regular = dataset[0]
    torch.testing.assert_close(fast["history"], regular["history"])
    torch.testing.assert_close(fast["pv_target"], regular["pv_target"])
    expected = torch.tensor(frame.pv_floating_disturbed_raw_kW.iloc[anchor-96:anchor].to_numpy(), dtype=torch.float32).unsqueeze(-1)
    torch.testing.assert_close(fast["persistence"], expected)
    frame.loc[anchor:, "pv_floating_disturbed_raw_kW"] *= 3
    frame.to_csv(config.data_path, index=False)
    altered = CSVDataset(config=config).for_split("val").forecast_view()[0]
    torch.testing.assert_close(fast["history"], altered["history"])
    torch.testing.assert_close(fast["persistence"], altered["persistence"])
    assert not torch.equal(fast["pv_target"], altered["pv_target"])


def test_scaler_restore_rejects_invalid_or_wrong_interval(tmp_path):
    _, config = make_data(tmp_path)
    dataset = CSVDataset(config=config)
    bad = dataset.state_dict()
    bad["scaling_factors_min"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        dataset.load_state_dict(bad)
    bad = dataset.state_dict()
    bad["scaler_fit_interval"] = ("2023-01-01", "2023-01-11")
    with pytest.raises(ValueError, match="differs"):
        dataset.load_state_dict(bad)


def test_closed_loop_rejects_quality_gap_before_simulation(tmp_path):
    from src.training_and_evaluation.learner import OnlineLearner
    frame, config = make_data(tmp_path)
    frame.loc[600, "history_valid"] = 0
    frame.to_csv(config.data_path, index=False)
    dataset = CSVDataset(config=config).for_split("val")
    learner = OnlineLearner(torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="continuous timestamps"):
        learner.run_closed_loop(dataset, steps=len(dataset))

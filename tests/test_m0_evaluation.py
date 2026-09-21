"""Evaluation uses its own frozen prediction results and past-only persistence."""
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external/ol-ems-hkust"))
from src.training_and_evaluation.m0_evaluation import ErrorAccumulator, evaluate_m0, persistence_from_history


def test_error_metrics_are_pair_weighted_with_signed_bias():
    accumulator = ErrorAccumulator(2)
    truth = np.array([[1., 2.], [3., 4.]])
    predictions = np.array([[2., 0.], [4., 0.]])
    accumulator.update(predictions, truth, np.ones_like(truth, dtype=bool))
    result = accumulator.metrics()
    assert result == {"count": 4, "mae_kw": 2., "rmse_kw": np.sqrt(5.5),
                      "mbe_kw": -1., "mse_kw2": 5.5}
    assert accumulator.metrics(0)["mbe_kw"] == 1.


def test_daily_persistence_uses_last_96_observations():
    class Bounds:
        prediction_horizon = 96
        input_features = ["temperature", "pv"]
        def denormalise_pv(self, x):
            return 100 * x
    history = torch.arange(100 * 2).reshape(1, 100, 2).float()
    result = persistence_from_history(history, Bounds())
    assert torch.equal(result, history[:, -96:, 1:2] * 100)


class EvaluationDataset(Dataset):
    split = "test"
    prediction_horizon = 2
    def __len__(self):
        return 3
    def __getitem__(self, index):
        timestamp = 1672531200 + index * 900
        return {"history": torch.zeros(2, 1), "pv_target": torch.tensor([[0.], [2.]]),
                "persistence": torch.tensor([[0.], [1.]]), "timestamp": torch.tensor(timestamp),
                "future_timestamps": torch.tensor([timestamp, timestamp + 900])}


class FixedForecast(torch.nn.Module):
    def forecast_pv(self, history):
        return torch.tensor([[[1.], [3.]]]).repeat(len(history), 1, 1)
    def forward(self, *args):
        raise AssertionError("Forecast evaluation must not run MPC")


def test_evaluation_exports_full_split_and_daylight_metrics(tmp_path):
    args = Namespace(device="cpu", val_batch_size=2, eval_limit_batches=1., daylight_threshold_kw=0.1)
    summary = evaluate_m0(FixedForecast(), EvaluationDataset(), tmp_path, args, {})
    assert summary["forecast_origins"] == 3
    assert summary["complete_split"]
    frame = pd.read_csv(tmp_path / "predictions.csv.gz")
    assert len(frame) == 6
    assert frame["daylight_by_truth"].sum() == 3
    by_horizon = pd.read_csv(tmp_path / "horizon_metrics.csv")
    assert len(by_horizon) == 12
    m0 = [row for row in summary["metrics"] if row["method"] == "M0" and row["period"] == "all"][0]
    assert m0["mae_kw"] == m0["rmse_kw"] == m0["mbe_kw"] == 1.

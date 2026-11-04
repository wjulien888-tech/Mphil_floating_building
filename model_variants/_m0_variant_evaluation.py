"""Variant-local independent forecast evaluation, including R2 and persistence.

Copied from the frozen M0 evaluator; the original evaluator remains unchanged.
R2 is pooled across origin/horizon pairs, not squared Pearson correlation.
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training_and_evaluation.m0_training import _forecast_view, batch_count, json_write

LOGGER = logging.getLogger("hkust_ol_ems")


class ErrorAccumulator:
    """Sufficient statistics across forecast origins, separately by horizon."""
    def __init__(self, horizon: int):
        self.count = np.zeros(horizon, dtype=np.int64)
        self.signed = np.zeros(horizon, dtype=np.float64)
        self.absolute = np.zeros(horizon, dtype=np.float64)
        self.squared = np.zeros(horizon, dtype=np.float64)
        self.target_sum = np.zeros(horizon, dtype=np.float64)
        self.target_squared = np.zeros(horizon, dtype=np.float64)

    def update(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        error = prediction.astype(np.float64) - target.astype(np.float64)
        self.count += mask.sum(axis=0)
        self.signed += np.where(mask, error, 0).sum(axis=0)
        self.absolute += np.where(mask, np.abs(error), 0).sum(axis=0)
        self.squared += np.where(mask, error ** 2, 0).sum(axis=0)
        truth = target.astype(np.float64)
        self.target_sum += np.where(mask, truth, 0).sum(axis=0)
        self.target_squared += np.where(mask, truth ** 2, 0).sum(axis=0)

    def metrics(self, index: int | None = None) -> dict[str, Any]:
        count = int(self.count.sum() if index is None else self.count[index])
        if not count:
            return {"count": 0, "mae_kw": None, "rmse_kw": None, "mbe_kw": None, "mse_kw2": None, "r2": None}
        def value(array: np.ndarray) -> float:
            return float(array.sum() if index is None else array[index]) / count
        total_y = float(self.target_sum.sum() if index is None else self.target_sum[index])
        total_y2 = float(self.target_squared.sum() if index is None else self.target_squared[index])
        total_error2 = float(self.squared.sum() if index is None else self.squared[index])
        sst = total_y2 - total_y * total_y / count
        tolerance = np.finfo(np.float64).eps * max(abs(total_y2), 1.0) * 16
        r2 = float(1.0 - total_error2 / sst) if sst > tolerance else None
        return {"count": count, "mae_kw": value(self.absolute), "rmse_kw": value(self.squared) ** 0.5,
                "mbe_kw": value(self.signed), "mse_kw2": value(self.squared), "r2": r2}


def persistence_from_history(history: torch.Tensor, dataset: Any) -> torch.Tensor:
    """Yesterday's same slot (fixed lag 96), independent of forecast length."""
    horizon = dataset.prediction_horizon
    if history.shape[1] < 96 or not 1 <= horizon <= 96:
        raise ValueError("Daily persistence requires 96 history steps and horizon in [1, 96]")
    index = dataset.input_features.index("pv")
    start = history.shape[1] - 96
    return dataset.denormalise_pv(history[:, start:start + horizon, index:index + 1])


def last_value_from_history(history: torch.Tensor, dataset: Any) -> torch.Tensor:
    """The last observed PV value, repeated over the requested forecast horizon."""
    index = dataset.input_features.index("pv")
    last = dataset.denormalise_pv(history[:, -1:, index:index + 1])
    return last.expand(-1, dataset.prediction_horizon, -1)


def evaluate_m0(controller: torch.nn.Module, dataset: Any, output_dir: Path,
                args: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    if args.daylight_threshold_kw < 0:
        raise ValueError("Daylight threshold must be non-negative")
    device = torch.device(args.device)
    controller.to(device).eval()
    loader = DataLoader(_forecast_view(dataset), batch_size=args.val_batch_size, shuffle=False,
                        num_workers=0)
    count = batch_count(len(loader), args.eval_limit_batches)
    horizon = dataset.prediction_horizon
    accumulators = {(method, period): ErrorAccumulator(horizon)
                    for method in ("M0", "persistence_24h", "persistence_last")
                    for period in ("all", "daylight", "non_daylight")}
    origins = 0
    first_timestamp, last_timestamp = None, None
    started = time.perf_counter()
    destination = output_dir / "predictions.csv.gz"
    with gzip.open(destination, "wt", newline="", encoding="utf-8") as stream, torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= count:
                break
            history = batch["history"].to(device)
            target = batch["pv_target"].cpu().numpy().squeeze(-1)
            prediction = controller.forecast_pv(history).cpu().numpy().squeeze(-1)
            if "persistence" in batch:
                baseline = batch["persistence"].cpu().numpy().reshape(target.shape)
            else:
                baseline = persistence_from_history(batch["history"], dataset).cpu().numpy().squeeze(-1)
            if "persistence_last" in batch:
                last_baseline = batch["persistence_last"].cpu().numpy().reshape(target.shape)
            else:
                last_baseline = last_value_from_history(batch["history"], dataset).cpu().numpy().squeeze(-1)
            if not all(np.isfinite(array).all() for array in (prediction, target, baseline, last_baseline)):
                raise FloatingPointError("Non-finite forecasts/labels in evaluation")
            daylight = target > args.daylight_threshold_kw
            masks = {"all": np.ones_like(daylight), "daylight": daylight, "non_daylight": ~daylight}
            for method, values in (("M0", prediction), ("persistence_24h", baseline),
                                   ("persistence_last", last_baseline)):
                for period, mask in masks.items():
                    accumulators[method, period].update(values, target, mask)
            timestamps = batch["future_timestamps"].cpu().numpy()
            origin_timestamps = batch["timestamp"].cpu().numpy().reshape(-1)
            first_timestamp = first_timestamp or pd.Timestamp(int(origin_timestamps[0]), unit="s").isoformat()
            last_timestamp = pd.Timestamp(int(origin_timestamps[-1]), unit="s").isoformat()
            forecast_rows = pd.DataFrame({
                "forecast_origin": np.repeat(pd.to_datetime(origin_timestamps, unit="s").astype(str), horizon),
                "target_timestamp": pd.to_datetime(timestamps.reshape(-1), unit="s").astype(str),
                "horizon_step": np.tile(np.arange(1, horizon + 1), len(target)),
                "pv_actual_kw": target.reshape(-1), "pv_forecast_kw": prediction.reshape(-1),
                "persistence_24h_kw": baseline.reshape(-1),
                "persistence_last_kw": last_baseline.reshape(-1),
                "daylight_by_truth": daylight.reshape(-1)})
            forecast_rows.to_csv(stream, index=False, header=batch_index == 0, float_format="%.7g")
            origins += len(target)
    if not origins:
        raise ValueError("Evaluation split is empty")
    overall_rows, horizon_rows = [], []
    for (method, period), accumulator in accumulators.items():
        overall_rows.append({"method": method, "period": period, **accumulator.metrics()})
        horizon_rows.extend({"method": method, "period": period, "horizon_step": step + 1,
                             **accumulator.metrics(step)} for step in range(horizon))
    pd.DataFrame(overall_rows).to_csv(output_dir / "forecast_metrics.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(output_dir / "horizon_metrics.csv", index=False)
    summary = {**metadata, "status": "completed", "split": dataset.split,
               "forecast_origins": origins, "available_origins": len(dataset),
               "complete_split": origins == len(dataset), "first_origin": first_timestamp,
               "last_origin": last_timestamp, "horizon_steps": horizon,
               "elapsed_seconds": time.perf_counter() - started, "metrics": overall_rows,
               "evaluation_protocol": {
                   "aggregation": "Each valid (forecast origin, horizon step) pair has equal weight; overlapping target timestamps are counted for each issued forecast. No timestamp deduplication.",
                   "units": "MAE/RMSE/MBE in kW; MSE in kW^2; MBE = prediction minus truth.",
                   "r2": "1-SSE/SST within each reported group; null for constant or numerically indistinguishable targets; no horizon-wise averaging for overall R2.",
                   "daylight": f"Post-hoc reporting mask: actual target PV > {args.daylight_threshold_kw} kW. This is not a solar-elevation or weather feature and is never supplied to the model.",
                   "persistence": "For each future target t, persistence_24h uses PV[t-96] (24 hours ago). persistence_last repeats the last PV observation before the first target across the complete horizon. Both use only observations available when the forecast is issued.",
                   "task": f"{dataset.lookback_window} observed quarter-hour PV steps directly predict the next {horizon} quarter-hour PV targets. Each origin contributes {horizon} forecast/target pairs.",
                   "selection": "No weight updates, no checkpoint selection, no MPC, no forecast clipping or calibration on this split.",
                   "data": "Constructed floating-PV scenario; metrics do not establish measured floating-building forecast performance."}}
    json_write(output_dir / "evaluation_summary.json", summary)
    LOGGER.info("%s forecast evaluation: %s", dataset.split, overall_rows)
    return summary

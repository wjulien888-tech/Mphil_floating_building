"""Audit annual inputs and print a manual initial-training command. Never train."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model_variants"), str(ROOT / "external/ol-ems-hkust")]
import numpy as np
import pandas as pd
from _m0_variant_runner import build_parser, _runtime_config, _dataset_identity
from _pv_only_dataset import PVOnlyDataset

DEFAULT_DATA = ROOT / "output/prepared/annual_18_2_12_20261006"


def initial_arguments(data, output):
    folder = Path(data).resolve() / "forecast/lookback_96"
    return ["--data_format", "pv_only", "--data_path", str(folder / "pv_timeseries.csv"),
            "--data_manifest", str(folder / "manifest.json"), "--lookback_steps", "96",
            "--prediction_horizon", "96", "--anchor_lookback_steps", "96",
            "--hidden_dim", "64", "--num_layers", "2", "--dropout", "0.2", "--fc_hidden_size", "64",
            "--learning_rate", "0.001", "--weight_decay", "0.000001", "--train_batch_size", "64",
            "--val_batch_size", "64", "--max_epochs", "50", "--patience", "8", "--min_delta", "0.01",
            "--num_workers", "0", "--device", "cpu", "--torch_num_threads", "1", "--seed", "42",
            "--log_every_batches", "100", "--mode", "train", "--save_dir", str(Path(output).resolve()),
            "--experiment_name", "initial_gru_L96_H96_seed42"]


def preflight(data=DEFAULT_DATA, output=ROOT / "experiments/annual_18_2_12"):
    data = Path(data).resolve()
    args = build_parser("GRU").parse_args(initial_arguments(data, output))
    identity = _dataset_identity(args)  # validates every frozen hash, not just the CSV
    config = _runtime_config(args)
    dataset = PVOnlyDataset(config)
    expected = {"train": ("2021-07-01", "2023-01-01"), "val": ("2023-01-01", "2023-03-01"),
                "test": ("2023-03-01", "2024-03-01")}
    for name, dates in expected.items():
        if tuple(map(pd.Timestamp, config.split_dates()[name])) != tuple(map(pd.Timestamp, dates)):
            raise ValueError(f"Unexpected agreed split: {name}")
    frame = dataset.source_frame
    splits = {}
    for name, view in dataset.split_views().items():
        anchors = view._anchors
        first = frame.datetime.iloc[anchors[0]]
        last = frame.datetime.iloc[anchors[-1] + 95]
        if last >= pd.Timestamp(config.split_windows[name].end):
            raise AssertionError("Target leakage across split boundary")
        splits[name] = dict(samples=len(view), first_target=str(first), last_target=str(last))
    test = frame.loc[frame.datetime.ge("2023-03-01") & frame.datetime.lt("2024-03-01")].copy()
    if len(test) != 366 * 96:  # Includes leap day 2024-02-29.
        raise AssertionError("Incomplete leap-year test axis")
    test["month"] = test.datetime.dt.strftime("%Y-%m")
    quality = test.groupby("month").agg(rows=("pv", "size"), valid_rows=("pv_target_valid", "sum"))
    quality["invalid_rows"] = quality.rows - quality.valid_rows
    for filename in ("dispatch.csv", "load_tiers.csv"):
        source = pd.read_csv(data / "dispatch" / filename, parse_dates=["datetime"])
        index = pd.DatetimeIndex(source.datetime)
        if index.has_duplicates or not index.equals(pd.date_range("2021-07-01", "2024-03-02", freq="15min", inclusive="left")):
            raise ValueError(f"Incomplete {filename}")
        if not np.isfinite(source.select_dtypes(include="number")).all().all():
            raise ValueError(f"Nonfinite {filename}")
    preparation = json.loads((data / "preparation.json").read_text())
    command = [sys.executable, "-u", str(ROOT / "model_variants/gru_2layer_m0.py"), *initial_arguments(data, output)]
    result = dict(status="forecast_inputs_ready_simulation_deferred", model_or_optimizer_run=False,
        dataset_identity=identity, splits=splits, split_dates=config.split_dates(),
        scaler={"min": dataset.pv_min, "max": dataset.pv_max, "fit_end_exclusive": "2023-01-01"},
        test_steps=len(test), monthly_quality=quality.reset_index().to_dict("records"),
        extension_invalid_daylight_quarters=preparation["extension_invalid_daylight_quarters"],
        confirmed={"daylight_missing_policy": "no imputation; valid forecasting windows only",
                   "monthly_update": "warm-start, expanding history, last 14 days validation",
                   "model_scope": "M0 first; M1/M2 later", "continuous_simulation": "deferred by user"},
        pending_for_forecast=[],
        pending_for_future_simulation=["missing_data_policy", "annual_event_protocol", "rho_definition"],
        initial_train_command=command, initial_train_command_shell=shlex.join(command),
        historical_checkpoints_reusable=False,
        note="Initial command is not a monthly-updating annual experiment. Do not reuse old 2023-trained weights.")
    (data / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "splits", "scaler", "test_steps", "monthly_quality", "pending_for_forecast")},
                     ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--experiment-output", type=Path, default=ROOT / "experiments/annual_18_2_12")
    args = parser.parse_args()
    preflight(args.data, args.experiment_output)

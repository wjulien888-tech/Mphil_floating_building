"""Forecast, checkpoint and dropout-resume contracts for the two M0 copies."""
from argparse import Namespace
from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "ol-ems-hkust"))
sys.path.insert(0, str(ROOT / "model_variants"))

import _m0_variant_runner as runner
from gru_2layer_m0 import PVGRUController
from lstm_2layer_m0 import PVLSTMController
from _m0_variant_dataset import CSVDataset
from configurations.hkust_15min import DEFAULT_CONFIG as ORIGINAL_CONFIG
from src.training_and_evaluation.m0_training import load_checkpoint, seed_run, train_m0
from test_m0_dataset import make_data


VARIANTS = [
    pytest.param(PVLSTMController, "_lstm", 51787, id="lstm"),
    pytest.param(PVGRUController, "_gru", 38923, id="gru"),
]


def make_model(controller_class, horizon=1):
    return controller_class(
        optimisation_layer=runner._ForecastOnlyLayer(horizon),
        input_dim=5,
        pv_min=0.0,
        pv_max=20.0,
    )


@pytest.mark.parametrize("controller_class,recurrent_name,parameter_count", VARIANTS)
def test_full_history_forecast_gradients_and_weight_roundtrip(
    tmp_path, controller_class, recurrent_name, parameter_count
):
    seed_run(42, 1)
    model = make_model(controller_class)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == parameter_count
    recurrent = getattr(model, recurrent_name)
    assert recurrent.num_layers == 2
    assert recurrent.hidden_size == 64
    assert recurrent.dropout == 0.2
    assert model.fc1.out_features == 5
    assert model.fc2.out_features == 1

    # Keep the small ReLU head active so this exercises both recurrent layers.
    with torch.no_grad():
        model.fc1.bias.fill_(1.0)
    history = torch.rand(2, 96, 5, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {name: value.detach().clone() for name, value in recurrent.named_parameters()}
    model.train()
    prediction = model.forecast_pv(history)
    assert prediction.shape == (2, 1, 1)
    assert torch.isfinite(prediction).all()
    assert ((prediction >= 0.0) & (prediction <= 20.0)).all()
    prediction.square().mean().backward()
    for layer in (0, 1):
        for prefix in ("weight_ih", "weight_hh"):
            name = f"{prefix}_l{layer}"
            gradient = getattr(recurrent, name).grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0
    optimizer.step()
    for layer in (0, 1):
        name = f"weight_ih_l{layer}"
        assert not torch.equal(before[name], getattr(recurrent, name))

    model.eval()
    with torch.no_grad():
        expected = model.forecast_pv(history)
        assert torch.equal(expected, model.forecast_pv(history))
    checkpoint_path = tmp_path / "roundtrip.pt"
    torch.save({"controller_state_dict": model.state_dict()}, checkpoint_path)
    restored = make_model(controller_class)
    restored.load_state_dict(load_checkpoint(checkpoint_path)["controller_state_dict"], strict=True)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(expected, restored.forecast_pv(history))


@pytest.mark.parametrize("controller_class,recurrent_name,parameter_count", VARIANTS)
def test_runner_loads_matching_model_and_rejects_other_type_or_dropout(
    tmp_path, controller_class, recurrent_name, parameter_count
):
    _, config = make_data(tmp_path)
    config = replace(config, lookback_steps=96, prediction_horizon=1)
    dataset = CSVDataset(config=config)
    settings = runner.build_parser(controller_class.recurrent_type).parse_args([])
    original = runner._build_controller(config, dataset, settings, controller_class)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "controller_state_dict": original.state_dict(),
            "dataset_scaler": dataset.state_dict(),
            "config": asdict(config),
            "model_architecture": runner._model_architecture(config, dataset, settings),
        },
        checkpoint_path,
    )
    settings.load_model = checkpoint_path
    restored = runner._build_controller(
        config, CSVDataset(config=config), settings, controller_class
    )
    assert original.state_dict().keys() == restored.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])

    other_class = PVGRUController if controller_class is PVLSTMController else PVLSTMController
    wrong_type = runner.build_parser(other_class.recurrent_type).parse_args(
        ["--load_model", str(checkpoint_path)]
    )
    with pytest.raises(ValueError, match="architecture"):
        runner._build_controller(config, CSVDataset(config=config), wrong_type, other_class)
    settings.dropout = 0.3
    with pytest.raises(ValueError, match="architecture"):
        runner._build_controller(config, CSVDataset(config=config), settings, controller_class)


class TinyPV(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, index):
        history = torch.arange(96 * 5, dtype=torch.float32).reshape(96, 5) / 480
        return {"history": history + index / 10, "pv_target": torch.full((1, 1), float(index))}


def training_args(**overrides):
    values = dict(
        max_epochs=2, patience=8, min_delta=0.01, learning_rate=1e-3,
        weight_decay=1e-6, train_batch_size=3, val_batch_size=4,
        num_workers=0, device="cpu", seed=42, torch_num_threads=1,
        limit_train_batches=1.0, limit_val_batches=1.0, fast_dev_run=0,
        resume=None,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("controller_class", [PVLSTMController, PVGRUController])
def test_dropout_resume_matches_uninterrupted_training(tmp_path, controller_class):
    metadata = {
        "dataset_identity": {"fixture": "tiny"},
        "code_sha256": {"fixture": "unchanged"},
        "config": {"lookback_steps": 96, "prediction_horizon": 1},
        "model_architecture": {"type": controller_class.recurrent_type, "dropout": 0.2},
        "dataset_scaler": {},
    }

    def run(directory, settings):
        directory.mkdir(exist_ok=True)
        seed_run(settings.seed, settings.torch_num_threads)
        model = make_model(controller_class)
        train_m0(model, TinyPV(), TinyPV(), directory, settings, metadata, {})
        return load_checkpoint(directory / "last.pt")

    full = run(tmp_path / "full", training_args())
    resume_dir = tmp_path / "resumed"
    run(resume_dir, training_args(max_epochs=1))
    resumed = run(resume_dir, training_args(resume=resume_dir / "last.pt"))
    assert full["epoch"] == resumed["epoch"] == 2
    for name, value in full["controller_state_dict"].items():
        assert torch.equal(value, resumed["controller_state_dict"][name])
    assert full["best_epoch"] == resumed["best_epoch"]
    assert full["best_val_loss_mse"] == resumed["best_val_loss_mse"]
    for expected, actual in zip(full["metrics_history"], resumed["metrics_history"]):
        assert expected["train_loss_mse"] == actual["train_loss_mse"]
        assert expected["val_loss_mse"] == actual["val_loss_mse"]


@pytest.mark.parametrize("recurrent_type", ["LSTM", "GRU"])
def test_runner_defaults_to_96_observations_predicting_next_quarter_hour(recurrent_type):
    args = runner.build_parser(recurrent_type).parse_args([])
    config = runner._runtime_config(args)
    assert config.lookback_steps == 96
    assert config.prediction_horizon == 1
    assert pd.Timedelta(config.frequency) == pd.Timedelta(minutes=15)
    assert config.dt_hours == 0.25
    assert config.history_features == ("pv",)
    assert args.data_format == "pv_only"
    # The original model's frozen task contract is not modified by these copies.
    assert ORIGINAL_CONFIG.lookback_steps == 672
    assert ORIGINAL_CONFIG.prediction_horizon == 96


@pytest.mark.parametrize("source", ["synthetic", "frozen"])
def test_single_step_windows_use_exactly_the_preceding_96_rows(tmp_path, source):
    if source == "synthetic":
        frame, config = make_data(tmp_path)
        config = replace(config, lookback_steps=96, prediction_horizon=1)
        # Distinct values make off-by-one and previous-day mistakes observable.
        frame["pv_floating_disturbed_raw_kW"] = np.arange(len(frame)) / 50
        frame.to_csv(config.data_path, index=False)
    else:
        config = runner._runtime_config(
            runner.build_parser("LSTM").parse_args(["--data_format", "legacy_dispatch"])
        )
        if not config.data_path.is_file():
            pytest.skip("Local frozen snapshot is unavailable")

    dataset = CSVDataset(config=config)
    canonical = dataset.source_frame
    for name, view in dataset.split_views().items():
        split = config.split_windows[name]
        times = pd.DatetimeIndex(canonical.datetime.iloc[view._anchors])
        assert (times >= pd.Timestamp(split.start)).all()
        assert (times < pd.Timestamp(split.end)).all()
        for index in sorted({0, len(view) // 2, len(view) - 1}):
            anchor = int(view._anchors[index])
            sample = view.forecast_view()[index]
            regular = view[index]
            raw_history = torch.tensor(
                canonical.loc[anchor - 96:anchor - 1, view.input_features].to_numpy(),
                dtype=torch.float32,
            )
            scale = view.scaling_factors_max - view.scaling_factors_min
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            expected_history = (raw_history - view.scaling_factors_min) / safe_scale
            assert sample["history"].shape == (96, 5)
            torch.testing.assert_close(sample["history"], expected_history)
            torch.testing.assert_close(sample["history"], regular["history"])
            assert sample["pv_target"].shape == (1, 1)
            assert sample["pv_target"].item() == pytest.approx(canonical.pv.iloc[anchor])
            torch.testing.assert_close(sample["pv_target"], regular["pv_target"])
            assert sample["future_timestamps"].shape == (1,)
            assert sample["future_timestamps"].item() == sample["timestamp"].item()
            target_time = pd.Timestamp(sample["timestamp"].item(), unit="s")
            assert target_time == canonical.datetime.iloc[anchor - 1] + pd.Timedelta(minutes=15)
            assert sample["persistence"].shape == (1, 1)
            assert sample["persistence"].item() == pytest.approx(canonical.pv.iloc[anchor - 96])


def test_single_step_quality_filters_only_its_target_and_all_history_rows(tmp_path):
    frame, config = make_data(tmp_path)
    config = replace(config, lookback_steps=96, prediction_horizon=1)
    frame.loc[200, "history_valid"] = 0
    frame.loc[350, "pv_target_valid"] = 0
    frame.to_csv(config.data_path, index=False)
    dataset = CSVDataset(config=config)
    for name, view in dataset.split_views().items():
        split = config.split_windows[name]
        expected = [
            anchor for anchor in range(96, len(frame))
            if pd.Timestamp(split.start) <= frame.datetime.iloc[anchor] < pd.Timestamp(split.end)
            and frame.history_valid.iloc[anchor - 96:anchor].all()
            and frame.pv_target_valid.iloc[anchor]
        ]
        assert view._anchors.tolist() == expected
    # An invalid label at t does not discard a previous sample whose only target is t-1.
    assert 349 in dataset._anchors
    assert 350 not in dataset._anchors
    assert 351 in dataset._anchors


def test_daily_persistence_is_lag_96_and_unchanged_by_future_values(tmp_path):
    from _m0_variant_evaluation import persistence_from_history

    frame, config = make_data(tmp_path)
    config = replace(config, lookback_steps=96, prediction_horizon=1)
    frame["pv_floating_disturbed_raw_kW"] = np.arange(len(frame)) / 50
    frame.to_csv(config.data_path, index=False)
    dataset = CSVDataset(config=config).for_split("val")
    anchor = int(dataset._anchors[0])
    sample = dataset.forecast_view()[0]
    baseline = persistence_from_history(sample["history"].unsqueeze(0), dataset)
    assert baseline.shape == (1, 1, 1)
    assert baseline.item() == pytest.approx(frame.pv_floating_disturbed_raw_kW.iloc[anchor - 96])
    assert baseline.item() != pytest.approx(frame.pv_floating_disturbed_raw_kW.iloc[anchor - 1])
    torch.testing.assert_close(baseline.squeeze(0), sample["persistence"])
    with pytest.raises(ValueError, match="96"):
        persistence_from_history(sample["history"][-95:].unsqueeze(0), dataset)

    frame.loc[anchor:, "pv_floating_disturbed_raw_kW"] *= 3
    frame.to_csv(config.data_path, index=False)
    changed = CSVDataset(config=config).for_split("val").forecast_view()[0]
    torch.testing.assert_close(sample["history"], changed["history"])
    torch.testing.assert_close(sample["persistence"], changed["persistence"])
    assert not torch.equal(sample["pv_target"], changed["pv_target"])


def test_evaluation_reports_distinct_daily_and_last_observation_baselines(tmp_path):
    from _m0_variant_evaluation import evaluate_m0

    frame, config = make_data(tmp_path)
    config = replace(config, lookback_steps=96, prediction_horizon=1)
    frame["pv_floating_disturbed_raw_kW"] = np.arange(len(frame)) / 50
    frame.to_csv(config.data_path, index=False)
    dataset = CSVDataset(config=config).for_split("val")
    args = runner.build_parser("LSTM").parse_args([])
    args.val_batch_size = 3
    args.eval_limit_batches = 1e-9  # One batch is sufficient to check exported alignment.
    summary = evaluate_m0(make_model(PVLSTMController), dataset, tmp_path, args, {})
    predictions = pd.read_csv(tmp_path / "predictions.csv.gz")
    anchors = dataset._anchors[:3]
    np.testing.assert_allclose(predictions.pv_actual_kw, frame.pv_floating_disturbed_raw_kW.iloc[anchors])
    np.testing.assert_allclose(predictions.persistence_24h_kw, frame.pv_floating_disturbed_raw_kW.iloc[anchors - 96])
    np.testing.assert_allclose(predictions.persistence_last_kw, frame.pv_floating_disturbed_raw_kW.iloc[anchors - 1])
    assert set(predictions.horizon_step) == {1}
    metrics = {(row["method"], row["period"]): row for row in summary["metrics"]}
    assert metrics["persistence_24h", "all"]["rmse_kw"] == pytest.approx(96 / 50)
    assert metrics["persistence_last", "all"]["rmse_kw"] == pytest.approx(1 / 50, abs=1e-6)


def test_variant_r2_pools_targets_across_horizons_and_batches():
    import numpy as np
    from _m0_variant_evaluation import ErrorAccumulator

    accumulator = ErrorAccumulator(2)
    # Flattened targets [0,1,2,3] have SST=5; forecasts have SSE=2.
    accumulator.update(np.array([[0., 1.]]), np.array([[0., 1.]]),
                       np.ones((1, 2), dtype=bool))
    accumulator.update(np.array([[1., 2.]]), np.array([[2., 3.]]),
                       np.ones((1, 2), dtype=bool))
    assert accumulator.metrics()["r2"] == pytest.approx(0.6)
    assert accumulator.metrics(0)["r2"] == pytest.approx(0.5)
    assert accumulator.metrics(1)["r2"] == pytest.approx(0.5)


def test_variant_r2_is_undefined_for_constant_or_empty_targets():
    import numpy as np
    from _m0_variant_evaluation import ErrorAccumulator

    accumulator = ErrorAccumulator(2)
    assert accumulator.metrics()["r2"] is None
    accumulator.update(np.array([[1., 2.]]), np.zeros((1, 2)),
                       np.ones((1, 2), dtype=bool))
    assert accumulator.metrics()["r2"] is None

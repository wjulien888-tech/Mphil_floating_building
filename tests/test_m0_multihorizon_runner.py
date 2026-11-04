"""Configuration, checkpoint and forward-only checks; no training is executed."""
from dataclasses import asdict, replace
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/ol-ems-hkust"))
sys.path.insert(0, str(ROOT / "model_variants"))

import _m0_variant_runner as runner
from _m0_variant_evaluation import last_value_from_history, persistence_from_history
from _pv_only_dataset import DEFAULT_CONFIG
from gru_2layer_m0 import PVGRUController
from lstm_2layer_m0 import PVLSTMController
from src.training_and_evaluation.m0_training import _epoch, sha256_file


@pytest.fixture(autouse=True)
def forbid_optimizer_steps(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Preparation checks must not perform optimizer steps")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    torch.set_num_threads(1)


def arguments(kind="LSTM", lookback=96, horizon=96):
    return runner.build_parser(kind).parse_args([
        "--lookback_steps", str(lookback), "--prediction_horizon", str(horizon),
        "--anchor_lookback_steps", "288", "--fc_hidden_size", "64",
        "--log_every_batches", "100",
    ])


def fake_dataset(horizon=96):
    return SimpleNamespace(
        input_dim=1, input_features=["pv"], prediction_horizon=horizon,
        pv_min=0.0, pv_max=20.0, load_state_dict=lambda _state: None,
    )


@pytest.mark.parametrize("model_class", [PVLSTMController, PVGRUController])
@pytest.mark.parametrize("lookback", [96, 192, 288])
def test_six_configurations_emit_full_day_without_gradient_updates(model_class, lookback):
    args = arguments(model_class.recurrent_type, lookback)
    config = runner._runtime_config(args)
    assert (config.lookback_steps, config.prediction_horizon, config.anchor_lookback_steps) == (lookback, 96, 288)
    dataset = fake_dataset()
    model = runner._build_controller(config, dataset, args, model_class).eval()
    weights_before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.inference_mode():
        prediction = model.forecast_pv(torch.rand(2, lookback, 1))
    assert prediction.shape == (2, 96, 1)
    assert torch.isfinite(prediction).all()
    assert model.fc1.in_features == model.fc1.out_features == 64
    assert model.fc2.in_features == 64 and model.fc2.out_features == 96
    recurrent = getattr(model, "_" + model_class.recurrent_type.lower())
    assert recurrent.num_layers == 2 and recurrent.hidden_size == 64
    assert recurrent.dropout == 0.2
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in weights_before.items())
    assert f"_{lookback}to96_mlp_v3" in runner._model_architecture(config, dataset, args)["architecture_id"]


def test_old_checkpoint_without_anchor_field_is_accepted_only_for_old_anchor(tmp_path):
    args = runner.build_parser("LSTM").parse_args([])
    config = runner._runtime_config(args)
    dataset = fake_dataset(horizon=1)
    model = runner._build_controller(config, dataset, args, PVLSTMController)
    saved_config = asdict(config)
    saved_config.pop("anchor_lookback_steps")
    checkpoint = tmp_path / "old.pt"
    torch.save({
        "controller_state_dict": model.state_dict(), "dataset_scaler": {},
        "config": saved_config,
        "model_architecture": runner._model_architecture(config, dataset, args),
    }, checkpoint)
    args.load_model = checkpoint
    runner._build_controller(config, dataset, args, PVLSTMController)
    assert runner._model_architecture(config, dataset, args)["architecture_id"] == "hkust_m0_lstm_96to1_mlp_v2"
    with pytest.raises(ValueError, match="anchor_lookback_steps"):
        runner._build_controller(replace(config, anchor_lookback_steps=288), dataset, args, PVLSTMController)
    with pytest.raises(ValueError, match="prediction_horizon"):
        runner._build_controller(replace(config, prediction_horizon=96), fake_dataset(), args, PVLSTMController)


def write_manifest(tmp_path, args, schema_version=3):
    data = tmp_path / "pv_timeseries.csv"
    data.write_text("datetime,pv_kW\n2023-01-01,1\n")
    parameters = {"lookback_steps": args.lookback_steps,
                  "prediction_horizon": args.prediction_horizon,
                  "history_features": ["pv"], "target_feature": "pv"}
    if schema_version == 3:
        parameters["anchor_lookback_steps"] = args.anchor_lookback_steps
    manifest = {
        "schema_version": schema_version,
        "artifact_type": "pv_only_forecast_multistep" if schema_version == 3 else "pv_only_forecast_96to1",
        "parameters": parameters,
        "splits": {name: {"start_inclusive": window.start, "end_exclusive": window.end}
                   for name, window in DEFAULT_CONFIG.split_windows.items()},
        "outputs": [{"path": data.name, "size_bytes": data.stat().st_size, "sha256": sha256_file(data)}],
    }
    args.data_path = data
    args.data_manifest = tmp_path / "manifest.json"
    args.data_manifest.write_text(json.dumps(manifest))
    return manifest


def test_multistep_manifest_binds_horizon_window_common_anchor_and_hash(tmp_path):
    args = arguments(lookback=192)
    write_manifest(tmp_path, args)
    identity = runner._dataset_identity(args)
    assert identity["task"]["anchor_lookback_steps"] == 288
    for field, wrong in (("lookback_steps", 96), ("prediction_horizon", 1), ("anchor_lookback_steps", 96)):
        previous = getattr(args, field)
        setattr(args, field, wrong)
        with pytest.raises(ValueError, match="task/features"):
            runner._dataset_identity(args)
        setattr(args, field, previous)
    args.data_path.write_text("tampered")
    with pytest.raises(ValueError, match="differs from manifest"):
        runner._dataset_identity(args)


def test_single_step_manifest_identity_remains_backward_compatible(tmp_path):
    args = runner.build_parser("LSTM").parse_args([])
    write_manifest(tmp_path, args, schema_version=2)
    identity = runner._dataset_identity(args)
    assert "anchor_lookback_steps" not in identity["task"]
    args.prediction_horizon = 96
    with pytest.raises(ValueError, match="96-to-1"):
        runner._dataset_identity(args)


def test_public_cli_forwards_window_and_logging_options_without_running_training(monkeypatch):
    spec = importlib.util.spec_from_file_location("multihorizon_m0_entry", ROOT / "external/ol-ems-hkust/main.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    captured = {}
    def capture(args, model_class):
        captured.update(args=args, model_class=model_class)
        return Path("not-executed")
    monkeypatch.setattr(entry._variant_runner, "main", capture)
    assert entry.cli([
        "--model", "gru", "--lookback_steps", "288", "--prediction_horizon", "96",
        "--anchor_lookback_steps", "288", "--fc_hidden_size", "64", "--log_every_batches", "100",
    ]) == Path("not-executed")
    args = captured["args"]
    assert (args.lookback_steps, args.prediction_horizon, args.anchor_lookback_steps) == (288, 96, 288)
    assert args.fc_hidden_size == 64 and args.log_every_batches == 100
    assert captured["model_class"] is PVGRUController


def test_full_horizon_persistence_uses_only_pre_origin_observations():
    dataset = SimpleNamespace(prediction_horizon=96, input_features=["pv"], denormalise_pv=lambda value: value)
    history = torch.arange(288, dtype=torch.float32).reshape(1, 288, 1)
    daily = persistence_from_history(history, dataset)
    last = last_value_from_history(history, dataset)
    torch.testing.assert_close(daily, history[:, -96:, :])
    assert last.shape == (1, 96, 1) and (last == 287).all()


def test_batch_progress_logs_cumulative_validation_loss_without_optimizer(caplog):
    class FixedForecast(torch.nn.Module):
        def forecast_pv(self, history):
            assert not torch.is_grad_enabled()
            return torch.zeros((history.shape[0], 96, 1))
    loader = DataLoader([{"history": torch.zeros((96, 1)), "pv_target": torch.ones((96, 1))}
                         for _ in range(7)], batch_size=2)
    with caplog.at_level(logging.INFO, logger="hkust_ol_ems"):
        mse, samples = _epoch(FixedForecast(), loader, torch.device("cpu"), len(loader), None,
                              log_every_batches=2, progress_label="Epoch 1/50 val")
    assert mse == 1.0 and samples == 7
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 3
    assert "batch 1/4" in messages[0] and "batch 2/4" in messages[1] and "batch 4/4" in messages[2]
    assert all("Epoch 1/50 val" in message and "MSE 1.000000" in message for message in messages)

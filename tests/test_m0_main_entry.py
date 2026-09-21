"""The public M0 entry selects the frozen 96-to-1 PV-only model explicitly."""
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/ol-ems-hkust"))
sys.path.insert(0, str(ROOT / "model_variants"))

from _pv_only_dataset import DEFAULT_CONFIG, PVOnlyDataset
from configurations.hkust_15min import SplitWindow
from gru_2layer_m0 import PVGRUController
from lstm_2layer_m0 import PVLSTMController


@pytest.fixture(scope="module")
def entry():
    spec = importlib.util.spec_from_file_location(
        "m0_main_entry_under_test", ROOT / "external/ol-ems-hkust/main.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pv_dataset(tmp_path):
    """Small, complete chronological fixture; no local frozen data needed."""
    times = pd.date_range("2023-01-01", periods=6 * 96, freq="15min")
    slot = np.arange(len(times)) % 96
    frame = pd.DataFrame({
        "datetime": times,
        "pv_kW": np.maximum(np.sin((slot - 24) * np.pi / 48), 0) * 20,
        "history_valid": 1,
        "pv_target_valid": 1,
        "split": np.repeat(["train", "val", "test"], [3 * 96, 96, 2 * 96]),
    })
    path = tmp_path / "pv_timeseries.csv"
    frame.to_csv(path, index=False)
    config = replace(DEFAULT_CONFIG, data_path=path, split_windows={
        "train": SplitWindow("2023-01-01", "2023-01-04"),
        "val": SplitWindow("2023-01-04", "2023-01-05"),
        "test": SplitWindow("2023-01-05", "2023-01-07"),
    })
    return PVOnlyDataset(config)


def capture_dispatch(entry, monkeypatch, argv):
    captured = {}

    def runner_main(args, controller_class):
        captured.update(args=args, controller_class=controller_class)
        return Path("captured-experiment")

    with monkeypatch.context() as patch:
        patch.setattr(entry._variant_runner, "main", runner_main)
        result = entry.main(entry.build_parser().parse_args(argv))
    assert result == Path("captured-experiment")
    return captured["args"], captured["controller_class"]


def test_default_parser_uses_two_layer_univariate_single_step_task(entry):
    args = entry.build_parser().parse_args([])
    assert args.model_type == "lstm"
    assert args.mode == "train"
    assert args.data_format == "pv_only"
    assert args.hidden_dim == 64
    assert args.num_layers == 2
    assert args.dropout == 0.2
    assert args.fc_hidden_size == 5
    config = entry._variant_runner._runtime_config(args)
    assert config.history_features == ("pv",)
    assert (config.lookback_steps, config.prediction_horizon) == (96, 1)
    assert pd.Timedelta(config.frequency) == pd.Timedelta(minutes=15)


@pytest.mark.parametrize("flag", ["--model_type", "--model"])
@pytest.mark.parametrize("name,controller_class", [
    ("lstm", PVLSTMController), ("gru", PVGRUController),
])
def test_main_dispatches_requested_model(entry, monkeypatch, flag, name, controller_class):
    args, selected = capture_dispatch(entry, monkeypatch, [flag, name])
    assert selected is controller_class
    assert args.recurrent_type == name.upper()


def test_default_output_directories_separate_models_and_preserve_override(entry, monkeypatch, tmp_path):
    lstm_args, _ = capture_dispatch(entry, monkeypatch, ["--model", "lstm"])
    gru_args, _ = capture_dispatch(entry, monkeypatch, ["--model", "gru"])
    assert Path(lstm_args.save_dir) != Path(gru_args.save_dir)
    assert "lstm" in Path(lstm_args.save_dir).name.lower()
    assert "gru" in Path(gru_args.save_dir).name.lower()
    custom = tmp_path / "explicit-results"
    args, _ = capture_dispatch(entry, monkeypatch, ["--model", "gru", "--save_dir", str(custom)])
    assert Path(args.save_dir) == custom


@pytest.mark.parametrize("name", ["lstm", "gru"])
def test_entry_selected_model_has_exact_pv_input_output_contract(
    entry, monkeypatch, pv_dataset, name,
):
    args, controller_class = capture_dispatch(entry, monkeypatch, ["--model", name])
    model = entry._variant_runner._build_controller(
        pv_dataset.config, pv_dataset, args, controller_class,
    )
    history = torch.stack([pv_dataset[0]["history"], pv_dataset[1]["history"]])
    assert history.shape == (2, 96, 1)
    recurrent = getattr(model, "_" + name)
    assert recurrent.input_size == 1
    assert recurrent.num_layers == 2
    assert recurrent.hidden_size == 64
    assert recurrent.dropout == 0.2
    model.eval()
    with torch.no_grad():
        prediction = model.forecast_pv(history)
    assert prediction.shape == (2, 1, 1)
    assert torch.isfinite(prediction).all()
    assert ((prediction >= pv_dataset.pv_min) & (prediction <= pv_dataset.pv_max)).all()


@pytest.mark.parametrize("name,wrong_name", [("lstm", "gru"), ("gru", "lstm")])
def test_entry_models_restore_matching_weights_and_reject_wrong_model_checkpoint(
    entry, monkeypatch, pv_dataset, tmp_path, name, wrong_name,
):
    args, controller_class = capture_dispatch(entry, monkeypatch, ["--model", name])
    runner = entry._variant_runner
    original = runner._build_controller(pv_dataset.config, pv_dataset, args, controller_class)
    checkpoint = tmp_path / "best.pt"
    torch.save({
        "controller_state_dict": original.state_dict(),
        "dataset_scaler": pv_dataset.state_dict(),
        "config": asdict(pv_dataset.config),
        "model_architecture": runner._model_architecture(pv_dataset.config, pv_dataset, args),
    }, checkpoint)
    args.load_model = checkpoint
    restored = runner._build_controller(pv_dataset.config, pv_dataset, args, controller_class)
    assert original.state_dict().keys() == restored.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])

    wrong_args, wrong_class = capture_dispatch(entry, monkeypatch, [
        "--model", wrong_name, "--mode", "evaluate", "--load_model", str(checkpoint),
    ])
    with pytest.raises(ValueError, match="architecture"):
        runner._build_controller(pv_dataset.config, pv_dataset, wrong_args, wrong_class)


def test_closed_loop_controller_rejects_single_step_forecast_checkpoint(entry, tmp_path):
    checkpoint = tmp_path / "forecast_only.pt"
    torch.save({
        "controller_state_dict": {},
        "dataset_scaler": {},
        "config": asdict(DEFAULT_CONFIG),
        "model_architecture": {"recurrent_type": "LSTM", "prediction_horizon": 1},
    }, checkpoint)
    args = entry.build_legacy_parser().parse_args([
        "--mode", "closed-loop", "--load_model", str(checkpoint),
    ])
    # This must fail before creating an optimisation layer or mutating scaler state.
    dataset = SimpleNamespace(input_dim=5)
    with pytest.raises(ValueError, match="(?i)(forecast|single|96|configuration|reproducibility)"):
        entry._build_controller(entry.DEFAULT_CONFIG, dataset, args)


@pytest.mark.parametrize("mode_argument", [["--mode", "closed-loop"], ["--mode=closed-loop"]])
def test_cli_routes_closed_loop_to_legacy_parser_and_executor(entry, monkeypatch, mode_argument):
    captured = {}

    def legacy_main(args):
        captured["args"] = args
        return Path("legacy-experiment")

    def unexpected_forecast(*_args, **_kwargs):
        pytest.fail("closed-loop must not run the single-step forecast runner")

    monkeypatch.setattr(entry, "_legacy_main", legacy_main)
    monkeypatch.setattr(entry._variant_runner, "main", unexpected_forecast)
    result = entry.cli(mode_argument + ["--allow_untrained"])
    assert result == Path("legacy-experiment")
    assert captured["args"].mode == "closed-loop"
    assert captured["args"].allow_untrained is True


def test_cli_routes_forecast_evaluation_to_requested_model(entry, monkeypatch, tmp_path):
    captured = {}

    def variant_main(args, controller_class):
        captured.update(args=args, controller_class=controller_class)
        return Path("forecast-evaluation")

    monkeypatch.setattr(entry._variant_runner, "main", variant_main)
    checkpoint = tmp_path / "best.pt"
    result = entry.cli([
        "--mode", "evaluate", "--model", "gru", "--load_model", str(checkpoint),
    ])
    assert result == Path("forecast-evaluation")
    assert captured["controller_class"] is PVGRUController
    assert captured["args"].mode == "evaluate"
    assert captured["args"].load_model == checkpoint
    assert captured["args"].recurrent_type == "GRU"


def test_cli_rejects_single_step_checkpoint_before_legacy_execution(entry, monkeypatch, tmp_path):
    checkpoint = tmp_path / "single_step.pt"
    torch.save({"controller_state_dict": {}, "config": asdict(DEFAULT_CONFIG)}, checkpoint)

    def unexpected_legacy(*_args, **_kwargs):
        pytest.fail("single-step checkpoints must be rejected before legacy MPC execution")

    monkeypatch.setattr(entry, "_legacy_main", unexpected_legacy)
    with pytest.raises(ValueError, match="(?i)(96|single|forecast|horizon|15-minute)"):
        entry.cli(["--mode", "closed-loop", "--load_model", str(checkpoint)])

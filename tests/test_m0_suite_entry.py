"""Formal M0 tools route both models to the same 96-to-1 experiment contract."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_formal_m0, run_pvonly_models, summarize_formal_m0


def options(command):
    return dict(zip(command[2::2], command[3::2]))


def test_suite_commands_use_official_entry_and_independent_best_weights(tmp_path):
    destination = tmp_path / "suite"
    runs = run_pvonly_models.build_runs(destination)
    assert not destination.exists()
    assert {run["model"] for run in runs} == {"lstm", "gru"}
    for run in runs:
        commands = [run["train_command"], *run["evaluation_commands"].values()]
        for command in commands:
            assert Path(command[1]) == ROOT / "external/ol-ems-hkust/main.py"
            args = options(command)
            assert args["--model_type"] == run["model"]
            assert args["--seed"] == "42"
            assert args["--data_format"] == "pv_only"
            assert args["--num_layers"] == "2" and args["--hidden_dim"] == "64"
            assert args["--dropout"] == "0.2" and args["--fc_hidden_size"] == "5"
            assert Path(args["--data_path"]).name == "pv_timeseries.csv"
            assert Path(args["--data_manifest"]).parent == Path(args["--data_path"]).parent
            assert "--fast_dev_run" not in args and "--resume" not in args
        train = options(run["train_command"])
        assert train["--mode"] == "train" and "--load_model" not in train
        assert set(run["evaluation_commands"]) == {"val", "test"}
        for split, command in run["evaluation_commands"].items():
            args = options(command)
            assert args["--mode"] == "evaluate" and args["--eval_split"] == split
            assert Path(args["--load_model"]) == destination / train["--experiment_name"] / "best.pt"
            assert args["--experiment_name"] != train["--experiment_name"]


def test_formal_entry_delegates_without_old_single_layer_protocol(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_formal_m0, "run_suite", calls.append)
    destination = tmp_path / "explicit_suite"
    run_formal_m0.main(["--output", str(destination)])
    assert calls == [destination]
    assert not destination.exists()


def test_summary_defaults_to_pvonly_and_legacy_is_explicit(tmp_path, monkeypatch):
    current, legacy = [], []
    monkeypatch.setattr(summarize_formal_m0.subprocess, "run", lambda *args, **kwargs: current.append((args, kwargs)))
    monkeypatch.setattr(summarize_formal_m0, "summarize_legacy", legacy.append)
    summarize_formal_m0.main(["--suite", str(tmp_path)])
    assert not legacy
    command = current[0][0][0]
    assert Path(command[1]) == ROOT / "tools/summarize_pvonly_models.py"
    assert command[2:] == ["--suite", str(tmp_path)]
    assert current[0][1]["check"] is True
    summarize_formal_m0.main(["--suite", str(tmp_path), "--legacy"])
    assert legacy == [tmp_path] and len(current) == 1
    with pytest.raises(ValueError, match="only supported"):
        summarize_formal_m0.main(["--suite", str(tmp_path), "--legacy", "--allow-partial"])


@pytest.mark.parametrize("script", ["run_formal_m0.py", "run_pvonly_models.py", "summarize_formal_m0.py"])
def test_tool_help_has_no_data_or_training_side_effect(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools" / script), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir())

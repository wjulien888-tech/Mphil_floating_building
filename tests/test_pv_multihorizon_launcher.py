"""Orchestration checks use tiny print-only Python children, never ML training."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pv_multihorizon as suite


def options(command):
    return dict(zip(command[3::2], command[4::2]))


def test_six_commands_share_protocol_and_keep_windows_separate(tmp_path):
    output = tmp_path / "new_suite"
    runs = suite.build_runs(output, python="test-python")
    assert {(run["model"], run["lookback_steps"]) for run in runs} == {
        (model, lookback) for model in ("lstm", "gru") for lookback in (96, 192, 288)}
    assert len({run["name"] for run in runs}) == 6
    for run in runs:
        command = run["train_command"]
        assert command[:3] == ["test-python", "-u", str(ROOT / "external/ol-ems-hkust/main.py")]
        opts = options(command)
        assert opts["--lookback_steps"] == str(run["lookback_steps"])
        assert opts["--prediction_horizon"] == "96"
        assert opts["--anchor_lookback_steps"] == "288"
        assert opts["--fc_hidden_size"] == "64" and opts["--num_layers"] == "2"
        assert opts["--seed"] == "42" and opts["--log_every_batches"] == "100"
        assert opts["--torch_num_threads"] == "1"
        assert "--fast_dev_run" not in opts and "--load_model" not in opts
        assert Path(opts["--data_path"]).parent.name == f"lookback_{run['lookback_steps']}"
        for split in ("val", "test"):
            evaluated = options(run["evaluation_commands"][split])
            assert evaluated["--load_model"] == str(output / run["name"] / "best.pt")
            assert evaluated["--eval_split"] == split
    assert not output.exists()


def test_dry_run_creates_nothing_and_launches_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(suite, "validate_inputs", lambda runs: ({}, {}))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("dry run launched a child"))
    suite.main(["--dry-run", "--output", str(tmp_path / "absent")])
    assert "DRY RUN" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_print_only_children_stream_to_console_and_log(tmp_path, capsys):
    live = suite.LiveProcesses(tmp_path, tmp_path / "status.json")
    commands = [(name, [sys.executable, "-u", "-c", "print('example batch progress', flush=True)"])
                for name in ("lstm_demo", "gru_demo")]
    live.phase("print_only_test", commands, 2)
    console = capsys.readouterr().out
    for name, _ in commands:
        assert f"[{name}] example batch progress" in console
        assert "example batch progress" in (tmp_path / f"{name}_console.log").read_text()
    status = json.loads((tmp_path / "status.json").read_text())
    assert all(job["returncode"] == 0 for job in status["jobs"].values())
    assert not live.active


def test_failure_stops_other_print_only_child(tmp_path):
    live = suite.LiveProcesses(tmp_path, tmp_path / "status.json")
    commands = [
        ("failing_demo", [sys.executable, "-u", "-c", "import time; time.sleep(0.2); raise SystemExit(3)"]),
        ("waiting_demo", [sys.executable, "-u", "-c", "import time; print('waiting',flush=True); time.sleep(20)"]),
    ]
    with pytest.raises(RuntimeError, match="exited 3"):
        live.phase("print_only_test", commands, 2)
    assert live.stopping.is_set()
    assert not live.active
    jobs = json.loads((tmp_path / "status.json").read_text())["jobs"]
    assert jobs["failing_demo"]["returncode"] == 3
    assert jobs["waiting_demo"]["returncode"] != 0


def test_training_does_not_accept_automatic_test_selection():
    with pytest.raises(ValueError, match="validation only"):
        suite.main(["--eval-split", "test"])


def test_summary_counts_all_horizons_and_keeps_selected_results(tmp_path):
    runs = suite.build_runs(tmp_path, models=["lstm"], lookbacks=[96, 288])
    for run in runs:
        train_dir = tmp_path / run["name"]
        val_dir = tmp_path / f"{run['name']}_val"
        train_dir.mkdir()
        val_dir.mkdir()
        (train_dir / "best.pt").write_bytes(b"not-a-real-model")
        identity = {"lookback_steps": run["lookback_steps"]}
        (train_dir / "run_metadata.json").write_text(json.dumps({
            "status": "completed", "dataset_identity": identity, "best_epoch": 4, "epochs_completed": 6}))
        (val_dir / "evaluation_summary.json").write_text(json.dumps({
            "complete_split": True, "forecast_origins": 2, "horizon_steps": 96, "dataset_identity": identity,
            "checkpoint_sha256": suite.digest(train_dir / "best.pt"),
            "metrics": [{"method": "M0", "period": "all", "count": 192,
                         "rmse_kw": 2.0, "mae_kw": 1.0, "r2": 0.5}]}))
    suite.summarize(tmp_path, runs[:1], "val")
    output = suite.summarize(tmp_path, runs[1:], "val")
    lines = output.read_text().splitlines()
    assert len(lines) == 3

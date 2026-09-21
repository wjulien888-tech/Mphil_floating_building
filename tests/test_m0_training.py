"""Behavior tests for persisted M0 best/last and exact epoch-boundary resume."""
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

OL_ROOT = Path(__file__).resolve().parents[1] / "external/ol-ems-hkust"
sys.path.insert(0, str(OL_ROOT))
from src.training_and_evaluation.m0_training import load_checkpoint, seed_run, train_m0


class TinyDataset(Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, index):
        value = torch.tensor([[index / 10., 0.2], [0.1, index / 20.]])
        return {"history": value, "pv_target": torch.tensor([[index / 4.], [0.5], [0.7]])}


class TinyForecast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 3)

    def forecast_pv(self, history):
        return self.head(history.reshape(-1, 4)).unsqueeze(-1)

    def forward(self, *args):
        raise AssertionError("M0 training must not solve MPC")


def args(**overrides):
    result = dict(max_epochs=3, patience=8, min_delta=0.01, learning_rate=1e-3,
                  weight_decay=1e-6, train_batch_size=4, val_batch_size=4,
                  num_workers=0, device="cpu", seed=42, torch_num_threads=1,
                  limit_train_batches=1., limit_val_batches=1., fast_dev_run=0,
                  resume=None)
    result.update(overrides)
    return Namespace(**result)


def checkpoint_metadata():
    return {"dataset_identity": {"manifest_sha256": "frozen"}, "code_sha256": {"main": "same"},
            "config": {"horizon": 3}, "model_architecture": {"tiny": True},
            "dataset_scaler": {"feature": "pv"}}


def run(path, settings):
    path.mkdir(exist_ok=True)
    seed_run(settings.seed, settings.torch_num_threads)
    model = TinyForecast()
    train_m0(model, TinyDataset(), TinyDataset(), path, settings, checkpoint_metadata(), {})
    return load_checkpoint(path / "last.pt")


def test_resume_reproduces_uninterrupted_training(tmp_path):
    uninterrupted = run(tmp_path / "full", args())
    run(tmp_path / "resumed", args(max_epochs=1))
    resumed = run(tmp_path / "resumed", args(resume=tmp_path / "resumed/last.pt"))
    assert uninterrupted["epoch"] == resumed["epoch"] == 3
    for key, value in uninterrupted["controller_state_dict"].items():
        assert torch.equal(value, resumed["controller_state_dict"][key])
    assert uninterrupted["best_epoch"] == resumed["best_epoch"]
    assert uninterrupted["best_val_loss_mse"] == resumed["best_val_loss_mse"]
    for row_a, row_b in zip(uninterrupted["metrics_history"], resumed["metrics_history"]):
        assert row_a["val_loss_mse"] == row_b["val_loss_mse"]
        assert row_a["train_loss_mse"] == row_b["train_loss_mse"]
    assert (tmp_path / "resumed/metrics.csv").is_file()


def test_resume_rejects_changed_seed(tmp_path):
    run(tmp_path, args(max_epochs=1))
    with pytest.raises(ValueError, match="Resume settings/data/code/runtime differ: seed"):
        run(tmp_path, args(seed=43, resume=tmp_path / "last.pt"))


def test_true_best_saved_even_below_early_stop_delta(tmp_path, monkeypatch):
    import src.training_and_evaluation.m0_training as module
    losses = iter([10., 10., 9., 9.995, 8., 9.99])
    monkeypatch.setattr(module, "_epoch", lambda *a, **kw: (next(losses), 11))
    last = run(tmp_path, args(min_delta=0.01, patience=2))
    best = load_checkpoint(tmp_path / "best.pt")
    assert best["epoch"] == last["epoch"] == 3
    assert best["val_loss_mse"] == 9.99
    assert last["bad_epochs"] == 2
    assert best["best_epoch"] == 3


def test_legacy_train_eval_and_closed_loop_checkpoint_contract(tmp_path, monkeypatch):
    """Historical multi-step weights remain MPC-loadable; public M0 is 96-to-1."""
    import importlib.util
    from test_m0_dataset import make_data
    from src.dataloader.csv_dataset import CSVDataset
    spec = importlib.util.spec_from_file_location("hkust_m0_entrypoint", OL_ROOT / "main.py")
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    _, config = make_data(tmp_path)
    monkeypatch.setattr(entrypoint, "_runtime_config", lambda args: config)
    original_for_split = CSVDataset.for_split
    def forbid_test_view(self, split):
        assert split != "test", "Training unexpectedly accessed the test split"
        return original_for_split(self, split)
    monkeypatch.setattr(CSVDataset, "for_split", forbid_test_view)
    common = ["--data_path", str(config.data_path), "--load_tiers_path", str(config.load_tiers_path),
              "--save_dir", str(tmp_path / "runs"), "--torch_num_threads", "1"]
    train_args = entrypoint.build_legacy_parser().parse_args(common + ["--fast_dev_run", "--experiment_name", "train"])
    path = entrypoint._legacy_main(train_args)
    checkpoint = load_checkpoint(path / "best.pt")
    assert checkpoint["model_kind"] == "M0"
    assert checkpoint["epoch"] == 1
    assert not (path / "closed_loop_actual.csv").exists()
    monkeypatch.setattr(CSVDataset, "for_split", original_for_split)
    eval_args = entrypoint.build_legacy_parser().parse_args(common + ["--mode", "evaluate", "--load_model", str(path / "best.pt"),
              "--experiment_name", "eval", "--eval_limit_batches", "2"])
    evaluated = entrypoint._legacy_main(eval_args)
    assert (evaluated / "forecast_metrics.csv").is_file()
    eval_args.mode = "closed-loop"
    controller = entrypoint._build_controller(config, CSVDataset(config=config), eval_args)
    for key, value in checkpoint["controller_state_dict"].items():
        assert torch.equal(controller.state_dict()[key], value)


def test_resume_recovers_best_after_incomplete_artifact_write(tmp_path):
    from src.training_and_evaluation.m0_training import atomic_save
    saved = run(tmp_path, args(max_epochs=1))
    # Simulate an interrupted best replacement; committed last.pt includes the
    # selected weights needed to repair best.pt before resuming.
    atomic_save(tmp_path / "best.pt", {"controller_state_dict": {}, "epoch": 999})
    resumed = run(tmp_path, args(max_epochs=2, resume=tmp_path / "last.pt"))
    selected = load_checkpoint(tmp_path / "best.pt")
    assert selected["epoch"] == resumed["best_epoch"]
    assert selected["val_loss_mse"] == resumed["best_val_loss_mse"]


def test_manifest_rejects_output_tampering(tmp_path):
    import importlib.util
    import json
    from src.training_and_evaluation.m0_training import sha256_file
    spec = importlib.util.spec_from_file_location("hkust_m0_manifest_entrypoint", OL_ROOT / "main.py")
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    dispatch, tiers, manifest = tmp_path / "dispatch.csv", tmp_path / "load_tiers.csv", tmp_path / "manifest.json"
    dispatch.write_text("pv\n1\n")
    tiers.write_text("load\n2\n")
    manifest.write_text(json.dumps({"schema_version": 1, "outputs": [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (dispatch, tiers)]}))
    settings = Namespace(data_path=dispatch, load_tiers_path=tiers, data_manifest=manifest)
    identity = entrypoint._dataset_identity(settings)
    assert identity["manifest_sha256"] == sha256_file(manifest)
    dispatch.write_text("pv\n9\n")
    with pytest.raises(ValueError, match="Frozen output differs from manifest: dispatch.csv"):
        entrypoint._dataset_identity(settings)

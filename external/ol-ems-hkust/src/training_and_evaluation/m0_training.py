"""Reproducible forecast-only M0 training, with no MPC or test evaluation.

Checkpoints retain the Controller state-dict contract used by closed-loop runs.
Resume is epoch-boundary exact: optimizer, RNG and sampler states are restored.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

LOGGER = logging.getLogger("hkust_ol_ems")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprint(root: Path) -> dict[str, str]:
    files = [root / "main.py", *sorted((root / "src").rglob("*.py")),
             *sorted((root / "configurations").rglob("*.py"))]
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def json_write(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    # Complete local checkpoints contain optimizer/RNG/config objects, not just
    # weights. Only load checkpoints created by this trusted experiment code.
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions predating weights_only
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or "controller_state_dict" not in value:
        raise ValueError("Expected a complete controller checkpoint.")
    return value


def seed_run(seed: int, threads: int = 4) -> None:
    if threads < 1:
        raise ValueError("torch_num_threads must be positive")
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_versions() -> dict[str, Any]:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "platform": platform.platform(),
            "torch_num_threads": torch.get_num_threads()}


def capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "sampler": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["sampler"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint requires CUDA for exact resume")
        torch.cuda.set_rng_state_all(state["cuda"])


def batch_count(total: int, limit: float) -> int:
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Batch limits must be positive finite numbers")
    if limit <= 1:
        return max(1, math.ceil(total * limit))
    if not float(limit).is_integer():
        raise ValueError("Batch limits above 1 must be integer counts")
    return min(total, int(limit))


def _forecast_view(dataset: Any) -> Any:
    return dataset.forecast_view() if hasattr(dataset, "forecast_view") else dataset


def _epoch(controller: torch.nn.Module, loader: DataLoader, device: torch.device,
           count: int, optimizer: torch.optim.Optimizer | None) -> tuple[float, int]:
    controller.train(optimizer is not None)
    squared_error, elements, samples = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for index, batch in enumerate(loader):
            if index >= count:
                break
            history = batch["history"].to(device)
            target = batch["pv_target"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            forecast = controller.forecast_pv(history)
            loss = F.mse_loss(forecast, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite M0 loss; previous last.pt is intact")
            if optimizer is not None:
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in controller.parameters()):
                    raise FloatingPointError("Non-finite M0 gradients")
                optimizer.step()
            squared_error += float((forecast.detach().double() - target.double()).square().sum())
            elements += target.numel()
            samples += history.shape[0]
    if elements == 0:
        raise ValueError("Empty training/validation loader")
    return squared_error / elements, samples


def train_m0(controller: torch.nn.Module, train_dataset: Any, val_dataset: Any,
             output_dir: Path, args: Any, checkpoint_metadata: dict[str, Any],
             run_metadata: dict[str, Any]) -> dict[str, Any]:
    """Train using train+val only. Test datasets must never be passed here."""
    if args.max_epochs < 1 or args.patience < 1 or args.min_delta < 0:
        raise ValueError("max_epochs/patience must be positive and min_delta non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid Adam learning rate/weight decay")
    if args.train_batch_size < 1 or args.val_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.num_workers != 0:
        raise ValueError("Exact epoch-boundary resume currently requires --num_workers 0")
    device = torch.device(args.device)
    controller.to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(_forecast_view(train_dataset), batch_size=args.train_batch_size,
                              shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(_forecast_view(val_dataset), batch_size=args.val_batch_size,
                            shuffle=False, num_workers=0)
    train_count = batch_count(len(train_loader), args.limit_train_batches)
    val_count = batch_count(len(val_loader), args.limit_val_batches)
    max_epochs = args.max_epochs
    if args.fast_dev_run:
        train_count, val_count = min(train_count, args.fast_dev_run), min(val_count, args.fast_dev_run)
        max_epochs = 1
    signature = {key: getattr(args, key) for key in (
        "seed", "learning_rate", "weight_decay", "train_batch_size", "val_batch_size",
        "patience", "min_delta", "limit_train_batches", "limit_val_batches",
        "fast_dev_run", "device", "torch_num_threads", "num_workers")}
    signature.update({"dataset_identity": checkpoint_metadata["dataset_identity"],
                      "code_sha256": checkpoint_metadata["code_sha256"],
                      "config": checkpoint_metadata["config"],
                      "model_architecture": checkpoint_metadata["model_architecture"],
                      "runtime": runtime_versions()})
    rows: list[dict[str, Any]] = []
    best_snapshot: dict[str, Any] | None = None
    best_loss, significant_best, best_epoch, bad_epochs, first_epoch = math.inf, math.inf, 0, 0, 1
    if args.resume:
        previous = load_checkpoint(args.resume)
        if previous.get("training_signature") != signature:
            saved = previous.get("training_signature", {})
            different = [key for key in signature if saved.get(key) != signature[key]]
            raise ValueError("Resume settings/data/code/runtime differ: " + ", ".join(different))
        if previous.get("format_version") != 2 or "optimizer_state_dict" not in previous:
            raise ValueError("Resume requires a complete format_version=2 last.pt")
        if Path(args.resume).resolve() != (output_dir / "last.pt").resolve():
            raise ValueError("Resume must use this experiment's last.pt")
        controller.load_state_dict(previous["controller_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        first_epoch = previous["epoch"] + 1
        best_loss, best_epoch = previous["best_val_loss_mse"], previous["best_epoch"]
        significant_best, bad_epochs = previous["early_stop_reference"], previous["bad_epochs"]
        rows = previous["metrics_history"]
        best_snapshot = previous.get("best_model_snapshot")
        if best_snapshot is None or best_snapshot["epoch"] != best_epoch:
            raise ValueError("last.pt is missing its recoverable best-model snapshot")
        # Recover the matching best even if a process stopped between the two
        # atomic checkpoint replacements. last.pt is the committed epoch.
        atomic_save(output_dir / "best.pt", best_snapshot)
        if first_epoch > max_epochs:
            raise ValueError("Resume max_epochs must exceed the saved epoch")
        if bad_epochs >= args.patience:
            raise ValueError("This run already met its frozen early-stopping rule")
        restore_rng(previous["rng_state"], generator)
        LOGGER.info("Resuming epoch %d with best epoch %d", first_epoch, best_epoch)
    metadata = {**run_metadata, "training_signature": signature, "status": "running",
                "selection_metric": "val_loss_mse in kW^2, all valid origin/horizon pairs",
                "best_selection": "strict minimum val_loss_mse; min_delta affects patience only",
                "scheduler": "none; fixed Adam learning rate", "test_evaluated": False,
                "train_samples_available": len(train_dataset), "val_samples_available": len(val_dataset)}
    json_write(output_dir / "run_metadata.json", metadata)
    started = time.perf_counter()
    for epoch in range(first_epoch, max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_samples = _epoch(controller, train_loader, device, train_count, optimizer)
        val_loss, val_samples = _epoch(controller, val_loader, device, val_count, None)
        is_best = val_loss < best_loss
        if is_best:
            best_loss, best_epoch = val_loss, epoch
        if val_loss < significant_best - args.min_delta:
            significant_best, bad_epochs = val_loss, 0
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "train_loss_mse": train_loss, "val_loss_mse": val_loss,
               "learning_rate": optimizer.param_groups[0]["lr"], "train_samples": train_samples,
               "val_samples": val_samples, "seconds": time.perf_counter() - epoch_started,
               "best_epoch": best_epoch, "bad_epochs": bad_epochs}
        rows.append(row)
        checkpoint = {**checkpoint_metadata, "format_version": 2, "model_kind": "M0",
                      "controller_state_dict": controller.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": None,
                      "epoch": epoch, "val_loss_mse": val_loss, "best_val_loss_mse": best_loss,
                      "best_epoch": best_epoch, "early_stop_reference": significant_best,
                      "bad_epochs": bad_epochs, "rng_state": capture_rng(generator),
                      "training_signature": signature, "metrics_history": rows}
        if is_best:
            best_snapshot = {**checkpoint_metadata, "format_version": 2, "model_kind": "M0",
                             "controller_state_dict": {key: value.detach().cpu().clone()
                                                       for key, value in controller.state_dict().items()},
                             "epoch": epoch, "val_loss_mse": val_loss, "best_epoch": best_epoch,
                             "best_val_loss_mse": best_loss, "selection_metric": "val_loss_mse"}
        checkpoint["best_model_snapshot"] = best_snapshot
        # last.pt is the recoverable epoch commit. Both contain immediately
        # usable weights; reporting/plotting cannot destroy completed training.
        atomic_save(output_dir / "last.pt", checkpoint)
        if is_best:
            atomic_save(output_dir / "best.pt", best_snapshot)
        with (output_dir / "metrics.csv.tmp").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(output_dir / "metrics.csv.tmp", output_dir / "metrics.csv")
        metadata.update({"epochs_completed": epoch, "best_epoch": best_epoch,
                         "best_val_loss_mse": best_loss, "bad_epochs": bad_epochs,
                         "latest_metrics": row, "elapsed_seconds_this_invocation": time.perf_counter() - started})
        json_write(output_dir / "run_metadata.json", metadata)
        LOGGER.info("Epoch %d/%d train MSE %.6f | val MSE %.6f | best %d | %.2fs",
                    epoch, max_epochs, train_loss, val_loss, best_epoch, row["seconds"])
        if bad_epochs >= args.patience:
            break
    metadata.update({"status": "completed", "stop_reason": "early_stopping" if bad_epochs >= args.patience else
                     "fast_dev_run" if args.fast_dev_run else "max_epochs",
                     "model_status": "smoke_trained" if args.fast_dev_run or args.limit_train_batches != 1.0
                     or args.limit_val_batches != 1.0 else "trained_M0"})
    json_write(output_dir / "run_metadata.json", metadata)
    return metadata

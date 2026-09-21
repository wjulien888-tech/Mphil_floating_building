"""Train the two frozen PV-only models, reload best weights and evaluate in full."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "external/ol-ems-hkust/.venv/bin/python"
FROZEN = ROOT / "output/frozen/pv_only_2021_2023_96to1_v1_20260920"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_runs(destination: Path) -> list[dict]:
    """Build both official-entry commands without loading data or launching jobs."""
    destination = destination.resolve()
    common = ["--data_format", "pv_only", "--data_path", str(FROZEN / "pv_timeseries.csv"),
              "--data_manifest", str(FROZEN / "manifest.json"),
              "--hidden_dim", "64", "--num_layers", "2", "--dropout", "0.2",
              "--fc_hidden_size", "5", "--learning_rate", "0.001",
              "--weight_decay", "0.000001", "--train_batch_size", "64",
              "--val_batch_size", "64", "--max_epochs", "50", "--patience", "8",
              "--min_delta", "0.01", "--num_workers", "0", "--device", "cpu",
              "--torch_num_threads", "1", "--seed", "42", "--daylight_threshold_kw", "0.1",
              "--save_dir", str(destination)]
    runs = []
    for model in ("lstm", "gru"):
        name = f"{model}_seed_42"
        entry = [str(PYTHON), str(ROOT / "external/ol-ems-hkust/main.py"),
                 "--model_type", model]
        runs.append({
            "model": model, "seed": 42, "name": name,
            "train_command": entry + common + ["--mode", "train", "--experiment_name", name],
            "evaluation_commands": {
                split: entry + common + ["--mode", "evaluate", "--load_model",
                        str(destination / name / "best.pt"), "--eval_split", split,
                        "--experiment_name", f"{name}_{split}"] for split in ("val", "test")},
        })
    return runs


def run_suite(destination: Path) -> None:
    sys.path.insert(0, str(ROOT / "model_variants"))
    from _m0_variant_runner import _dataset_identity, _variant_code_fingerprint, build_parser

    # Verify every frozen output before creating a formal experiment.
    identity = _dataset_identity(build_parser("LSTM").parse_args([]))
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    runs = build_runs(destination)
    protocol = {
        "protocol": "PV-only-2021-2023-LSTM-GRU-96to1-seed42-v1", "created_utc": now(),
        "seeds": [42], "dataset_identity": identity,
        "architecture": {"lookback_steps": 96, "horizon_steps": 1, "features": ["pv"],
                         "layers": 2, "hidden_size": 64, "dropout": 0.2, "head": [64, 5, 1]},
        "selection": "Each model's strict minimum full validation MSE; never test-based selection",
        "stopping": "At most 50 epochs; patience 8 with min_delta 0.01 kW^2",
        "evaluation": "After both trainings finish, reload each best.pt in independent full val/test processes",
        "provenance": "Constructed building PV: annual PVGIS physical baseline times causal HKUST PV disturbance",
        "parallel_train_processes": 2, "torch_threads_per_process": 1,
        "limitations": "One common seed, not a multiple-seed stability result; old 672-to-96 M0 is a different task",
        "runs": runs,
    }
    write_json(destination / "protocol.json", protocol)
    hashes = _variant_code_fingerprint()
    hashes[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
    formal_entry = ROOT / "tools/run_formal_m0.py"
    hashes[str(formal_entry.relative_to(ROOT))] = digest(formal_entry)
    for relative in hashes:
        copied = destination / "source_snapshot" / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, copied)
    write_json(destination / "source_sha256.json", hashes)
    shutil.copy2(FROZEN / "manifest.json", destination / "data_manifest.json")
    with (destination / "environment.txt").open("x") as stream:
        subprocess.run([str(PYTHON), "-m", "pip", "freeze"], cwd=ROOT, stdout=stream, check=True)
    status = {"stage": "training", "pid": os.getpid(), "started_utc": now(), "completed": [],
              "runs": {run["name"]: {"stage": "pending"} for run in runs}}
    write_json(destination / "suite_status.json", status)

    def execute(name: str, command: list[str]) -> str:
        print(f"{now()} START {name}", flush=True)
        with (destination / f"{name}_console.log").open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            write_json(destination / f"{name}_process.json", {"pid": child.pid, "started_utc": now(),
                                                           "command": command})
            result = child.wait()
        if result:
            raise RuntimeError(f"{name} exited {result}; see {name}_console.log")
        print(f"{now()} COMPLETE {name}", flush=True)
        return name

    def verify_sources() -> None:
        for relative, recorded in hashes.items():
            if digest(ROOT / relative) != recorded:
                raise RuntimeError(f"Source changed during this experiment: {relative}")
        if _dataset_identity(build_parser("LSTM").parse_args([])) != identity:
            raise RuntimeError("Frozen data identity changed during this experiment")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(execute, run["name"], run["train_command"]): run for run in runs}
            for run in runs:
                status["runs"][run["name"]]["stage"] = "training"
            write_json(destination / "suite_status.json", status)
            for future in as_completed(futures):
                name = future.result()
                status["completed"].append(name)
                status["runs"][name]["stage"] = "trained"
                write_json(destination / "suite_status.json", status)
        verify_sources()
        status["stage"] = "independent_evaluation"
        write_json(destination / "suite_status.json", status)
        for split in ("val", "test"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(execute, f"{run['name']}_{split}", run["evaluation_commands"][split])
                           for run in runs]
                for future in as_completed(futures):
                    status["completed"].append(future.result())
                    write_json(destination / "suite_status.json", status)
        verify_sources()
        status["stage"] = "summarizing"
        write_json(destination / "suite_status.json", status)
        summarizer = ROOT / "tools/summarize_pvonly_models.py"
        shutil.copy2(summarizer, destination / "source_snapshot" / "tools" / summarizer.name)
        write_json(destination / "summary_source_sha256.json", {str(summarizer.relative_to(ROOT)): digest(summarizer)})
        execute("summarize", [str(PYTHON), str(summarizer), "--suite", str(destination)])
        status.update(stage="completed", finished_utc=now())
        for run in status["runs"].values():
            run["stage"] = "evaluated"
    except BaseException as error:
        status.update(stage="failed", error=str(error), failed_utc=now())
        raise
    finally:
        write_json(destination / "suite_status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run_suite(parser.parse_args().output)

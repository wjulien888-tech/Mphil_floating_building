"""User-started six-run PV suite with live, labelled terminal output.

--dry-run verifies the frozen inputs and prints commands without creating an
experiment or running a model. Training is followed by validation only. Use
--mode evaluate --eval-split test after choosing windows on validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "output/frozen/pv_only_2021_2023_multihorizon96_v1_20260921"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--frozen-root", type=Path, default=FROZEN)
    parser.add_argument("--output", type=Path, help="New training suite directory; never overwritten.")
    parser.add_argument("--suite", type=Path, help="Existing suite, required for independent evaluation.")
    parser.add_argument("--models", nargs="+", choices=("lstm", "gru"), default=["lstm", "gru"])
    parser.add_argument("--lookbacks", nargs="+", type=int, choices=(96, 192, 288), default=[288, 192, 96])
    parser.add_argument("--jobs", type=int, choices=range(1, 7), default=6,
                        help="Parallel processes; use 2 if six processes use too much memory.")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print commands; do not train/evaluate.")
    return parser


def build_runs(destination: Path, frozen_root: Path = FROZEN, *, models=None,
               lookbacks=None, python: str | None = None) -> list[dict]:
    """Commands only: importing this module or calling this function is inert."""
    runs = []
    for model in models or ["lstm", "gru"]:
        for lookback in lookbacks or [288, 192, 96]:
            name = f"{model}_L{lookback}_H96_seed42"
            data = frozen_root.resolve() / f"lookback_{lookback}"
            common = [python or sys.executable, "-u", str(ROOT / "external/ol-ems-hkust/main.py"),
                      "--model_type", model, "--data_format", "pv_only",
                      "--data_path", str(data / "pv_timeseries.csv"),
                      "--data_manifest", str(data / "manifest.json"),
                      "--lookback_steps", str(lookback), "--prediction_horizon", "96",
                      "--anchor_lookback_steps", "288", "--hidden_dim", "64", "--num_layers", "2",
                      "--dropout", "0.2", "--fc_hidden_size", "64", "--learning_rate", "0.001",
                      "--weight_decay", "0.000001", "--train_batch_size", "64", "--val_batch_size", "64",
                      "--max_epochs", "50", "--patience", "8", "--min_delta", "0.01",
                      "--num_workers", "0", "--device", "cpu", "--torch_num_threads", "1",
                      "--log_every_batches", "100", "--seed", "42", "--daylight_threshold_kw", "0.1",
                      "--save_dir", str(destination.resolve())]
            runs.append({"name": name, "model": model, "lookback_steps": lookback,
                         "prediction_horizon": 96, "seed": 42,
                         "train_command": common + ["--mode", "train", "--experiment_name", name],
                         "evaluation_commands": {
                             split: common + ["--mode", "evaluate", "--load_model", str(destination / name / "best.pt"),
                                              "--eval_split", split, "--experiment_name", f"{name}_{split}"]
                             for split in ("val", "test")}})
    return runs


def validate_inputs(runs: list[dict]) -> tuple[dict, dict]:
    sys.path.insert(0, str(ROOT / "model_variants"))
    from _m0_variant_runner import _dataset_identity, _runtime_config, _variant_code_fingerprint, build_parser
    identities = {}
    for run in runs:
        key = str(run["lookback_steps"])
        if key in identities:
            continue
        command = run["train_command"]
        # The public main adds --model_type; the shared runner receives the type separately.
        args = build_parser(run["model"].upper()).parse_args(command[5:])
        _runtime_config(args)
        identities[key] = _dataset_identity(args)
    hashes = _variant_code_fingerprint()
    for path in [Path(__file__), ROOT / "prepare_pv_only_multihorizon.py"]:
        hashes[str(path.relative_to(ROOT))] = digest(path)
    return identities, hashes


class LiveProcesses:
    """Own every child process and terminate its group on Ctrl+C or failure."""
    def __init__(self, destination: Path, status_path: Path):
        self.destination = destination
        self.status_path = status_path
        self.status = {"stage": "starting", "pid": os.getpid(), "started_utc": now(), "jobs": {}}
        self.lock = threading.RLock()
        self.console_lock = threading.Lock()
        self.stopping = threading.Event()
        self.active: dict[str, subprocess.Popen] = {}

    def emit(self, line: str) -> None:
        with self.console_lock:
            print(line, flush=True)

    def update(self, **values) -> None:
        with self.lock:
            self.status.update(values)
            write_json(self.status_path, self.status)

    def execute(self, name: str, command: list[str]) -> str:
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        started = now()
        with (self.destination / f"{name}_console.log").open("x", encoding="utf-8") as logfile:
            with self.lock:
                if self.stopping.is_set():
                    raise RuntimeError("Suite stopped before starting the next process")
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, start_new_session=True)
                self.active[name] = child
                self.status["jobs"][name] = {"stage": "running", "pid": child.pid, "started_utc": started}
                write_json(self.status_path, self.status)
            self.emit(f"[{name}] START pid={child.pid}")
            write_json(self.destination / f"{name}_process.json",
                       {"pid": child.pid, "started_utc": started, "command": command})
            try:
                assert child.stdout is not None
                for line in child.stdout:
                    logfile.write(line)
                    logfile.flush()
                    self.emit(f"[{name}] {line.rstrip()}")
                code = child.wait()
            finally:
                if child.poll() is None:
                    self._signal(child, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._signal(child, signal.SIGKILL)
                        child.wait()
                if child.stdout:
                    child.stdout.close()
                with self.lock:
                    self.active.pop(name, None)
                    self.status["jobs"][name].update(stage="completed" if child.returncode == 0 else "failed",
                                                       returncode=child.returncode, finished_utc=now())
                    write_json(self.status_path, self.status)
            if code:
                raise RuntimeError(f"{name} exited {code}; see {name}_console.log")
            self.emit(f"[{name}] COMPLETE")
            return name

    @staticmethod
    def _signal(child: subprocess.Popen, sig: int) -> None:
        if child.poll() is None:
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass

    def stop(self) -> None:
        self.stopping.set()
        with self.lock:
            children = list(self.active.values())
        for child in children:
            self._signal(child, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal(child, signal.SIGKILL)

    def phase(self, name: str, commands: list[tuple[str, list[str]]], jobs: int) -> None:
        self.update(stage=name)
        pool = ThreadPoolExecutor(max_workers=jobs)
        futures = []
        try:
            futures = [pool.submit(self.execute, label, command) for label, command in commands]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            self.stop()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def summarize(destination: Path, runs: list[dict], split: str) -> Path:
    rows = []
    for run in runs:
        summary = json.loads((destination / f"{run['name']}_{split}" / "evaluation_summary.json").read_text())
        training = json.loads((destination / run["name"] / "run_metadata.json").read_text())
        if (training.get("status") != "completed"
                or not summary["complete_split"] or summary["horizon_steps"] != 96
                or summary["checkpoint_sha256"] != digest(destination / run["name"] / "best.pt")
                or summary["dataset_identity"] != training["dataset_identity"]):
            raise ValueError(f"Incomplete or mismatched evaluation for {run['name']}")
        for metric in summary["metrics"]:
            if metric["period"] == "all" and metric["count"] != summary["forecast_origins"] * 96:
                raise ValueError(f"Evaluation did not score every horizon step for {run['name']}")
            rows.append({"model": run["model"], "lookback_steps": run["lookback_steps"],
                         "history_days": run["lookback_steps"] // 96, "horizon_steps": 96,
                         "seed": 42, "split": split, "best_epoch": training["best_epoch"],
                         "epochs_completed": training["epochs_completed"], **metric})
    path = destination / f"{split}_comparison.csv"
    # Additional selected evaluations can be summarized without dropping prior configurations.
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            previous = list(csv.DictReader(stream))
        replaced = {(row["model"], str(row["lookback_steps"])) for row in rows}
        rows = [row for row in previous if (row["model"], row["lookback_steps"]) not in replaced] + rows
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{split} comparison: {path}", flush=True)
    print("model  history  best_epoch  RMSE_kW  MAE_kW  R2 (all 96 horizon steps)", flush=True)
    for row in rows:
        if row["method"] == "M0" and row["period"] == "all":
            r2 = "n/a" if row["r2"] in (None, "") else f"{float(row['r2']):.5f}"
            print(f"{row['model']:5s}  {row['history_days']} days   {row['best_epoch']:>3}         "
                  f"{float(row['rmse_kw']):.4f}   {float(row['mae_kw']):.4f}   {r2}", flush=True)
    return path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if len(set(args.models)) != len(args.models) or len(set(args.lookbacks)) != len(args.lookbacks):
        raise ValueError("Do not repeat models or lookbacks")
    if args.mode == "evaluate":
        if args.suite is None or args.output is not None:
            raise ValueError("Evaluation requires --suite and does not accept --output")
        destination = args.suite.expanduser().resolve()
    else:
        if args.suite is not None or args.eval_split != "val":
            raise ValueError("Training uses --output and validation only; evaluate test separately")
        destination = (args.output or ROOT / "experiments" /
                       f"pv_multihorizon96_{datetime.now():%Y%m%d_%H%M%S_%f}").expanduser().resolve()
    runs = build_runs(destination, args.frozen_root, models=args.models, lookbacks=args.lookbacks)
    identities, hashes = validate_inputs(runs)
    print(f"Frozen data verified. Models={args.models}, lookbacks={args.lookbacks}, horizon=96, jobs={args.jobs}", flush=True)
    print(f"Output: {destination}", flush=True)
    if args.dry_run:
        for run in runs:
            command = run["train_command"] if args.mode == "train" else run["evaluation_commands"][args.eval_split]
            print(shlex.join(command), flush=True)
        print("DRY RUN: no model executed; no experiment directory created.", flush=True)
        return
    if args.mode == "train":
        destination.mkdir(parents=True, exist_ok=False)
        protocol = {"protocol": "PV-only-L96-L192-L288-to-H96-seed42-v1", "created_utc": now(),
                    "architecture": {"layers": 2, "hidden_dim": 64, "dropout": 0.2, "head": [64, 64, 96]},
                    "selection": "Minimum full-validation MSE; test is not automatically evaluated or used for selection",
                    "common_history_quality_steps": 288, "max_epochs": 50, "patience": 8,
                    "min_delta_kw2": 0.01, "parallel_jobs": args.jobs, "torch_threads_per_process": 1,
                    "provenance": "Constructed building PV, unchanged from the 2021-2023 frozen source",
                    "seeds": [42], "dataset_identities": identities, "runs": runs}
        write_json(destination / "protocol.json", protocol)
        write_json(destination / "source_sha256.json", hashes)
        for relative in hashes:
            copied = destination / "source_snapshot" / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, copied)
        for lookback in identities:
            shutil.copy2(args.frozen_root / f"lookback_{lookback}" / "manifest.json",
                         destination / f"data_manifest_L{lookback}.json")
        with (destination / "environment.txt").open("x") as stream:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
        status_path = destination / "suite_status.json"
    else:
        protocol = json.loads((destination / "protocol.json").read_text())
        for lookback, identity in identities.items():
            if identity != protocol["dataset_identities"].get(lookback):
                raise ValueError("Evaluation data differs from the suite's frozen inputs")
        known_runs = {run["name"] for run in protocol["runs"]}
        for run in runs:
            if run["name"] not in known_runs or not (destination / run["name"] / "best.pt").is_file():
                raise ValueError(f"Missing trained run: {run['name']}")
            training = json.loads((destination / run["name"] / "run_metadata.json").read_text())
            if training.get("status") != "completed":
                raise ValueError(f"Training is not complete: {run['name']}")
            if (destination / f"{run['name']}_{args.eval_split}").exists():
                raise FileExistsError(f"Evaluation already exists: {run['name']}_{args.eval_split}")
        status_path = destination / f"evaluation_status_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    live = LiveProcesses(destination, status_path)

    def verify_unchanged():
        current_identities, current_hashes = validate_inputs(runs)
        if current_identities != identities or current_hashes != hashes:
            raise RuntimeError("Source or frozen data changed while the suite was running")

    try:
        if args.mode == "train":
            live.phase("training", [(run["name"], run["train_command"]) for run in runs], args.jobs)
            verify_unchanged()
        split = "val" if args.mode == "train" else args.eval_split
        live.phase(f"evaluating_{split}", [(f"{run['name']}_{split}", run["evaluation_commands"][split]) for run in runs], args.jobs)
        verify_unchanged()
        summarize(destination, runs, split)
        live.update(stage="completed", finished_utc=now())
    except BaseException as error:
        live.stop()
        live.update(stage="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error) or type(error).__name__, finished_utc=now())
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped this suite's child processes. Completed epoch checkpoints remain on disk.", flush=True)
        raise SystemExit(130)

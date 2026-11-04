"""Train GRU/PatchTST at L288/L672 against a shared 672-step history guard.

All four runs start from seed42, train on the full eligible split, select by
validation MSE, and evaluate untouched test targets. Prior L96 metrics are
included only after verifying the ordered validation/test origin hashes match.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "output/frozen/pv_only_L288_L672_H96_common672_20261104"
PREVIOUS = ROOT / "experiments/open_source_comparison_20261104_L96_H96_seed42_v2"
LOCK = threading.Lock()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def build_commands(output, frozen=FROZEN):
    commands = {}
    for model in ("gru", "patchtst"):
        for lookback in (288, 672):
            name = f"{model}_L{lookback}_H96_seed42"
            script = ROOT / ("model_variants/gru_2layer_m0.py" if model == "gru"
                             else "model_variants/open_source_m0.py")
            common = [sys.executable, "-u", str(script)]
            if model == "patchtst":
                common += ["--model_type", "patchtst"]
            common += ["--data_path", str(frozen / f"lookback_{lookback}/pv_timeseries.csv"),
                       "--data_manifest", str(frozen / f"lookback_{lookback}/manifest.json"),
                       "--lookback_steps", str(lookback), "--prediction_horizon", "96",
                       "--anchor_lookback_steps", "672", "--hidden_dim", "64",
                       "--num_layers", "2", "--fc_hidden_size", "64", "--dropout", "0.2",
                       "--learning_rate", "0.001", "--weight_decay", "0.000001",
                       "--train_batch_size", "64", "--val_batch_size", "64",
                       "--max_epochs", "50", "--patience", "8", "--min_delta", "0.01",
                       "--device", "cpu", "--torch_num_threads", "1", "--num_workers", "0",
                       "--log_every_batches", "100", "--seed", "42",
                       "--save_dir", str(output)]
            commands[name] = {"train": common + ["--mode", "train", "--experiment_name", name]}
            for split in ("val", "test"):
                commands[name][split] = common + ["--mode", "evaluate", "--load_model",
                    str(output / name / "best.pt"), "--eval_split", split,
                    "--experiment_name", f"{name}_{split}"]
    return commands


def validate_origin_hashes(frozen=FROZEN):
    manifests = [json.loads((frozen / f"lookback_{length}/manifest.json").read_text())
                 for length in (288, 672)]
    previous = json.loads((ROOT / "output/frozen/pv_only_2021_2023_multihorizon96_v1_20260921/lookback_96/manifest.json").read_text())
    for split in ("train", "val", "test"):
        assert manifests[0]["splits"][split]["anchor_indices_sha256"] == manifests[1]["splits"][split]["anchor_indices_sha256"]
    for split in ("val", "test"):
        assert manifests[0]["splits"][split]["anchor_indices_sha256"] == previous["splits"][split]["anchor_indices_sha256"]
    return {"common_long_window_origins": True, "same_heldout_origins_as_L96": True,
            "long_window_train_samples": manifests[0]["splits"]["train"]["valid_windows"],
            "prior_L96_train_samples": previous["splits"]["train"]["valid_windows"],
            "val_samples": manifests[0]["splits"]["val"]["valid_windows"],
            "test_samples": manifests[0]["splits"]["test"]["valid_windows"]}


def collect_results(output):
    frames = []
    for model in ("gru", "patchtst"):
        for length in (96, 288, 672):
            name = f"{model}_L{length}_H96_seed42"
            for split in ("val", "test"):
                directory = PREVIOUS / f"{model}_{split}" if length == 96 else output / f"{name}_{split}"
                frame = pd.read_csv(directory / "forecast_metrics.csv")
                frame.insert(0, "model", model)
                frame.insert(1, "lookback_steps", length)
                frame.insert(2, "history_days", length // 96)
                frame.insert(3, "split", split)
                frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    for split in ("val", "test"):
        reference = table[(table.split == split) & (table.method == "persistence_24h")]
        for period in reference.period.unique():
            group = reference[reference.period == period]
            for column in ("count", "mae_kw", "rmse_kw", "r2"):
                if group[column].isna().any() or group[column].max() - group[column].min() > 1e-10:
                    raise ValueError(f"Mismatched evaluation samples: {split}/{period}/{column}")
    table.to_csv(output / "comparison.csv", index=False)
    return table


def run_suite(output, jobs=4):
    eligibility = validate_origin_hashes()
    output.mkdir(parents=True, exist_ok=False)
    commands = build_commands(output)
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "pid": os.getpid(), "runs": {}}
    write_json(output / "protocol.json", {"commands": commands, "eligibility": eligibility,
        "source": "Constructed PV; not measured floating-building telemetry",
        "prior_L96": str(PREVIOUS), "selection": "validation MSE only; full test split after training",
        "note": "All long-window runs freshly initialized at seed42. Original native output heads retained. "
                "50 epochs maximum, patience8. Previous L96 trained on 384 additional eligible windows."})
    for relative in ("model_variants", "external/forecast_baselines"):
        shutil.copytree(ROOT / relative, output / "source_snapshot" / relative,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for relative in ("tools/run_history_window_comparison.py", "prepare_pv_only_multihorizon.py"):
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    def update(name, stage, **values):
        with LOCK:
            status["runs"][name] = {"stage": stage, **values}
            write_json(output / "status.json", status)

    def work(name, stages):
        try:
            for stage, command in stages.items():
                with (output / f"{name}_{stage}_console.log").open("x") as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    update(name, stage, pid=child.pid)
                    code = child.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, command)
            update(name, "completed")
        except BaseException as error:
            update(name, "failed", error=repr(error))
            raise
        print(f"Completed {name}", flush=True)

    write_json(output / "status.json", status)
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(work, name, stages) for name, stages in commands.items()]
            for future in as_completed(futures):
                future.result()
        table = collect_results(output)
        status["status"] = "completed"
        write_json(output / "status.json", status)
        print(table[(table.method == "M0") & (table.split == "test") &
                    (table.period != "non_daylight")].to_string(index=False), flush=True)
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        write_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    run_suite(args.output.resolve(), args.jobs)

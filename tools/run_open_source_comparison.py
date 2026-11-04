"""One-seed L96/H96 comparison, reusing the frozen GRU training protocol."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "experiments/pv_multihorizon96_20260921_182457_137681"


def run_suite(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((PREVIOUS / "protocol.json").read_text())
    # Recover exact original commands instead of hand-copying data/optimizer flags.
    def find_gru(value):
        if isinstance(value, dict):
            if value.get("name") == "gru_L96_H96_seed42":
                return value
            for item in value.values():
                found = find_gru(item)
                if found:
                    return found
        if isinstance(value, list):
            for item in value:
                found = find_gru(item)
                if found:
                    return found
    previous = find_gru(protocol)
    if previous is None:
        raise ValueError("Original GRU protocol not found")

    def replace_flag(command, flag, value):
        command[command.index(flag) + 1] = str(value)

    commands = {}
    for model in ("dlinear", "patchtst", "gru"):
        commands[model] = {}
        for stage in (("train", "val", "test") if model != "gru" else ("val", "test")):
            command = list(previous["train_command"] if stage == "train" else
                           previous["evaluation_commands"][stage])
            command[0] = sys.executable
            if model != "gru":
                command[2] = str(ROOT / "model_variants/open_source_m0.py")
                replace_flag(command, "--model_type", model)
                if stage != "train":
                    replace_flag(command, "--load_model", output / f"{model}_train/best.pt")
            replace_flag(command, "--save_dir", output)
            replace_flag(command, "--experiment_name", f"{model}_{stage}")
            commands[model][stage] = command
    manifest = {"created": datetime.now().isoformat(), "protocol": "L96/H96, PV-only, seed42",
                "gru_reused": str(PREVIOUS / "gru_L96_H96_seed42/best.pt"),
                "note": "Single-seed architecture comparison; no hyperparameter search. "
                        "New baselines retain linear outputs in training and clamp negative PV at evaluation. "
                        "GRU retains its original sigmoid head. Constructed PV labels, not field telemetry.",
                "commands": commands}
    (output / "protocol.json").write_text(json.dumps(manifest, indent=2))

    def work(model):
        for stage, command in commands[model].items():
            with (output / f"{model}_{stage}_console.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return model

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(work, commands))
        frames = []
        for model in commands:
            for split in ("val", "test"):
                frame = pd.read_csv(output / f"{model}_{split}/forecast_metrics.csv")
                frame.insert(0, "model", model)
                frame.insert(1, "split", split)
                frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        # Identical persistence/target statistics verify comparison eligibility.
        for split in ("val", "test"):
            reference = table[(table.split == split) & (table.method == "persistence_24h")]
            for period in reference.period.unique():
                subset = reference[reference.period == period]
                for column in ("count", "mae_kw", "rmse_kw", "r2"):
                    if subset[column].max() - subset[column].min() > 1e-10:
                        raise ValueError(f"Different evaluation samples: {split}/{period}/{column}")
        table.to_csv(output / "comparison.csv", index=False)
        (output / "status.json").write_text(json.dumps({"status": "completed"}))
        print(table[(table.method == "M0") & (table.period != "non_daylight")].to_string(index=False), flush=True)
    except BaseException as error:
        (output / "status.json").write_text(json.dumps({"status": "failed", "error": repr(error)}))
        raise


if __name__ == "__main__":
    run_suite(Path(sys.argv[1]).resolve())

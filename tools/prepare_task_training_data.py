#!/usr/bin/env python3
"""Prepare M1/M2 event tables from M0 metadata, without a model or MPC solve.

Reads frozen input tables and the original M0 run metadata. No checkpoint
weights are loaded, no forecast is computed, and no training is started.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "model_variants"),
               str(ROOT / "external/ol-ems-hkust")]

import pandas as pd

from _task_training_data import (
    prepare_training_data, validate_checkpoint_splits, write_event_tables,
)
from configurations.hkust_15min import SplitWindow
from _pv_only_dataset import PVForecastConfig, PVOnlyDataset


DEFAULT_METADATA = ROOT / (
    "experiments/pv_multihorizon96_20260921_182457_137681/"
    "gru_L96_H96_seed42/run_metadata.json"
)
DEFAULT_DISPATCH_ROOT = ROOT / "output/frozen/task_dispatch_2021_2023_v1_20261003"
DEFAULT_OUTPUT = ROOT / "output/prepared/task_training_2021_2023_v2_20261003"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False, default=str) + "\n")


def dataset_from_metadata(path: Path):
    """Recreate only the data interface used by the shared event builder."""
    metadata = json.loads(path.read_text())
    config = dict(metadata["config"])
    validate_checkpoint_splits(config)
    summary = metadata["dataset"]
    identity = metadata["dataset_identity"]
    pv_path = Path(config["data_path"]).expanduser()
    if not pv_path.is_absolute():
        pv_path = ROOT / pv_path
    pv_path = pv_path.resolve()
    manifest_path = pv_path.parent / "manifest.json"
    for source, key in ((pv_path, "pv_timeseries_sha256"),
                        (manifest_path, "manifest_sha256")):
        expected = identity[key]
        if sha256_file(source) != expected:
            raise ValueError(f"Frozen source hash differs from M0 metadata: {source}")

    for key in ("lookback_steps", "prediction_horizon"):
        if int(config[key]) != int(summary[key]):
            raise ValueError(f"M0 config and dataset summary disagree on {key}")
    splits = config["split_windows"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("M0 metadata must contain train, val and test split windows")
    for name, window in splits.items():
        if name in summary.get("split_dates", {}):
            dates = summary["split_dates"][name]
            if [pd.Timestamp(window["start"]), pd.Timestamp(window["end"])] != [
                    pd.Timestamp(value) for value in dates]:
                raise ValueError(f"M0 config and dataset summary disagree on {name} dates")

    source_frame = pd.read_csv(pv_path)
    source_frame = source_frame.rename(columns={
        config.get("timestamp_column", "datetime"): "datetime",
        config.get("source_columns", {"pv": "pv_kW"})["pv"]: "pv",
    })
    config["data_path"] = pv_path
    config["split_windows"] = {
        name: SimpleNamespace(**window) for name, window in splits.items()
    }
    dataset = SimpleNamespace(
        config=SimpleNamespace(**config), source_frame=source_frame,
        lookback_window=int(config["lookback_steps"]),
        prediction_horizon=int(config["prediction_horizon"]),
        pv_min=float(summary["pv_min_kw"]), pv_max=float(summary["pv_max_kw"]),
    )
    return dataset, identity


def main(args):
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    metadata_path = None
    if args.forecast_data_dir:
        folder = args.forecast_data_dir.expanduser().resolve()
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["outputs"]:
            path = (folder / record["path"]).resolve()
            if not path.is_relative_to(folder) or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Frozen forecast input changed: {record['path']}")
        parameters = manifest["parameters"]
        config = PVForecastConfig(data_path=folder / "pv_timeseries.csv",
            lookback_steps=parameters["lookback_steps"], prediction_horizon=parameters["prediction_horizon"],
            anchor_lookback_steps=parameters["anchor_lookback_steps"],
            split_windows={name: SplitWindow(item["start_inclusive"], item["end_exclusive"])
                           for name, item in manifest["splits"].items()},
            pv_input_provenance=manifest.get("pv_input_provenance", PVForecastConfig().pv_input_provenance))
        dataset = PVOnlyDataset(config)
        original_identity = {"manifest_sha256": sha256_file(manifest_path),
                             "pv_timeseries_sha256": sha256_file(config.data_path)}
    else:
        metadata_path = (args.metadata or DEFAULT_METADATA).expanduser().resolve()
        dataset, original_identity = dataset_from_metadata(metadata_path)
    bundle = prepare_training_data(dataset, args.scenario_config,
                                   args.dispatch, args.load_tiers)
    # Create only after all data checks pass; exist_ok=False also guards races.
    output.mkdir(parents=True, exist_ok=False)
    write_event_tables(bundle, output)
    write_json(output / "preparation_manifest.json", {
        "format_version": "task-training-data-only-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "formal_training": False,
        "model_constructed": False,
        "checkpoint_weights_loaded": False,
        "mpc_solved": False,
        "m0_run_metadata": ({"path": str(metadata_path), "sha256": sha256_file(metadata_path)}
                            if metadata_path else None),
        "forecast_data_dir": str(args.forecast_data_dir) if args.forecast_data_dir else None,
        "original_m0_dataset_identity": original_identity,
        "preparation_script_sha256": sha256_file(Path(__file__)),
        "split_intervals": bundle.metadata["split_intervals"],
        "pv_scaler": bundle.metadata["pv_scaler"],
        "sources": bundle.metadata["sources"],
        "episode_counts": bundle.metadata["episode_counts"],
        "excluded_counts": bundle.metadata["excluded_counts"],
        "event_manifest_sha256": sha256_file(output / "event_manifest.json"),
    })
    print("Data preparation only: no model, checkpoint weights, MPC solve or training.", flush=True)
    for split in ("train", "val"):
        lower, upper = bundle.metadata["split_intervals"][split]
        episodes = bundle.episodes[split]
        print(f"{split}: [{lower}, {upper}) | {len(episodes)} episodes | "
              f"{sum(episode.steps for episode in episodes)} steps", flush=True)
    for row in bundle.metadata["monthly_coverage"]:
        print(f"  {row['month']} {row['split']}: {row['included']} included / "
              f"{row['excluded']} excluded / {row['requested']} templates / "
              f"{row['steps']} steps", flush=True)
    print(f"Prepared event tables: {output}", flush=True)
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    origin = result.add_mutually_exclusive_group()
    origin.add_argument("--metadata", type=Path)
    origin.add_argument("--forecast-data-dir", type=Path,
                        help="Prepare episode tables before a new M0 checkpoint exists; no weights needed.")
    result.add_argument("--scenario-config", type=Path,
                        default=ROOT / "configs/m0_s0_s5_v2.json")
    result.add_argument("--dispatch", type=Path,
                        default=DEFAULT_DISPATCH_ROOT / "dispatch.csv")
    result.add_argument("--load-tiers", type=Path,
                        default=DEFAULT_DISPATCH_ROOT / "load_tiers.csv")
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    main(parser().parse_args())

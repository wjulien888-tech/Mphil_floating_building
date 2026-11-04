"""Freeze common-origin PV-only 96/192/288-to-96 data without running a model.

The original constructed PV values, quality flags and split labels are retained.
Only sample windows and the derived forecast_origin_valid flag change. A common
288-step history guard and complete 96-step targets make all three lookbacks
comparable at the same forecast origins.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "external/ol-ems-hkust"))
from configurations.hkust_15min import SplitWindow
from model_variants._pv_only_dataset import PVForecastConfig, PVOnlyDataset


DEFAULT_SOURCE = ROOT / "output/frozen/pv_only_2021_2023_96to1_v1_20260920"
DEFAULT_OUTPUT = ROOT / "output/frozen/pv_only_2021_2023_multihorizon96_v1_20260921"
LOOKBACKS = (96, 192, 288)
HORIZON = 96
ANCHOR_LOOKBACK = 288


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, relative_to: Path | None = None) -> dict:
    return {"path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
            "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_source(source: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest.get("outputs", [])
               if record["path"] == "pv_timeseries.csv"]
    if len(records) != 1:
        raise ValueError("Source manifest must identify exactly one pv_timeseries.csv")
    actual = file_record(source / "pv_timeseries.csv")
    if any(actual[key] != records[0][key] for key in ("sha256", "size_bytes")):
        raise ValueError("Source PV CSV differs from its frozen manifest")
    if not manifest.get("provenance"):
        raise ValueError("Source manifest must explain the constructed PV provenance")
    return manifest


def make_datasets(source: Path, source_manifest: dict, lookbacks=LOOKBACKS,
                  anchor_lookback=ANCHOR_LOOKBACK) -> dict[int, PVOnlyDataset]:
    splits = {name: SplitWindow(record["start_inclusive"], record["end_exclusive"])
              for name, record in source_manifest["splits"].items()}
    datasets = {}
    for lookback in lookbacks:
        config = PVForecastConfig(
            data_path=source / "pv_timeseries.csv", lookback_steps=lookback,
            prediction_horizon=HORIZON, anchor_lookback_steps=anchor_lookback,
            split_windows=splits,
        )
        datasets[lookback] = PVOnlyDataset(config)
    for name in splits:
        reference = datasets[lookbacks[0]].for_split(name)._anchors
        if any(not np.array_equal(dataset.for_split(name)._anchors, reference)
               for dataset in datasets.values()):
            raise AssertionError(f"Lookbacks do not share exactly the same {name} forecast origins")
    return datasets


def export_arrays(view: PVOnlyDataset) -> dict[str, np.ndarray]:
    """Export the same tensors that forecast_item serves, without model execution."""
    anchors = view._anchors
    raw = view._prepared.raw.numpy()[:, 0]
    normalised = view._prepared.normalised.numpy()[:, 0]
    timestamps = view._prepared.unix_seconds.numpy()
    lookback, horizon = view.lookback_window, view.prediction_horizon
    return {
        "X": np.lib.stride_tricks.sliding_window_view(normalised, lookback)[anchors - lookback, :, None],
        "y_kw": np.lib.stride_tricks.sliding_window_view(raw, horizon)[anchors, :, None],
        "y_scaled": np.lib.stride_tricks.sliding_window_view(normalised, horizon)[anchors, :, None],
        "anchor_indices": anchors.copy(),
        "target_timestamps": timestamps[anchors].copy(),
        "future_timestamps": np.lib.stride_tricks.sliding_window_view(timestamps, horizon)[anchors].copy(),
    }


def sample_index(view: PVOnlyDataset) -> pd.DataFrame:
    anchors = view._anchors
    times = view._prepared.frame.datetime
    return pd.DataFrame({
        "split": view.split, "sample_index": np.arange(len(anchors)), "anchor_index": anchors,
        "history_start": times.iloc[anchors - view.lookback_window].to_numpy(),
        "history_end": times.iloc[anchors - 1].to_numpy(),
        "target_start": times.iloc[anchors].to_numpy(),
        "target_end": times.iloc[anchors + view.prediction_horizon - 1].to_numpy(),
    })


def _scaler(dataset: PVOnlyDataset) -> dict:
    window = dataset.config.split_windows["train"]
    return {
        "pv_min": dataset.pv_min, "pv_max": dataset.pv_max,
        "scale": dataset.pv_max - dataset.pv_min or 1.0,
        "dtype": "float32", "feature_names": ["pv"],
        "fit_start_inclusive": window.start, "fit_end_exclusive": window.end,
        "fit_valid_rows": dataset._prepared.quality["scaler_valid_training_rows"],
        "fit_policy": "training interval AND history_valid=1", "clip": False,
        "target_in_model_loss": "physical kW",
    }


def _preview(dataset: PVOnlyDataset) -> pd.DataFrame:
    rows = []
    for name, view in dataset.split_views().items():
        anchor = int(view._anchors[0])
        for role, start, count in (("history", anchor - view.lookback_window, view.lookback_window),
                                   ("target", anchor, view.prediction_horizon)):
            # Short excerpt only; sample_index and NPZ contain the full windows.
            for offset in sorted(set(range(min(4, count))) | set(range(max(0, count - 4), count))):
                row = start + offset
                rows.append({"split": name, "sample_index": 0, "role": role, "step": offset + 1,
                             "datetime": view._prepared.frame.datetime.iloc[row],
                             "pv_kW": float(view._prepared.raw[row, 0]),
                             "pv_scaled": float(view._prepared.normalised[row, 0])})
    return pd.DataFrame(rows)


def copy_with_updated_origins(source: Path, destination: Path, dataset: PVOnlyDataset) -> None:
    """Retain every original field's text except the derived sample-origin flag."""
    valid_origins = set(np.concatenate(list(dataset._prepared.anchors.values())).tolist())
    with source.open(newline="", encoding="utf-8") as input_stream, destination.open(
        "w", newline="", encoding="utf-8"
    ) as output_stream:
        reader = csv.DictReader(input_stream)
        fields = list(reader.fieldnames or [])
        if "forecast_origin_valid" not in fields:
            fields.append("forecast_origin_valid")
        writer = csv.DictWriter(output_stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(reader):
            row["forecast_origin_valid"] = str(int(index in valid_origins))
            writer.writerow(row)


def freeze(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT, *,
           lookbacks=LOOKBACKS, anchor_lookback=ANCHOR_LOOKBACK) -> dict:
    """Create a new immutable-by-convention artifact; refuse an existing destination."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen dataset: {output}")
    source_manifest = read_source(source)
    source_records = [file_record(source / "manifest.json"), file_record(source / "pv_timeseries.csv")]
    code_paths = [Path(__file__).resolve(), ROOT / "model_variants/_pv_only_dataset.py"]
    code_records = [file_record(path, ROOT) for path in code_paths]
    if not lookbacks or len(set(lookbacks)) != len(lookbacks) or anchor_lookback < max(lookbacks):
        raise ValueError("Common anchor history must cover all unique requested lookbacks")
    datasets = make_datasets(source, source_manifest, lookbacks, anchor_lookback)
    created = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        artifacts = []
        for lookback, dataset in datasets.items():
            destination = staging / f"lookback_{lookback}"
            destination.mkdir()
            copy_with_updated_origins(source / "pv_timeseries.csv", destination / "pv_timeseries.csv", dataset)
            snapshots = destination / "source_snapshot"
            snapshots.mkdir()
            for path in code_paths:
                shutil.copy2(path, snapshots / path.name)
            shutil.copy2(source / "manifest.json", destination / "source_manifest.json")
            scaler = _scaler(dataset)
            write_json(destination / "scaler.json", scaler)
            indices, splits = [], {}
            for name, view in dataset.split_views().items():
                arrays = export_arrays(view)
                np.savez_compressed(destination / f"{name}.npz", **arrays)
                index = sample_index(view)
                indices.append(index)
                frame = view._prepared.frame
                rows = frame[frame.split == name]
                window = dataset.config.split_windows[name]
                before = view._prepared.quality["samples_before_quality_filter"][name]
                splits[name] = {
                    "start_inclusive": window.start, "end_exclusive": window.end,
                    "rows": len(rows), "history_valid_rows": int(rows.history_valid.sum()),
                    "pv_target_valid_rows": int(rows.pv_target_valid.sum()),
                    "candidate_windows": before, "valid_windows": len(view),
                    "rejected_windows": before - len(view),
                    "split_boundary_rejected_windows": view._prepared.quality["samples_rejected_by_split_boundary"][name],
                    "first_target": str(index.target_start.iloc[0]),
                    "last_target": str(index.target_start.iloc[-1]),
                    "last_target_end": str(index.target_end.iloc[-1]),
                    "X_shape": list(arrays["X"].shape), "y_shape": list(arrays["y_kw"].shape),
                    "target_scaled_above_one": int((arrays["y_scaled"] > 1).sum()),
                    "anchor_indices_sha256": hashlib.sha256(view._anchors.astype("<i8").tobytes()).hexdigest(),
                }
                del arrays
            pd.concat(indices, ignore_index=True).to_csv(destination / "sample_index.csv", index=False)
            pd.DataFrame(splits).T.to_csv(destination / "split_summary.csv", index_label="split")
            _preview(dataset).to_csv(destination / "sample_preview.csv", index=False)
            manifest = {
                "schema_version": 3, "artifact_type": "pv_only_forecast_multistep",
                "dataset_version": f"{output.name}/lookback_{lookback}", "created_utc": created,
                "frequency": "15min", "timezone": dataset.config.timezone,
                "rows": len(dataset._prepared.frame),
                "parameters": {"lookback_steps": lookback, "prediction_horizon": HORIZON,
                               "anchor_lookback_steps": anchor_lookback,
                               "history_features": ["pv"], "target_feature": "pv",
                               "source_column": "pv_kW", "pv_values_unchanged": True,
                               "forecast_origin_valid_recomputed": True,
                               "building_parameters_unchanged": True},
                "splits": splits, "scaler": scaler, "quality": dict(dataset._prepared.quality),
                "inputs": source_records, "source_dataset_version": source_manifest.get("dataset_version"),
                "source_manifest_sha256": source_records[0]["sha256"],
                "source_pv_timeseries_sha256": source_records[1]["sha256"],
                "provenance": source_manifest["provenance"], "code": code_records,
                "pv_input_provenance": source_manifest.get("pv_input_provenance", dataset.config.pv_input_provenance),
                "planning_tail_end_exclusive": source_manifest.get("planning_tail_end_exclusive"),
                "preparation": (
                    "Preserve source CSV fields verbatim except forecast_origin_valid, which is recomputed "
                    f"for shared {anchor_lookback}-step valid history and 96 valid targets within one split; rebuild sample windows."
                ),
                "sample_policy": dataset._prepared.quality["sample_policy"],
                "anchor_hash_policy": "SHA256 of ordered anchor row indices as little-endian int64 bytes",
                "rolling_policy": "Forecast at every valid 15-minute origin; horizon includes origin through origin+95 steps.",
                "split_policy": "All 96 target timestamps must belong to the same split; historical inputs may precede that split.",
                "scaler_policy": "Fit only history_valid PV rows in the original training interval; no held-out clipping.",
                "validation": "Requested lookbacks share forecast origins and target windows; preparation executes no model.",
            }
            manifest["outputs"] = [file_record(path, destination) for path in sorted(destination.rglob("*"))
                                   if path.is_file()]
            write_json(destination / "manifest.json", manifest)
            artifacts.append({"lookback_steps": lookback, "prediction_horizon": HORIZON,
                              "path": destination.name,
                              "split_samples": {name: item["valid_windows"] for name, item in splits.items()},
                              "manifest_sha256": sha256(destination / "manifest.json")})
        suite = {
            "schema_version": 3, "artifact_type": "pv_only_forecast_multistep_suite",
            "dataset_version": output.name, "created_utc": created,
            "lookback_steps": list(lookbacks), "prediction_horizon": HORIZON,
            "anchor_lookback_steps": anchor_lookback, "common_forecast_origins": True,
            "source": source_records, "provenance": source_manifest["provenance"],
            "datasets": artifacts, "code": code_records,
        }
        write_json(staging / "manifest.json", suite)
        for record in source_records:
            if file_record(Path(record["path"])) != record:
                raise RuntimeError(f"Source changed during preparation: {record['path']}")
        for record in code_records:
            if file_record(ROOT / record["path"], ROOT) != record:
                raise RuntimeError(f"Preparation code changed while freezing data: {record['path']}")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite frozen dataset: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    suite = freeze(args.source, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "datasets": suite["datasets"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

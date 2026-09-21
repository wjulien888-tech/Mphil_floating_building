"""Shared M0 runner for the official entry and both recurrent model scripts.

Use external/ol-ems-hkust/main.py --model_type lstm|gru, or either standalone
model script, for the same 96-to-1 forecast training/evaluation implementation.
These variants do not generate the 96 future values required by the old MPC.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Mapping

VARIANT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VARIANT_ROOT.parent
OL_ROOT = PROJECT_ROOT / "external/ol-ems-hkust"
if str(OL_ROOT) not in sys.path:
    sys.path.insert(0, str(OL_ROOT))

import torch
import pandas as pd

from configurations.hkust_15min import HKUSTConfig
from _m0_variant_dataset import DEFAULT_CONFIG as LEGACY_CONFIG, CSVDataset
from _pv_only_dataset import DEFAULT_CONFIG, PVForecastConfig, PVOnlyDataset
from src.models.controller import Controller
from src.training_and_evaluation.m0_training import (
    code_fingerprint, load_checkpoint, runtime_versions, seed_run, sha256_file, train_m0,
)
from _m0_variant_evaluation import evaluate_m0


LOGGER = logging.getLogger("hkust_ol_ems")


def _model_architecture(config: HKUSTConfig | PVForecastConfig, dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "architecture_id": f"hkust_m0_{args.recurrent_type.lower()}_96to1_mlp_v2",
        "recurrent_type": args.recurrent_type,
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "input_dim": dataset.input_dim, "dropout": args.dropout,
        "history_features": list(dataset.input_features),
        "fc_hidden_size": args.fc_hidden_size,
        "lookback_steps": config.lookback_steps,
        "prediction_horizon": config.prediction_horizon,
        "output_activation": "sigmoid_train_pv_range",
    }


def _variant_code_fingerprint() -> dict[str, str]:
    hashes = {f"external/ol-ems-hkust/{name}": digest
              for name, digest in code_fingerprint(OL_ROOT).items()}
    for path in sorted(VARIANT_ROOT.glob("*.py")):
        hashes[f"model_variants/{path.name}"] = sha256_file(path)
    return hashes


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _experiment_path(args: argparse.Namespace) -> Path:
    name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(args.save_dir).expanduser().resolve() / name
    if path.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {path}. "
            "Choose another --experiment_name so results from two runs cannot mix."
        )
    path.mkdir(parents=True, exist_ok=False)
    return path


def _configure_logging(experiment_path: Path, *, resume: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                experiment_path / "run.log", mode="a" if resume else "x", encoding="utf-8"
            ),
        ],
        force=True,
    )


def _runtime_config(args: argparse.Namespace) -> HKUSTConfig | PVForecastConfig:
    if args.data_format == "legacy_dispatch":
        if Path(args.data_path) == DEFAULT_CONFIG.data_path:
            args.data_path = str(LEGACY_CONFIG.data_path)
        if args.data_manifest == DEFAULT_CONFIG.data_path.parent / "manifest.json":
            args.data_manifest = LEGACY_CONFIG.data_path.parent / "manifest.json"
        args.load_tiers_path = args.load_tiers_path or str(LEGACY_CONFIG.load_tiers_path)
        return replace(LEGACY_CONFIG, data_path=Path(args.data_path).expanduser().resolve(),
                       load_tiers_path=Path(args.load_tiers_path).expanduser().resolve())
    if args.load_tiers_path:
        raise ValueError("PV-only input does not accept --load_tiers_path")
    return replace(DEFAULT_CONFIG, data_path=Path(args.data_path).expanduser().resolve())


class _ForecastOnlyLayer(torch.nn.Module):
    """No optimization variables or weights; compatible Controller state keys."""
    def __init__(self, prediction_horizon: int) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("The 96-to-1 variants are forecast-only and do not solve MPC")


def _build_controller(
    config: HKUSTConfig | PVForecastConfig, dataset: Any, args: argparse.Namespace,
    controller_class: type[Controller],
) -> Controller:
    checkpoint: Mapping[str, Any] | None = None
    if args.load_model is not None:
        loaded = load_checkpoint(args.load_model)
        if not isinstance(loaded, Mapping) or "controller_state_dict" not in loaded:
            raise ValueError(
                "--load_model must point to a complete best.pt/last.pt produced "
                "by the matching variant, including dataset_scaler and config metadata."
            )
        checkpoint = loaded
        if "dataset_scaler" not in checkpoint or "config" not in checkpoint:
            raise ValueError(
                "Checkpoint is missing dataset_scaler/config; refusing a silent "
                "forecast scaling or configuration mismatch."
            )
        saved_config = checkpoint["config"]
        if not isinstance(saved_config, Mapping):
            raise ValueError("Checkpoint config must be a mapping.")
        required_equal_fields = (
            "split_windows",
            "frequency",
            "history_features",
            "target_feature",
            "lookback_steps",
            "prediction_horizon",
            "source_columns",
            "dt_hours",
        )
        if isinstance(config, HKUSTConfig):
            required_equal_fields += (
                "battery_capacity_kWh",
                "battery_power_kW",
                "roundtrip_efficiency",
                "allow_grid_charging",
                "min_soc_fraction",
                "max_soc_fraction",
                "initial_soc_fraction",
                "self_discharge_per_step",
                "grid_import_limit_kW",
                "grid_export_limit_kW",
                "no_arbitrage_margin_CNY_per_kWh",
                "objective_weights",
            )
        else:
            required_equal_fields += ("timezone", "require_quality_flags", "pv_input_provenance")
        current_config = asdict(config)
        missing_fields = [
            field for field in required_equal_fields if field not in saved_config
        ]
        if missing_fields:
            raise ValueError(
                "Checkpoint config is missing required reproducibility fields: "
                + ", ".join(missing_fields)
            )
        mismatches = [
            field
            for field in required_equal_fields
            if field in saved_config and saved_config[field] != current_config[field]
        ]
        if mismatches:
            raise ValueError(
                "Checkpoint and current data/model configuration differ for: "
                + ", ".join(mismatches)
            )
        saved_architecture = checkpoint.get("model_architecture")
        expected_architecture = _model_architecture(config, dataset, args)
        if saved_architecture != expected_architecture:
            raise ValueError(
                "Checkpoint model architecture does not match current arguments: "
                f"saved={saved_architecture}, current={expected_architecture}"
            )
        dataset.load_state_dict(checkpoint["dataset_scaler"])

    if args.mode not in {"train", "evaluate"}:
        raise ValueError("The 96-to-1 variants support forecast training/evaluation only")
    optimisation_layer = _ForecastOnlyLayer(config.prediction_horizon)
    controller = controller_class(
        optimisation_layer=optimisation_layer,
        dataset=dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        fc_hidden_size=args.fc_hidden_size,
    )
    if checkpoint is not None:
        controller.load_state_dict(checkpoint["controller_state_dict"])
    return controller


def _dataset_identity(args: argparse.Namespace) -> dict[str, Any]:
    pv_only = args.data_format == "pv_only"
    identity = {
        "data_format": args.data_format,
        "pv_timeseries_sha256" if pv_only else "dispatch_sha256": sha256_file(args.data_path),
        "manifest_sha256": sha256_file(args.data_manifest) if args.data_manifest else None,
    }
    if not pv_only:
        identity["load_tiers_sha256"] = sha256_file(args.load_tiers_path)
    if args.data_manifest:
        # Parse now so corrupt/incomplete JSON cannot be silently archived.
        manifest = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Data manifest must contain a JSON object")
        # The frozen snapshot directory is an inseparable artifact; do not mix
        # a manifest with similarly named files from another location.
        directory = Path(args.data_manifest).resolve().parent
        required_paths = [Path(args.data_path).resolve()]
        if not pv_only:
            required_paths.append(Path(args.load_tiers_path).resolve())
        if any(path.parent != directory for path in required_paths):
            raise ValueError("Data and manifest must come from the same frozen directory")
        schema_version = 2 if pv_only else 1
        if manifest.get("schema_version") != schema_version or not isinstance(manifest.get("outputs"), list):
            raise ValueError(f"Unsupported frozen manifest schema; expected schema_version={schema_version} and outputs list")
        if pv_only:
            if manifest.get("artifact_type") != "pv_only_forecast_96to1":
                raise ValueError("Expected artifact_type=pv_only_forecast_96to1")
            parameters = manifest.get("parameters", {})
            expected_parameters = {"lookback_steps": 96, "prediction_horizon": 1,
                                   "history_features": ["pv"], "target_feature": "pv"}
            if any(parameters.get(key) != value for key, value in expected_parameters.items()):
                raise ValueError("Frozen manifest task/features do not match the PV-only 96-to-1 task")
            splits = manifest.get("splits", {})
            if set(splits) != set(DEFAULT_CONFIG.split_windows):
                raise ValueError("Frozen manifest must declare train, val and test splits")
            for name, window in DEFAULT_CONFIG.split_windows.items():
                declared = splits[name]
                if (pd.Timestamp(declared.get("start_inclusive")) != pd.Timestamp(window.start)
                        or pd.Timestamp(declared.get("end_exclusive")) != pd.Timestamp(window.end)):
                    raise ValueError(f"Frozen manifest {name} dates differ from the configured split")
            identity["task"] = expected_parameters
            identity["splits"] = splits
        recorded = {}
        for record in manifest["outputs"]:
            path = (directory / record["path"]).resolve()
            if not path.is_relative_to(directory) or path == directory or path in recorded:
                raise ValueError("Invalid or duplicate frozen output path")
            recorded[path] = record
            if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Frozen output differs from manifest: {path.name}")
        if any(path not in recorded for path in required_paths):
            raise ValueError("Model input files are not declared frozen manifest outputs")
    return identity


def main(args: argparse.Namespace, controller_class: type[Controller]) -> Path:
    if args.recurrent_type != controller_class.recurrent_type:
        raise ValueError("Entry point and controller recurrent types differ")
    if args.mode not in {"train", "evaluate"}:
        raise ValueError("96-to-1 forecasts cannot directly supply the old 24-hour MPC")
    if args.gpus:
        args.device = "cuda"
    if args.mode == "evaluate" and not args.load_model:
        raise ValueError("--mode evaluate requires --load_model")
    if args.mode == "evaluate" and args.resume:
        raise ValueError("--resume applies only to train")
    if args.mode == "train" and args.load_model and not args.resume:
        raise ValueError("M0 train starts from seeded weights; use --resume last.pt to continue")
    if args.resume:
        if args.load_model and Path(args.load_model).resolve() != Path(args.resume).resolve():
            raise ValueError("--load_model and --resume disagree")
        args.load_model = args.resume
    smoke = bool(args.fast_dev_run or args.limit_train_batches != 1.0 or args.limit_val_batches != 1.0)
    if args.mode == "train" and not smoke and not args.data_manifest:
        raise ValueError("Formal M0 training requires --data_manifest for the frozen snapshot")
    seed_run(args.seed, args.torch_num_threads)
    config = _runtime_config(args)
    identity = _dataset_identity(args)
    dataset_class = PVOnlyDataset if args.data_format == "pv_only" else CSVDataset
    train_dataset = dataset_class(config=config, split="train")
    checkpoint = load_checkpoint(args.load_model) if args.load_model else None
    if checkpoint is not None and checkpoint.get("dataset_identity") != identity:
        raise ValueError("Checkpoint does not match frozen data/manifest; evaluation/resume refused")
    controller = _build_controller(config, train_dataset, args, controller_class)
    # Build views after restoring the exact training scaler.
    experiment_path = Path(args.resume).resolve().parent if args.resume else _experiment_path(args)
    _configure_logging(experiment_path, resume=bool(args.resume))
    dataset_summary = train_dataset.validation_summary()
    LOGGER.info("M0 %s | data %s", args.mode, identity)
    LOGGER.info("Dataset: %s", dataset_summary)
    code_hashes = _variant_code_fingerprint()
    shared = {
        "config": asdict(config), "dataset_scaler": train_dataset.state_dict(),
        "model_architecture": _model_architecture(config, train_dataset, args),
        "dataset_identity": identity, "code_sha256": code_hashes, "seed": args.seed,
    }
    metadata = {"arguments": vars(args), "runtime": runtime_versions(), "config": asdict(config),
                "dataset": dataset_summary, "dataset_identity": identity,
                "model_architecture": _model_architecture(config, train_dataset, args),
                "code_sha256": code_hashes, "seed": args.seed,
                "trainable_parameters": sum(p.numel() for p in controller.parameters() if p.requires_grad)}
    if args.mode == "train":
        train_m0(controller, train_dataset, train_dataset.for_split("val"), experiment_path,
                 args, shared, metadata)
    else:
        evaluation_metadata = {**metadata, "checkpoint": str(Path(args.load_model).resolve()),
                               "checkpoint_sha256": sha256_file(args.load_model),
                               "checkpoint_epoch": checkpoint.get("epoch"),
                               "checkpoint_seed": checkpoint.get("seed"),
                               "checkpoint_code_sha256": checkpoint.get("code_sha256")}
        evaluate_m0(controller, train_dataset.for_split(args.eval_split), experiment_path,
                    args, evaluation_metadata)
    LOGGER.info("Experiment saved to %s", experiment_path)
    return experiment_path


def build_parser(recurrent_type: str) -> argparse.ArgumentParser:
    if recurrent_type not in {"LSTM", "GRU"}:
        raise ValueError("recurrent_type must be LSTM or GRU")
    parser = argparse.ArgumentParser(
        description=f"Two-layer {recurrent_type}: 96 historical PV points to the next 15-minute PV value, frozen 2021-2023 data"
    )
    parser.set_defaults(recurrent_type=recurrent_type)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    data = parser.add_argument_group("data")
    data.add_argument("--data_format", choices=("pv_only", "legacy_dispatch"), default="pv_only",
                      help="Default: one PV feature and the frozen 2021-2023 forecast dataset.")
    data.add_argument("--data_path", default=str(DEFAULT_CONFIG.data_path))
    data.add_argument("--load_tiers_path", default=None, help="Only for legacy_dispatch compatibility.")
    data.add_argument("--num_workers", type=int, default=0)
    data.add_argument("--data_manifest", type=Path, default=DEFAULT_CONFIG.data_path.parent / "manifest.json")

    model = parser.add_argument_group("model")
    model.add_argument("--hidden_dim", type=int, default=64)
    model.add_argument("--num_layers", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.2)
    model.add_argument("--fc_hidden_size", type=int, default=5)
    model.add_argument("--learning_rate", type=float, default=1e-3)
    model.add_argument(
        "--weight_decay",
        "--weight_decay_end_weight",
        dest="weight_decay",
        type=float,
        default=1e-6,
        help="Adam weight decay; the old miswired end-weight name is a compatible alias.",
    )
    model.add_argument("--load_model", type=Path, default=None)

    training = parser.add_argument_group("training")
    training.add_argument("--max_epochs", type=int, default=50)
    training.add_argument("--train_batch_size", type=int, default=64)
    training.add_argument("--val_batch_size", type=int, default=64)
    training.add_argument("--limit_train_batches", type=float, default=1.0)
    training.add_argument("--limit_val_batches", type=float, default=1.0)
    training.add_argument("--gpus", type=int, default=0)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--torch_num_threads", type=int, default=1)
    training.add_argument("--patience", type=int, default=8)
    training.add_argument("--min_delta", type=float, default=0.01)
    training.add_argument("--resume", type=Path, default=None)
    training.add_argument(
        "--fast_dev_run",
        nargs="?",
        const=1,
        type=int,
        default=0,
        help="Run only N train/validation batches (default N=1 when flag is present).",
    )

    evaluation = parser.add_argument_group("independent forecast evaluation")
    evaluation.add_argument("--eval_split", choices=("val", "test"), default="test")
    evaluation.add_argument("--eval_limit_batches", type=float, default=1.0)
    evaluation.add_argument("--daylight_threshold_kw", type=float, default=0.1)

    output = parser.add_argument_group("experiment")
    output.add_argument("--seed", type=int, default=42)
    output.add_argument("--save_dir", default=str(PROJECT_ROOT / "experiments" / f"m0_{recurrent_type.lower()}_2layer_pvonly_2021_2023_96to1"))
    output.add_argument("--experiment_name", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit("Run lstm_2layer_m0.py or gru_2layer_m0.py instead.")

"""M0 entry point: two-layer LSTM/GRU, PV-only, 96-to-1 forecasting.

Training never executes MPC or evaluates test forecasts. Use --mode evaluate
with a selected best.pt to evaluate frozen weights on validation/test data.
The explicit --mode closed-loop path retains the historical multi-step MPC
contract; the current one-step checkpoints cannot supply its 96-step horizon.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import logging
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from configurations.hkust_15min import DEFAULT_CONFIG, HKUSTConfig
from src.dataloader.csv_dataset import CSVDataset
from src.models.controller import Controller, OptimisationLayer
from src.plants.plants import MicroGridSystem
from src.plants.pymgrid_adapter import PymgridAdapter
from src.training_and_evaluation.learner import OnlineLearner
from src.training_and_evaluation.m0_training import (
    code_fingerprint, load_checkpoint, runtime_versions, seed_run, sha256_file, train_m0,
)
from src.training_and_evaluation.m0_evaluation import evaluate_m0


LOGGER = logging.getLogger("hkust_ol_ems")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANT_ROOT = PROJECT_ROOT / "model_variants"
if str(VARIANT_ROOT) not in sys.path:
    sys.path.insert(0, str(VARIANT_ROOT))
import _m0_variant_runner as _variant_runner
from lstm_2layer_m0 import PVLSTMController
from gru_2layer_m0 import PVGRUController


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


def _runtime_config(args: argparse.Namespace) -> HKUSTConfig:
    return replace(
        DEFAULT_CONFIG,
        data_path=Path(args.data_path).expanduser().resolve(),
        load_tiers_path=Path(args.load_tiers_path).expanduser().resolve(),
    )


class _ForecastOnlyLayer(torch.nn.Module):
    """No optimization variables or weights; compatible Controller state keys."""
    def __init__(self, prediction_horizon: int) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Forecast-only mode cannot solve MPC; use --mode closed-loop")


def _build_controller(
    config: HKUSTConfig, dataset: CSVDataset, args: argparse.Namespace
) -> Controller:
    checkpoint: Mapping[str, Any] | None = None
    if args.load_model is not None:
        loaded = load_checkpoint(args.load_model)
        if not isinstance(loaded, Mapping) or "controller_state_dict" not in loaded:
            raise ValueError(
                "--load_model must point to the complete controller.pt produced "
                "by this main.py, including dataset_scaler and config metadata."
            )
        checkpoint = loaded
        if "dataset_scaler" not in checkpoint or "config" not in checkpoint:
            raise ValueError(
                "Checkpoint is missing dataset_scaler/config; refusing a silent "
                "forecast scaling or MPC configuration mismatch."
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
        expected_architecture = {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "input_dim": dataset.input_dim,
        }
        if saved_architecture != expected_architecture:
            raise ValueError(
                "Checkpoint model architecture does not match current arguments: "
                f"saved={saved_architecture}, current={expected_architecture}"
            )
        dataset.load_state_dict(checkpoint["dataset_scaler"])

    if getattr(args, "mode", "closed-loop") in ("train", "evaluate"):
        optimisation_layer = _ForecastOnlyLayer(config.prediction_horizon)
    else:
        prediction_model = MicroGridSystem(
            self_discharge=config.self_discharge_per_step,
            grid_power_min_max=config.grid_power_min_max_kW,
            sample_time=config.dt_hours,
            battery_capacity_kwh=config.battery_capacity_kWh,
            battery_power_kw=config.battery_power_kW,
            charge_efficiency=config.charge_efficiency,
            discharge_efficiency=config.discharge_efficiency,
            min_soc_fraction=config.min_soc_fraction,
            max_soc_fraction=config.max_soc_fraction,
            initial_soc_fraction=config.initial_soc_fraction,
        )
        if prediction_model.manages_physical_soc:
            raise RuntimeError("MPC prediction model must not own realised SOC.")
        A_matrix, B_matrix = prediction_model.get_model_matrices()
        optimisation_layer = OptimisationLayer(
            prediction_horizon=config.prediction_horizon,
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            charge_min_max=prediction_model.soc_min_max_kwh,
            storage_power_min_max=prediction_model.storage_power_min_max_kw,
            grid_power_min_max=prediction_model.grid_power_min_max_kw,
            sample_time=config.dt_hours,
            charge_efficiency=config.charge_efficiency,
            discharge_efficiency=config.discharge_efficiency,
            self_discharge=config.self_discharge_per_step,
            allow_grid_charging=config.allow_grid_charging,
            no_arbitrage_margin=config.no_arbitrage_margin_CNY_per_kWh,
            objective_weights=config.objective_weights,
        )
    controller = Controller(
        optimisation_layer=optimisation_layer,
        dataset=dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    if checkpoint is not None:
        controller.load_state_dict(checkpoint["controller_state_dict"])
    return controller


def _build_adapter(config: HKUSTConfig) -> PymgridAdapter:
    return PymgridAdapter(
        dt_hours=config.dt_hours,
        battery_capacity_kwh=config.battery_capacity_kWh,
        battery_power_kw=config.battery_power_kW,
        # Pymgrid requires one-way efficiency; the config derives sqrt(0.92).
        efficiency=config.charge_efficiency,
        initial_soc=config.initial_soc_fraction,
        soc_min=config.min_soc_fraction,
        soc_max=config.max_soc_fraction,
        grid_import_limit_kw=config.grid_import_limit_kW,
        grid_export_limit_kw=config.grid_export_limit_kW,
        battery_cost_per_kwh=config.objective_weights["battery_cycle"],
        loss_load_cost_per_kwh=config.objective_weights["noncritical_unmet"],
        critical_loss_load_cost_per_kwh=config.objective_weights["critical_unmet"],
        noncritical_loss_load_cost_per_kwh=config.objective_weights[
            "noncritical_unmet"
        ],
        pv_curtailment_cost_per_kwh=config.objective_weights["pv_curtailment"],
    )


def _dataloader(
    dataset: CSVDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    use_gpu: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_gpu,
    )


def _closed_loop_main(args: argparse.Namespace) -> Path:
    """Train, execute the rolling day, and save both plans and reality."""
    if args.skip_training and args.load_model is None and not args.allow_untrained:
        raise ValueError(
            "--skip_training requires --load_model. For an explicit code-only "
            "smoke test with random weights, also pass --allow_untrained."
        )
    if not args.no_plots:
        if args.plot_dpi <= 0:
            raise ValueError("--plot_dpi must be positive.")
        if not 0 <= args.plot_plan_step < args.rollout_steps:
            raise ValueError(
                "--plot_plan_step must identify one executed decision in "
                "[0, rollout_steps)."
            )
    pl.seed_everything(args.seed, workers=True)
    config = _runtime_config(args)
    train_dataset = CSVDataset(config=config, split="train")
    controller = _build_controller(config, train_dataset, args)
    # Split views are created after checkpoint scaler restoration so all three
    # use the exact same training-fitted transform.
    val_dataset = train_dataset.for_split("val")
    test_dataset = train_dataset.for_split("test")
    dataset_summary = train_dataset.validation_summary()

    experiment_path = _experiment_path(args)
    _configure_logging(experiment_path)
    if args.load_model is not None:
        LOGGER.info("Loaded controller state from %s", args.load_model)
    LOGGER.info("Validated dataset: %s", dataset_summary)
    for column, provenance in train_dataset.generated_columns.items():
        LOGGER.warning("%s is scenario/generated input: %s", column, provenance)
    if dataset_summary["outage_rows"] == 0:
        LOGGER.warning(
            "grid_available never falls below 1; this run validates the grid-connected "
            "closed loop, not an outage-resilience scenario."
        )

    adapter = _build_adapter(config)
    learner = OnlineLearner(
        controller=controller,
        adapter=adapter,
        learning_rate=args.learning_rate,
        # This is the actual optimiser value. The old end-weight wiring is gone.
        weight_decay=args.weight_decay,
        initial_soc_kwh=config.initial_soc_kWh,
    )

    training_metrics: dict[str, float] = {}
    if args.load_model is not None:
        LOGGER.info("Frozen checkpoint; closed-loop evaluation only.")
    else:
        LOGGER.warning("UNTRAINED SMOKE TEST: metrics are not research results.")

    # CvxpyLayer and the external simulator are evaluated sequentially on CPU.
    learner.to(torch.device("cpu"))
    results = learner.run_closed_loop(
        test_dataset,
        steps=args.rollout_steps,
        start_index=args.rollout_start_index,
    )
    learner.save_closed_loop(results, experiment_path)
    plot_paths: Mapping[str, Path] = {}
    if not args.no_plots:
        # Import lazily so data/training-only use remains independent of the
        # Matplotlib backend. The plotting module never opens a GUI window.
        from src.plots import plot_closed_loop_results

        plot_paths = plot_closed_loop_results(
            results["actual"],
            results["plans"],
            experiment_path / "plots",
            dt_hours=config.dt_hours,
            currency="CNY",
            plan_decision_step=args.plot_plan_step,
            dpi=args.plot_dpi,
        )
        LOGGER.info("Plots saved: %s", plot_paths)

    checkpoint = {
        "format_version": 1,
        "controller_state_dict": controller.state_dict(),
        "dataset_scaler": train_dataset.state_dict(),
        "config": asdict(config),
        "model_architecture": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "input_dim": train_dataset.input_dim,
        },
    }
    torch.save(checkpoint, experiment_path / "controller.pt")
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "arguments": vars(args),
        "config": asdict(config),
        "dataset": dataset_summary,
        "training_metrics": training_metrics,
        "plots": plot_paths,
        "model_status": (
            "untrained_smoke"
            if args.skip_training and args.load_model is None
            else "loaded_checkpoint"
            if args.skip_training
            else "fast_dev_trained"
            if args.fast_dev_run
            else "trained"
        ),
        "closed_loop_summary": results["summary"],
    }
    _write_json(experiment_path / "run_metadata.json", metadata)
    _write_json(experiment_path / "closed_loop_summary.json", results["summary"])
    LOGGER.info("Closed-loop summary: %s", results["summary"])
    LOGGER.info("Experiment saved to %s", experiment_path)
    return experiment_path


def _dataset_identity(args: argparse.Namespace) -> dict[str, Any]:
    identity = {
        "dispatch_sha256": sha256_file(args.data_path),
        "load_tiers_sha256": sha256_file(args.load_tiers_path),
        "manifest_sha256": sha256_file(args.data_manifest) if args.data_manifest else None,
    }
    if args.data_manifest:
        # Parse now so corrupt/incomplete JSON cannot be silently archived.
        manifest = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Data manifest must contain a JSON object")
        # The frozen snapshot directory is an inseparable artifact; do not mix
        # a manifest with similarly named files from another location.
        directory = Path(args.data_manifest).resolve().parent
        if Path(args.data_path).resolve().parent != directory or Path(args.load_tiers_path).resolve().parent != directory:
            raise ValueError("Dispatch, load tiers and manifest must come from the same frozen directory")
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("outputs"), list):
            raise ValueError("Unsupported frozen manifest schema; expected schema_version=1 and outputs list")
        recorded = {}
        for record in manifest["outputs"]:
            path = (directory / record["path"]).resolve()
            if path.parent != directory or path in recorded:
                raise ValueError("Invalid or duplicate frozen output path")
            recorded[path] = record
            if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Frozen output differs from manifest: {path.name}")
        if Path(args.data_path).resolve() not in recorded or Path(args.load_tiers_path).resolve() not in recorded:
            raise ValueError("Dispatch/load tiers are not declared frozen manifest outputs")
    return identity


def _legacy_main(args: argparse.Namespace) -> Path:
    """Historical multi-step implementation, retained for old MPC/replay only."""
    if args.skip_training:
        if args.mode == "evaluate":
            raise ValueError("--skip_training is the legacy closed-loop alias; omit it for evaluate")
        args.mode = "closed-loop"
    if args.gpus:
        args.device = "cuda"
    if args.mode == "closed-loop":
        if args.resume:
            raise ValueError("--resume is only valid for train")
        args.skip_training = True
        return _closed_loop_main(args)
    if args.allow_untrained:
        raise ValueError("--allow_untrained applies only to closed-loop smoke runs")
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
    train_dataset = CSVDataset(config=config, split="train")
    checkpoint = load_checkpoint(args.load_model) if args.load_model else None
    if checkpoint is not None and checkpoint.get("dataset_identity") != identity:
        raise ValueError("Checkpoint does not match frozen data/manifest; evaluation/resume refused")
    controller = _build_controller(config, train_dataset, args)
    # Build views after restoring the exact training scaler.
    experiment_path = Path(args.resume).resolve().parent if args.resume else _experiment_path(args)
    _configure_logging(experiment_path, resume=bool(args.resume))
    dataset_summary = train_dataset.validation_summary()
    LOGGER.info("M0 %s | data %s", args.mode, identity)
    LOGGER.info("Dataset: %s", dataset_summary)
    code_hashes = code_fingerprint(Path(__file__).resolve().parent)
    shared = {
        "config": asdict(config), "dataset_scaler": train_dataset.state_dict(),
        "model_architecture": {"hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
                               "input_dim": train_dataset.input_dim},
        "dataset_identity": identity, "code_sha256": code_hashes, "seed": args.seed,
    }
    metadata = {"arguments": vars(args), "runtime": runtime_versions(), "config": asdict(config),
                "dataset": dataset_summary, "dataset_identity": identity,
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


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HKUST 15-minute PV-LSTM plus rolling resilience MPC"
    )
    parser.add_argument("--mode", choices=("train", "evaluate", "closed-loop"), default="train")
    data = parser.add_argument_group("data")
    data.add_argument("--data_path", default=str(DEFAULT_CONFIG.data_path))
    data.add_argument("--load_tiers_path", default=str(DEFAULT_CONFIG.load_tiers_path))
    data.add_argument("--num_workers", type=int, default=0)
    data.add_argument("--data_manifest", type=Path, default=None)

    model = parser.add_argument_group("model")
    model.add_argument("--hidden_dim", type=int, default=32)
    model.add_argument("--num_layers", type=int, default=1)
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
    training.add_argument("--torch_num_threads", type=int, default=4)
    training.add_argument("--patience", type=int, default=8)
    training.add_argument("--min_delta", type=float, default=0.01)
    training.add_argument("--resume", type=Path, default=None)
    training.add_argument("--skip_training", action="store_true")
    training.add_argument(
        "--allow_untrained",
        action="store_true",
        help="Allow random-weight rollout for code smoke tests; never use as a result.",
    )
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

    rollout = parser.add_argument_group("closed loop")
    rollout.add_argument("--rollout_steps", type=int, default=96)
    rollout.add_argument("--rollout_start_index", type=int, default=0)

    output = parser.add_argument_group("experiment")
    output.add_argument("--seed", type=int, default=42)
    output.add_argument("--save_dir", default="experiments")
    output.add_argument("--experiment_name", default=None)
    output.add_argument("--no_plots", action="store_true")
    output.add_argument("--plot_plan_step", type=int, default=0)
    output.add_argument("--plot_dpi", type=int, default=180)
    output.add_argument("--no_progress_bar", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """The public M0 forecast interface always defaults to the new models."""
    parser = _variant_runner.build_parser("LSTM")
    parser.description = "M0: two-layer LSTM/GRU, 96 historical PV points to the next 15-minute PV"
    parser.add_argument("--model_type", "--model", choices=("lstm", "gru"), default="lstm",
                        help="PV forecaster (default: two-layer LSTM).")
    # Resolve after model selection so a GRU invocation cannot write under LSTM.
    parser.set_defaults(save_dir=None)
    parser.epilog = ("Historical multi-step MPC remains available through --mode closed-loop. "
                     "It requires the old multi-step checkpoint, not a 96-to-1 model.")
    return parser


def main(args: argparse.Namespace) -> Path:
    if args.mode == "closed-loop":
        if args.load_model is not None:
            checkpoint = load_checkpoint(args.load_model)
            if checkpoint.get("config", {}).get("prediction_horizon") == 1:
                raise ValueError("96-to-1 LSTM/GRU checkpoints cannot supply the legacy 96-step MPC; "
                                 "use --mode evaluate for these models")
        return _legacy_main(args)
    model_type = args.model_type
    args.recurrent_type = model_type.upper()
    if args.save_dir is None:
        args.save_dir = str(PROJECT_ROOT / "experiments" / f"m0_{model_type}_2layer_pvonly_2021_2023_96to1")
    controller_class = {"lstm": PVLSTMController, "gru": PVGRUController}[model_type]
    return _variant_runner.main(args, controller_class)


def cli(argv: list[str] | None = None) -> Path:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--mode", default="train")
    selection, _ = selector.parse_known_args(argv)
    parser = build_legacy_parser() if selection.mode == "closed-loop" else build_parser()
    return main(parser.parse_args(argv))


if __name__ == "__main__":
    cli()

"""Training and closed-loop execution for the HKUST PV-MPC controller."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.models.controller import Controller
from src.plants.pymgrid_adapter import PymgridAdapter


FUTURE_KEYS = (
    "load",
    "import_price",
    "export_price",
    "grid_available",
    "critical_load",
)
ACTION_DECISION_MAP = {
    # Adapter action name: controller decision name.
    "battery_charge": "charge",
    "battery_discharge": "discharge",
    "grid_import": "grid_import",
    "grid_export": "grid_export",
}

# Retained for the existing tuning/plotting utilities.
METRICS_TENDENCY = {
    "train_loss_mse": "min",
    "val_loss_mse": "min",
    "test_loss_mse": "min",
}


class OnlineLearner(pl.LightningModule):
    """Train the PV forecaster and execute receding-horizon control.

    Training uses PV forecast MSE. During closed-loop evaluation the controller
    solves the complete MPC horizon, but only the first action is sent to
    :class:`PymgridAdapter`. The adapter is the sole owner of physical SOC
    evolution; its returned SOC is fed back to the next MPC solve.
    """

    def __init__(
        self,
        controller: Controller,
        adapter: PymgridAdapter | None = None,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        initial_soc_kwh: float = 107.5,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if initial_soc_kwh < 0:
            raise ValueError("initial_soc_kwh must be non-negative.")

        self.save_hyperparameters(ignore=["controller", "adapter"])
        self.controller = controller
        self.adapter = adapter
        self.closed_loop_actual = pd.DataFrame()
        self.mpc_plans = pd.DataFrame()
        self.closed_loop_summary: dict[str, float | int | str] = {}

    @staticmethod
    def _future(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        missing = [key for key in FUTURE_KEYS if key not in batch]
        if missing:
            raise KeyError(f"Batch is missing future MPC inputs: {missing}")
        return {key: batch[key] for key in FUTURE_KEYS}

    def _initial_soc(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.full(
            (batch_size, 1),
            float(self.hparams.initial_soc_kwh),
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        history: torch.Tensor,
        future: Mapping[str, torch.Tensor],
        initial_soc: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.controller(history, future, initial_soc)

    def _shared_step(self, batch: Mapping[str, Any], stage: str) -> torch.Tensor:
        history = batch["history"]
        target = batch["pv_target"]
        forecast = self.controller.forecast_pv(history)
        target = target.to(device=forecast.device, dtype=forecast.dtype)
        mse = F.mse_loss(forecast, target)
        self.log(
            f"{stage}_loss_mse",
            mse,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=history.shape[0],
        )
        return mse

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "test")

    def predict_step(
        self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del batch_idx, dataloader_idx
        history = batch["history"]
        initial_soc = self._initial_soc(
            history.shape[0], device=history.device, dtype=history.dtype
        )
        return self(history, self._future(batch), initial_soc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.controller.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )

    @staticmethod
    def _batched_tensor(
        value: Any,
        *,
        device: torch.device,
        add_batch_for_ndim: int,
    ) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == add_batch_for_ndim:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device=device, dtype=torch.float32)

    @classmethod
    def _sample_batch(
        cls, sample: Mapping[str, Any], device: torch.device
    ) -> dict[str, torch.Tensor]:
        batch = {
            "history": cls._batched_tensor(
                sample["history"], device=device, add_batch_for_ndim=2
            ),
            "pv_target": cls._batched_tensor(
                sample["pv_target"], device=device, add_batch_for_ndim=2
            ),
        }
        for key in FUTURE_KEYS:
            batch[key] = cls._batched_tensor(
                sample[key], device=device, add_batch_for_ndim=2
            )
        return batch

    @staticmethod
    def _first(value: Any) -> float:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return float(tensor.reshape(-1)[0].detach().cpu().item())

    @staticmethod
    def _timestamp(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            value = value[0]
        if isinstance(value, torch.Tensor):
            value = value.reshape(-1)[0].detach().cpu().item()
        if isinstance(value, (int, float)) and abs(value) >= 100_000_000:
            return pd.Timestamp(value, unit="s").isoformat()
        return str(value)

    @staticmethod
    def _decision_at(
        decisions: Mapping[str, torch.Tensor], key: str, horizon_index: int
    ) -> float:
        value = decisions[key].detach().cpu()
        if value.ndim == 1:
            return float(value[horizon_index].item())
        return float(value[0, horizon_index].item())

    def run_closed_loop(
        self,
        dataset: Any,
        *,
        adapter: PymgridAdapter | None = None,
        steps: int = 96,
        start_index: int = 0,
        reset_adapter: bool = True,
    ) -> dict[str, Any]:
        """Run ``steps`` receding-horizon decisions in chronological order."""
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if start_index < 0:
            raise ValueError("start_index must be non-negative.")
        if len(dataset) < start_index + steps:
            raise ValueError(
                f"Dataset contains {len(dataset)} samples, but indices "
                f"{start_index}:{start_index + steps} were requested."
            )

        if hasattr(dataset, "timestamps"):
            selected_times = pd.DatetimeIndex(dataset.timestamps[start_index : start_index + steps])
            expected_step = pd.Timedelta(getattr(getattr(dataset, "config", None), "frequency", "15min"))
            if len(selected_times) > 1 and not (
                selected_times[1:] - selected_times[:-1] == expected_step
            ).all():
                raise ValueError(
                    "Closed-loop execution requires continuous timestamps; the requested "
                    "range crosses samples excluded for data quality. Select a continuous "
                    "event instead of carrying SOC across an unexecuted time gap."
                )

        simulator = adapter or self.adapter
        if simulator is None:
            raise ValueError("A PymgridAdapter is required for closed-loop execution.")
        if reset_adapter:
            simulator.reset(soc_kwh=float(self.hparams.initial_soc_kwh))

        device = next(self.controller.parameters()).device
        actual_rows: list[dict[str, Any]] = []
        plan_rows: list[dict[str, Any]] = []
        was_training = self.controller.training
        self.controller.eval()

        for step_index in range(steps):
            sample_index = start_index + step_index
            sample = dataset[sample_index]
            batch = self._sample_batch(sample, device)
            initial_soc = torch.tensor(
                [[simulator.get_soc_kwh()]], device=device, dtype=torch.float32
            )

            with torch.no_grad():
                pv_forecast, decisions = self(
                    batch["history"], self._future(batch), initial_soc
                )

            missing_actions = [
                decision_key
                for decision_key in ACTION_DECISION_MAP.values()
                if decision_key not in decisions
            ]
            if missing_actions:
                raise KeyError(f"MPC decisions are missing actions: {missing_actions}")

            first_action = {
                adapter_key: max(
                    0.0, self._decision_at(decisions, decision_key, 0)
                )
                for adapter_key, decision_key in ACTION_DECISION_MAP.items()
            }
            actual = {
                "pv": self._first(sample.get("pv_actual", sample["pv_target"])),
                "load": self._first(sample["load"]),
                "import_price": self._first(sample["import_price"]),
                "export_price": self._first(sample["export_price"]),
                "grid_available": self._first(sample["grid_available"]),
                "critical_load": self._first(sample["critical_load"]),
            }
            execution = simulator.step(first_action, actual)
            execution.update(
                {
                    "timestamp": self._timestamp(sample["timestamp"]),
                    "dataset_index": sample_index,
                    "pv_forecast_first_kw": float(
                        pv_forecast[0, 0, 0].detach().cpu().item()
                    ),
                }
            )
            for key, value in first_action.items():
                execution[f"planned_{key}_kw"] = value
            actual_rows.append(execution)

            horizon = pv_forecast.shape[1]
            future_timestamps = sample.get("future_timestamps", ())
            for horizon_index in range(horizon):
                if len(future_timestamps) > horizon_index:
                    plan_timestamp = self._timestamp(future_timestamps[horizon_index])
                else:
                    plan_timestamp = str(horizon_index)
                row: dict[str, Any] = {
                    "decision_step": step_index,
                    "dataset_index": sample_index,
                    "decision_timestamp": self._timestamp(sample["timestamp"]),
                    "horizon_step": horizon_index,
                    "horizon_timestamp": plan_timestamp,
                    "pv_forecast_kw": float(
                        pv_forecast[0, horizon_index, 0].detach().cpu().item()
                    ),
                }
                for key in FUTURE_KEYS:
                    row[key] = float(batch[key][0, horizon_index].detach().cpu().item())
                for key, value in decisions.items():
                    if value.ndim < 2 or value.shape[1] <= horizon_index:
                        continue
                    row[key] = self._decision_at(decisions, key, horizon_index)
                if "soc" in decisions and decisions["soc"].shape[1] > horizon_index + 1:
                    row["soc_end"] = self._decision_at(decisions, "soc", horizon_index + 1)
                plan_rows.append(row)

        if was_training:
            self.controller.train()

        actual_df = pd.DataFrame(actual_rows)
        plans_df = pd.DataFrame(plan_rows)
        summary = self._summarise(actual_df)
        self.closed_loop_actual = actual_df
        self.mpc_plans = plans_df
        self.closed_loop_summary = summary
        return {"actual": actual_df, "plans": plans_df, "summary": summary}

    @staticmethod
    def _summarise(actual: pd.DataFrame) -> dict[str, float | int | str]:
        if actual.empty:
            return {"steps": 0}

        def total(column: str) -> float:
            return float(actual[column].sum()) if column in actual else 0.0

        realised_cost = total("component_cost_sum")
        return {
            "steps": int(len(actual)),
            "start": str(actual["timestamp"].iloc[0]),
            "end": str(actual["timestamp"].iloc[-1]),
            "outage_steps": int((actual["grid_available"] < 1.0).sum()),
            "islanded_steps": int((actual["grid_available"] <= 0.0).sum()),
            "pv_available_kwh": total("pv_available_kwh"),
            "pv_used_kwh": total("pv_used_kwh"),
            "pv_curtailed_kwh": total("pv_curtailed_kwh"),
            "load_kwh": total("load_kwh"),
            "load_served_kwh": total("load_served_kwh"),
            "loss_load_kwh": total("loss_load_kwh"),
            "critical_loss_load_kwh": total("critical_loss_load_kwh"),
            "noncritical_loss_load_kwh": total("noncritical_loss_load_kwh"),
            "grid_import_kwh": total("grid_import_kwh"),
            "grid_export_kwh": total("grid_export_kwh"),
            "realised_operating_cost": realised_cost,
            "total_cost": realised_cost,
            "minimum_soc_kwh": float(actual["soc_after_kwh"].min()),
            "final_soc_kwh": float(actual["soc_after_kwh"].iloc[-1]),
        }

    @staticmethod
    def save_closed_loop(results: Mapping[str, Any], output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results["actual"].to_csv(output_path / "closed_loop_actual.csv", index=False)
        results["plans"].to_csv(output_path / "mpc_plans.csv", index=False)
        pd.Series(results["summary"], name="value").to_csv(
            output_path / "closed_loop_summary.csv", header=True
        )

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        return parser


__all__ = [
    "ACTION_DECISION_MAP",
    "FUTURE_KEYS",
    "METRICS_TENDENCY",
    "OnlineLearner",
]

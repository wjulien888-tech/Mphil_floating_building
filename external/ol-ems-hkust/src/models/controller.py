"""PV forecasting and differentiable resilience MPC.

The neural network in this module predicts PV only. Quantities known over the
MPC horizon (demand, tariffs, grid availability, and critical demand) are
explicit optimisation inputs.
"""

import argparse
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cvxpy as cp
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer


FUTURE_KEYS = (
    "load",
    "import_price",
    "export_price",
    "grid_available",
    "critical_load",
)

VARIABLES = (
    "soc",
    "charge",
    "discharge",
    "grid_import",
    "grid_export",
    "pv_used",
    "pv_curtailment",
    "unmet_critical",
    "unmet_noncritical",
)


class OptimisationLayer(nn.Module):
    """Linear, DPP-compliant resilience MPC.

    Power is in kW and ``soc`` is in kWh. With ``sample_time=0.25`` the
    storage equation represents one 15-minute step::

        soc[t+1] = retention * soc[t]
                   + 0.25 * eta_charge * charge[t]
                   - 0.25 / eta_discharge * discharge[t]

    ``grid_available`` can be binary or a derating factor in ``[0, 1]``.
    By default battery charging is PV-attributable (``charge <= pv_used``),
    matching the project's E4/E5 comparison baseline; grid charging must be
    enabled explicitly.
    Fixed linear objective coefficients prioritise critical unmet demand, then
    non-critical unmet demand, cost, curtailment, and battery cycling. They are
    constants rather than CVXPY Parameters to retain compatibility with the
    project's older cvxpylayers version.

    ``A_matrix`` and ``B_matrix`` remain accepted for compatibility. If they
    are supplied, their scalar coefficients override the efficiency formula.
    """

    DEFAULT_OBJECTIVE_WEIGHTS = {
        "critical_unmet": 10_000.0,
        "noncritical_unmet": 1_000.0,
        "energy_cost": 1.0,
        "pv_curtailment": 0.01,
        "battery_cycle": 0.001,
    }

    def __init__(
        self,
        prediction_horizon: int,
        A_matrix: Optional[torch.Tensor] = None,
        B_matrix: Optional[torch.Tensor] = None,
        charge_min_max: Optional[Sequence[float]] = None,
        storage_power_min_max: Optional[Sequence[float]] = None,
        grid_power_min_max: Optional[Sequence[float]] = None,
        *,
        sample_time: float = 0.25,
        charge_efficiency: float = 0.95,
        discharge_efficiency: Optional[float] = None,
        self_discharge: float = 0.0,
        allow_grid_charging: bool = False,
        no_arbitrage_margin: float = 1.0e-3,
        objective_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()

        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive.")
        if charge_min_max is None:
            raise ValueError("charge_min_max must be provided in kWh.")
        if storage_power_min_max is None:
            raise ValueError("storage_power_min_max must be provided in kW.")
        if grid_power_min_max is None:
            raise ValueError("grid_power_min_max must be provided in kW.")

        self._prediction_horizon = int(prediction_horizon)
        self._sample_time = float(sample_time)
        if self._sample_time <= 0:
            raise ValueError("sample_time must be positive.")
        self.allow_grid_charging = bool(allow_grid_charging)
        self.no_arbitrage_margin = float(no_arbitrage_margin)
        if self.no_arbitrage_margin < 0:
            raise ValueError("no_arbitrage_margin must be non-negative.")

        soc_min, soc_max = self._ordered_pair(charge_min_max, "charge_min_max")
        storage_min, storage_max = self._ordered_pair(
            storage_power_min_max, "storage_power_min_max"
        )
        grid_min, grid_max = self._ordered_pair(
            grid_power_min_max, "grid_power_min_max"
        )
        if storage_min > 0 or storage_max < 0:
            raise ValueError("storage_power_min_max must contain zero.")
        if grid_min > 0 or grid_max < 0:
            raise ValueError("grid_power_min_max must contain zero.")

        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charge_power_max = storage_max
        self.discharge_power_max = -storage_min if storage_min < 0 else storage_max
        self.grid_import_max = -grid_min if grid_min < 0 else grid_max
        self.grid_export_max = grid_max if grid_max > 0 else -grid_min
        if self.charge_power_max <= 0 or self.discharge_power_max <= 0:
            raise ValueError("Battery charge and discharge limits must be positive.")

        (
            self.soc_retention,
            self.charge_coefficient,
            self.discharge_coefficient,
        ) = self._storage_coefficients(
            A_matrix=A_matrix,
            B_matrix=B_matrix,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            self_discharge=self_discharge,
        )

        weights = dict(self.DEFAULT_OBJECTIVE_WEIGHTS)
        if objective_weights is not None:
            unknown = set(objective_weights).difference(weights)
            if unknown:
                raise ValueError(
                    "Unknown objective weight(s): " + ", ".join(sorted(unknown))
                )
            weights.update(objective_weights)
        if any(float(value) < 0 for value in weights.values()):
            raise ValueError("All objective weights must be non-negative.")
        self.objective_weights = {key: float(value) for key, value in weights.items()}
        self._build_problem()

    @staticmethod
    def _ordered_pair(values: Sequence[float], name: str) -> Tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        lower, upper = float(values[0]), float(values[1])
        if lower >= upper:
            raise ValueError(f"{name}[0] must be smaller than {name}[1].")
        return lower, upper

    def _storage_coefficients(
        self,
        A_matrix: Optional[torch.Tensor],
        B_matrix: Optional[torch.Tensor],
        charge_efficiency: float,
        discharge_efficiency: Optional[float],
        self_discharge: float,
    ) -> Tuple[float, float, float]:
        if (A_matrix is None) != (B_matrix is None):
            raise ValueError("A_matrix and B_matrix must be supplied together.")

        if A_matrix is not None and B_matrix is not None:
            A = torch.as_tensor(A_matrix).detach().cpu()
            B = torch.as_tensor(B_matrix).detach().cpu()
            if A.numel() != 1 or B.shape != (1, 2):
                raise ValueError("Expected A_matrix shape (1, 1) and B_matrix shape (1, 2).")
            retention = float(A.reshape(-1)[0])
            charge_coefficient = float(B[0, 0])
            discharge_coefficient = -float(B[0, 1])
        else:
            eta_charge = float(charge_efficiency)
            eta_discharge = (
                eta_charge
                if discharge_efficiency is None
                else float(discharge_efficiency)
            )
            if not 0 < eta_charge <= 1 or not 0 < eta_discharge <= 1:
                raise ValueError("Battery efficiencies must be in (0, 1].")
            if not 0 <= self_discharge < 1:
                raise ValueError("self_discharge must be in [0, 1).")
            retention = 1.0 - float(self_discharge)
            charge_coefficient = self._sample_time * eta_charge
            discharge_coefficient = self._sample_time / eta_discharge

        if not 0 < retention <= 1:
            raise ValueError("The SOC retention coefficient must be in (0, 1].")
        if charge_coefficient <= 0 or discharge_coefficient <= 0:
            raise ValueError("Storage input coefficients must be positive.")
        return retention, charge_coefficient, discharge_coefficient

    def _build_problem(self) -> None:
        horizon = self._prediction_horizon

        pv = cp.Parameter(horizon, nonneg=True, name="pv")
        load = cp.Parameter(horizon, nonneg=True, name="load")
        import_price = cp.Parameter(horizon, name="import_price")
        export_price = cp.Parameter(horizon, name="export_price")
        grid_available = cp.Parameter(horizon, nonneg=True, name="grid_available")
        critical_load = cp.Parameter(horizon, nonneg=True, name="critical_load")
        initial_soc = cp.Parameter(1, name="initial_soc")

        soc = cp.Variable(horizon + 1, name="soc")
        charge = cp.Variable(horizon, nonneg=True, name="charge")
        discharge = cp.Variable(horizon, nonneg=True, name="discharge")
        grid_import = cp.Variable(horizon, nonneg=True, name="grid_import")
        grid_export = cp.Variable(horizon, nonneg=True, name="grid_export")
        pv_used = cp.Variable(horizon, nonneg=True, name="pv_used")
        pv_curtailment = cp.Variable(horizon, nonneg=True, name="pv_curtailment")
        unmet_critical = cp.Variable(horizon, nonneg=True, name="unmet_critical")
        unmet_noncritical = cp.Variable(
            horizon, nonneg=True, name="unmet_noncritical"
        )

        noncritical_load = load - critical_load
        served_load = load - unmet_critical - unmet_noncritical
        constraints = [
            critical_load <= load,
            unmet_critical <= critical_load,
            unmet_noncritical <= noncritical_load,
            soc[0] == initial_soc[0],
            soc >= self.soc_min,
            soc <= self.soc_max,
            charge <= self.charge_power_max,
            discharge <= self.discharge_power_max,
            grid_import <= self.grid_import_max * grid_available,
            grid_export <= self.grid_export_max * grid_available,
            pv_used + pv_curtailment == pv,
            soc[1:]
            == self.soc_retention * soc[:-1]
            + self.charge_coefficient * charge
            - self.discharge_coefficient * discharge,
            pv_used + discharge + grid_import
            == served_load + charge + grid_export,
        ]
        if not self.allow_grid_charging:
            # Every unit charged must be attributable to PV. The power balance
            # may then let the grid serve load while PV charges the battery,
            # but PV=0 can never produce a charging command.
            constraints.append(charge <= pv_used)

        weights = self.objective_weights
        energy_cost = cp.sum(
            cp.multiply(import_price, grid_import)
            - cp.multiply(export_price, grid_export)
        )
        objective = self._sample_time * (
            weights["critical_unmet"] * cp.sum(unmet_critical)
            + weights["noncritical_unmet"] * cp.sum(unmet_noncritical)
            + weights["energy_cost"] * energy_cost
            + weights["pv_curtailment"] * cp.sum(pv_curtailment)
            + weights["battery_cycle"] * cp.sum(charge + discharge)
        )

        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dcp(dpp=True):
            raise RuntimeError("The resilience MPC problem must be DCP and DPP compliant.")
        self._problem = problem
        self._optimisation_layer = CvxpyLayer(
            problem,
            parameters=[
                pv,
                load,
                import_price,
                export_price,
                grid_available,
                critical_load,
                initial_soc,
            ],
            variables=[
                soc,
                charge,
                discharge,
                grid_import,
                grid_export,
                pv_used,
                pv_curtailment,
                unmet_critical,
                unmet_noncritical,
            ],
        )

    @property
    def prediction_horizon(self) -> int:
        return self._prediction_horizon

    @staticmethod
    def _horizon_tensor(
        value: torch.Tensor,
        name: str,
        batch_size: int,
        horizon: int,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        if tensor.ndim == 1 and tensor.shape[0] == horizon:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != horizon:
            raise ValueError(
                f"{name} must have shape [batch, {horizon}] or "
                f"[batch, {horizon}, 1]; got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1)
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch size must be {batch_size}; got {tensor.shape[0]}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        return tensor.to(dtype=torch.float64)

    def forward(
        self,
        pv_forecast: torch.Tensor,
        load: torch.Tensor,
        import_price: torch.Tensor,
        export_price: torch.Tensor,
        grid_available: torch.Tensor,
        critical_load: torch.Tensor,
        initial_soc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Solve batched MPC and return named horizon decisions."""
        if pv_forecast.ndim not in (2, 3):
            raise ValueError("pv_forecast must include batch and horizon dimensions.")
        batch_size = pv_forecast.shape[0]
        horizon = self._prediction_horizon
        device = pv_forecast.device

        pv = self._horizon_tensor(
            pv_forecast, "pv_forecast", batch_size, horizon, device
        )
        load_tensor = self._horizon_tensor(load, "load", batch_size, horizon, device)
        import_price_tensor = self._horizon_tensor(
            import_price, "import_price", batch_size, horizon, device
        )
        export_price_tensor = self._horizon_tensor(
            export_price, "export_price", batch_size, horizon, device
        )
        grid_tensor = self._horizon_tensor(
            grid_available, "grid_available", batch_size, horizon, device
        )
        critical_tensor = self._horizon_tensor(
            critical_load, "critical_load", batch_size, horizon, device
        )

        soc_tensor = torch.as_tensor(initial_soc, device=device)
        if soc_tensor.ndim == 0:
            soc_tensor = soc_tensor.reshape(1, 1)
        elif soc_tensor.ndim == 1:
            soc_tensor = soc_tensor.reshape(-1, 1)
        else:
            soc_tensor = soc_tensor.reshape(soc_tensor.shape[0], -1)
        if soc_tensor.shape[1] != 1:
            raise ValueError(
                "initial_soc must contain one scalar per batch item; got "
                f"{tuple(soc_tensor.shape)}."
            )
        if soc_tensor.shape[0] == 1 and batch_size > 1:
            soc_tensor = soc_tensor.expand(batch_size, -1)
        if soc_tensor.shape[0] != batch_size:
            raise ValueError(
                f"initial_soc batch size must be {batch_size}; got {soc_tensor.shape[0]}."
            )
        if not torch.isfinite(soc_tensor).all():
            raise ValueError("initial_soc contains NaN or infinite values.")
        soc_tensor = soc_tensor.to(dtype=torch.float64)

        tolerance = 1e-7
        if torch.any(pv < -tolerance):
            raise ValueError("pv_forecast must be non-negative.")
        if torch.any(load_tensor < -tolerance):
            raise ValueError("load must be non-negative.")
        if torch.any(import_price_tensor < -tolerance) or torch.any(
            export_price_tensor < -tolerance
        ):
            raise ValueError("import_price and export_price must be non-negative.")
        if torch.any(critical_tensor < -tolerance):
            raise ValueError("critical_load must be non-negative.")
        if torch.any(critical_tensor > load_tensor + tolerance):
            raise ValueError("critical_load must not exceed total load.")
        if torch.any(grid_tensor < -tolerance) or torch.any(
            grid_tensor > 1.0 + tolerance
        ):
            raise ValueError("grid_available must lie in [0, 1].")
        if torch.any(soc_tensor < self.soc_min - tolerance) or torch.any(
            soc_tensor > self.soc_max + tolerance
        ):
            raise ValueError(
                f"initial_soc must lie in [{self.soc_min}, {self.soc_max}] kWh."
            )

        pv = pv.clamp_min(0.0)
        load_tensor = load_tensor.clamp_min(0.0)
        critical_tensor = critical_tensor.clamp_min(0.0).minimum(load_tensor)
        grid_tensor = grid_tensor.clamp(0.0, 1.0)
        soc_tensor = soc_tensor.clamp(self.soc_min, self.soc_max)

        # A convex LP cannot impose exact import/export mutual exclusivity.  If
        # a scenario feed-in tariff exceeds its simultaneous import tariff,
        # unconstrained non-negative flows would create artificial arbitrage.
        # Use a transparent conservative price only inside MPC; the adapter
        # still accounts realised exports at the original scenario tariff.
        effective_export_price = torch.minimum(
            export_price_tensor,
            (import_price_tensor - self.no_arbitrage_margin).clamp_min(0.0),
        )

        solution = self._optimisation_layer(
            pv,
            load_tensor,
            import_price_tensor,
            effective_export_price,
            grid_tensor,
            critical_tensor,
            soc_tensor,
        )
        decisions = dict(zip(VARIABLES, solution))
        # Import and export affect every constraint only through their net
        # value. Remove solver-sized overlap before exposing the physical plan;
        # this preserves the exact net exchange and makes the returned actions
        # mutually exclusive without introducing integer variables.
        raw_grid_import = decisions["grid_import"]
        raw_grid_export = decisions["grid_export"]
        net_grid_import = raw_grid_import - raw_grid_export
        decisions["grid_overlap_removed"] = torch.minimum(
            raw_grid_import, raw_grid_export
        ).clamp_min(0.0)
        decisions["grid_import"] = net_grid_import.clamp_min(0.0)
        decisions["grid_export"] = (-net_grid_import).clamp_min(0.0)
        decisions["served_critical"] = critical_tensor - decisions["unmet_critical"]
        decisions["served_noncritical"] = (
            load_tensor - critical_tensor - decisions["unmet_noncritical"]
        )
        decisions["load_served"] = (
            decisions["served_critical"] + decisions["served_noncritical"]
        )
        decisions["net_grid_export"] = (
            decisions["grid_export"] - decisions["grid_import"]
        )
        decisions["optimisation_export_price"] = effective_export_price
        return decisions


class Controller(nn.Module):
    """LSTM PV forecaster followed by differentiable resilience MPC.

    ``dataset`` is optional and is used only to infer ``input_dim`` and the PV
    scaler range. New code should pass those values directly.
    """

    def __init__(
        self,
        optimisation_layer: OptimisationLayer,
        dataset: Optional[Any] = None,
        hidden_dim: int = 32,
        num_layers: int = 1,
        input_dim: Optional[int] = None,
        pv_min: Optional[float] = None,
        pv_max: Optional[float] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive.")

        inferred_input_dim = self._infer_input_dim(dataset)
        if input_dim is None:
            input_dim = inferred_input_dim
        if input_dim is None or int(input_dim) <= 0:
            raise ValueError("input_dim must be provided or exposed by dataset.")
        if inferred_input_dim is not None and int(input_dim) != inferred_input_dim:
            raise ValueError(
                f"input_dim={input_dim} disagrees with dataset input_dim="
                f"{inferred_input_dim}."
            )

        if dataset is not None and hasattr(dataset, "prediction_horizon"):
            dataset_horizon = int(dataset.prediction_horizon)
            if dataset_horizon != optimisation_layer.prediction_horizon:
                raise ValueError(
                    "Dataset and optimisation horizons differ: "
                    f"{dataset_horizon} != {optimisation_layer.prediction_horizon}."
                )

        inferred_bounds = self._infer_pv_bounds(dataset)
        if pv_min is None and inferred_bounds is not None:
            pv_min = inferred_bounds[0]
        if pv_max is None and inferred_bounds is not None:
            pv_max = inferred_bounds[1]
        if pv_min is None:
            pv_min = 0.0
        if pv_max is None:
            pv_max = 1.0
        pv_min = max(0.0, float(pv_min))
        pv_max = float(pv_max)
        if pv_max <= pv_min:
            raise ValueError("pv_max must be greater than pv_min.")

        self._hidden_dim = int(hidden_dim)
        self._num_layers = int(num_layers)
        self._input_dim = int(input_dim)
        self._prediction_horizon = optimisation_layer.prediction_horizon
        self._optimisation_layer = optimisation_layer
        self._lstm = nn.LSTM(
            input_size=self._input_dim,
            hidden_size=self._hidden_dim,
            num_layers=self._num_layers,
            batch_first=True,
            bias=True,
        )
        self._pv_head = nn.Linear(self._hidden_dim, self._prediction_horizon)
        self.register_buffer("_pv_min", torch.tensor(pv_min, dtype=torch.float32))
        self.register_buffer(
            "_pv_range", torch.tensor(pv_max - pv_min, dtype=torch.float32)
        )

    @staticmethod
    def _infer_input_dim(dataset: Optional[Any]) -> Optional[int]:
        if dataset is None:
            return None
        if hasattr(dataset, "input_dim"):
            input_dim = getattr(dataset, "input_dim")
            input_dim = input_dim() if callable(input_dim) else input_dim
            return int(input_dim)
        if hasattr(dataset, "input_features"):
            return len(dataset.input_features)
        scaler = getattr(dataset, "scaler", None)
        if scaler is not None and hasattr(scaler, "n_features_in_"):
            return int(scaler.n_features_in_)
        return None

    @staticmethod
    def _infer_pv_bounds(dataset: Optional[Any]) -> Optional[Tuple[float, float]]:
        if dataset is None:
            return None
        if hasattr(dataset, "pv_min") and hasattr(dataset, "pv_max"):
            return float(dataset.pv_min), float(dataset.pv_max)

        feature_names = getattr(dataset, "available_features", None)
        if feature_names is None:
            feature_names = getattr(dataset, "input_features", None)
        pv_index = None
        if feature_names is not None:
            for index, name in enumerate(feature_names):
                if str(name).strip().lower() == "pv":
                    pv_index = index
                    break

        minimum = getattr(dataset, "scaling_factors_min", None)
        maximum = getattr(dataset, "scaling_factors_max", None)
        if minimum is not None and maximum is not None and pv_index is not None:
            minimum_tensor = torch.as_tensor(minimum).reshape(-1)
            maximum_tensor = torch.as_tensor(maximum).reshape(-1)
            if pv_index < minimum_tensor.numel() and pv_index < maximum_tensor.numel():
                return (
                    float(minimum_tensor[pv_index]),
                    float(maximum_tensor[pv_index]),
                )

        scaler = getattr(dataset, "scaler", None)
        if isinstance(scaler, Mapping) and "pv" in scaler:
            scaler = scaler["pv"]
            pv_index = 0
        if scaler is not None:
            data_min = getattr(scaler, "data_min_", None)
            data_max = getattr(scaler, "data_max_", None)
            if data_min is not None and data_max is not None:
                data_min = torch.as_tensor(data_min).reshape(-1)
                data_max = torch.as_tensor(data_max).reshape(-1)
                index = 0 if pv_index is None else pv_index
                if index < data_min.numel() and index < data_max.numel():
                    return float(data_min[index]), float(data_max[index])
        return None

    @staticmethod
    def _future_dict(
        future: Union[Mapping[str, torch.Tensor], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(future, Mapping):
            missing = [key for key in FUTURE_KEYS if key not in future]
            if missing:
                raise ValueError(
                    "future is missing required field(s): " + ", ".join(missing)
                )
            return {key: future[key] for key in FUTURE_KEYS}
        if torch.is_tensor(future):
            if future.ndim == 2 and future.shape[-1] == len(FUTURE_KEYS):
                future = future.unsqueeze(0)
            if future.ndim != 3 or future.shape[-1] != len(FUTURE_KEYS):
                raise ValueError(
                    "Tensor future must have shape [batch, H, 5] in FUTURE_KEYS order."
                )
            return {
                key: future[..., index] for index, key in enumerate(FUTURE_KEYS)
            }
        raise TypeError("future must be a mapping or a [batch, H, 5] tensor.")

    def forward(
        self,
        history: torch.Tensor,
        future: Union[Mapping[str, torch.Tensor], torch.Tensor],
        initial_soc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forecast PV and optimise the horizon.

        ``history`` has shape ``[batch, lookback, input_dim]``. ``future`` is a
        mapping with all ``FUTURE_KEYS`` or ``[batch, H, 5]`` in that order.
        Returns physical PV in kW as ``[batch, H, 1]`` and named decisions.
        """
        pv_forecast = self.forecast_pv(history)

        future_values = self._future_dict(future)
        decisions = self._optimisation_layer(
            pv_forecast=pv_forecast,
            load=future_values["load"],
            import_price=future_values["import_price"],
            export_price=future_values["export_price"],
            grid_available=future_values["grid_available"],
            critical_load=future_values["critical_load"],
            initial_soc=initial_soc,
        )
        return pv_forecast, decisions

    def forecast_pv(self, history: torch.Tensor) -> torch.Tensor:
        """Forecast physical PV without solving MPC.

        Forecast-only training should use this method: the supervised loss is
        defined solely on PV, so solving an optimisation problem for every
        training batch would add substantial work without changing that loss.
        """
        if history.ndim != 3:
            raise ValueError(
                "history must have shape [batch, lookback, input_dim]; got "
                f"{tuple(history.shape)}."
            )
        if history.shape[-1] != self._input_dim:
            raise ValueError(
                f"history input_dim must be {self._input_dim}; got {history.shape[-1]}."
            )
        if history.shape[1] == 0:
            raise ValueError("history lookback dimension must not be empty.")

        history = history.to(dtype=self._pv_head.weight.dtype)
        encoded, _ = self._lstm(history)
        normalised_pv = torch.sigmoid(self._pv_head(encoded[:, -1, :]))
        pv_forecast = (
            self._pv_min + self._pv_range * normalised_pv
        ).unsqueeze(-1)
        return pv_forecast

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--hidden_dim", type=int, default=32, help="LSTM hidden dimension."
        )
        parser.add_argument(
            "--num_layers", type=int, default=1, help="Number of LSTM layers."
        )
        return parser

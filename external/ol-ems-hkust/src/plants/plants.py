"""State-space models used inside the EMS controller.

This module contains prediction models, not the physical closed-loop plant.
For the HKUST experiment, measured PV/load and the executed first action are
advanced by :class:`src.plants.pymgrid_adapter.PymgridAdapter`; that adapter is
the sole owner of realised battery SOC.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Optional, Tuple
import warnings

import torch


class StateSpaceSystem(ABC):
    """Base class for a pure state-space prediction model.

    ``sample_time`` is measured in hours. ``initial_state`` is retained for
    compatibility with the original ol-ems API; this object never stores or
    mutates a running physical state.
    """

    def __init__(self, sample_time: float, initial_state: Sequence[float]) -> None:
        sample_time = float(sample_time)
        if sample_time <= 0:
            raise ValueError("sample_time must be positive hours.")
        if len(initial_state) == 0:
            raise ValueError("initial_state must contain at least one value.")
        self._sample_time = sample_time
        self._initial_state = [float(value) for value in initial_state]

    @abstractmethod
    def _get_plant_matrices(
        self, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy alias returning the internal prediction-model matrices."""

    @abstractmethod
    def get_model_matrices(
        self, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return matrices ``A, B`` for ``x[t+1] = A x[t] + B u[t]``."""

    def get_initial_state(self) -> torch.Tensor:
        """Return the configured initial model state without retaining it."""
        return torch.tensor(self._initial_state, dtype=torch.float64).reshape(-1, 1)

    def plant_simulation(self, x_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """Legacy pure model propagation; it does not advance physical state.

        New HKUST closed-loop code must use ``PymgridAdapter.step`` for realised
        SOC. This method remains only so old analytical callers do not break.
        """
        warnings.warn(
            "plant_simulation() is a prediction-only compatibility method; "
            "PymgridAdapter owns realised SOC progression.",
            DeprecationWarning,
            stacklevel=2,
        )
        A_matrix, B_matrix = self.get_model_matrices(x_t.device)
        return torch.matmul(A_matrix, x_t) + torch.matmul(B_matrix, u_t)

    def get_state_dimension(self) -> int:
        """Return the internal model state dimension."""
        return int(self.get_model_matrices()[0].shape[0])

    def get_input_dimension(self) -> int:
        """Return the internal model input dimension."""
        return int(self.get_model_matrices()[1].shape[1])


class MicroGridSystem(StateSpaceSystem):
    """HKUST battery model used only inside MPC.

    The default model represents the planned 215 kWh / 50 kW installation at a
    15-minute interval. Energy/SOC values are in kWh, power values are in kW,
    and ``sample_time`` is in hours. Inputs are ordered as non-negative
    ``[charge_kw, discharge_kw]``::

        soc_kwh[t+1] = (1 - self_discharge) * soc_kwh[t]
                         + dt * eta_charge * charge_kw[t]
                         - dt / eta_discharge * discharge_kw[t]

    This class exposes MPC matrices and limits only. It has no current-SOC
    member, ``step`` method, or reset method. Realised SOC is managed solely by
    :class:`src.plants.pymgrid_adapter.PymgridAdapter`.

    ``site_id`` and the original public attributes ``charge_min_max``,
    ``storage_power_min_max`` and ``efficiency`` are retained for source-level
    compatibility. ``site_id`` no longer selects ``data/site_*/storage_data``;
    all battery parameters are explicit and independent of the working
    directory.
    """

    DEFAULT_BATTERY_CAPACITY_KWH = 215.0
    DEFAULT_BATTERY_POWER_KW = 50.0
    DEFAULT_SAMPLE_TIME_HOURS = 0.25
    DEFAULT_ROUNDTRIP_EFFICIENCY = 0.92
    DEFAULT_ONE_WAY_EFFICIENCY = DEFAULT_ROUNDTRIP_EFFICIENCY**0.5
    DEFAULT_SOC_MIN_FRACTION = 0.10
    DEFAULT_SOC_MAX_FRACTION = 0.90
    DEFAULT_INITIAL_SOC_FRACTION = 0.50

    def __init__(
        self,
        site_id: Optional[int] = None,
        self_discharge: float = 0.0,
        grid_power_min_max: Sequence[float] = (-4000.0, 4000.0),
        *,
        sample_time: float = DEFAULT_SAMPLE_TIME_HOURS,
        initial_state: Optional[Sequence[float] | float] = None,
        battery_capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
        battery_power_kw: float = DEFAULT_BATTERY_POWER_KW,
        charge_efficiency: float = DEFAULT_ONE_WAY_EFFICIENCY,
        discharge_efficiency: Optional[float] = None,
        min_soc_fraction: float = DEFAULT_SOC_MIN_FRACTION,
        max_soc_fraction: float = DEFAULT_SOC_MAX_FRACTION,
        initial_soc_fraction: float = DEFAULT_INITIAL_SOC_FRACTION,
    ) -> None:
        # Kept as metadata for legacy callers; it has no effect on parameters.
        self.site_id = site_id

        self.battery_capacity_kwh = self._positive(
            "battery_capacity_kwh", battery_capacity_kwh
        )
        self.battery_power_kw = self._positive(
            "battery_power_kw", battery_power_kw
        )
        self.charge_efficiency = self._efficiency(
            "charge_efficiency", charge_efficiency
        )
        self.discharge_efficiency = self._efficiency(
            "discharge_efficiency",
            charge_efficiency
            if discharge_efficiency is None
            else discharge_efficiency,
        )
        # Original code exposed one symmetric ``efficiency`` attribute.
        self.efficiency = self.charge_efficiency

        self._self_discharge = float(self_discharge)
        if not 0.0 <= self._self_discharge < 1.0:
            raise ValueError("self_discharge must be in [0, 1) per step.")

        self.min_soc_fraction = float(min_soc_fraction)
        self.max_soc_fraction = float(max_soc_fraction)
        self.initial_soc_fraction = float(initial_soc_fraction)
        if not 0.0 <= self.min_soc_fraction < self.max_soc_fraction <= 1.0:
            raise ValueError(
                "SOC limits must satisfy 0 <= min_soc_fraction < "
                "max_soc_fraction <= 1."
            )
        if not self.min_soc_fraction <= self.initial_soc_fraction <= self.max_soc_fraction:
            raise ValueError(
                "initial_soc_fraction must lie within the configured SOC limits."
            )

        self.charge_min_max = [
            self.battery_capacity_kwh * self.min_soc_fraction,
            self.battery_capacity_kwh * self.max_soc_fraction,
        ]
        self.storage_power_min_max = [
            -self.battery_power_kw,
            self.battery_power_kw,
        ]
        self.grid_power_min_max = list(
            self._signed_bounds("grid_power_min_max", grid_power_min_max)
        )

        if initial_state is None:
            initial_soc_kwh = self.battery_capacity_kwh * self.initial_soc_fraction
        elif isinstance(initial_state, Sequence) and not isinstance(
            initial_state, (str, bytes)
        ):
            if len(initial_state) != 1:
                raise ValueError("initial_state must contain exactly one SOC value.")
            initial_soc_kwh = float(initial_state[0])
        else:
            initial_soc_kwh = float(initial_state)
        if not self.charge_min_max[0] <= initial_soc_kwh <= self.charge_min_max[1]:
            raise ValueError(
                "initial_state must lie within the SOC energy bounds "
                f"{tuple(self.charge_min_max)} kWh."
            )
        # An explicit legacy ``initial_state`` takes precedence over the
        # fraction argument; keep the public fraction consistent with the
        # actual initial model value.
        self.initial_soc_fraction = initial_soc_kwh / self.battery_capacity_kwh

        super().__init__(sample_time=sample_time, initial_state=[initial_soc_kwh])

    @staticmethod
    def _positive(name: str, value: float) -> float:
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        return value

    @staticmethod
    def _efficiency(name: str, value: float) -> float:
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must lie in (0, 1].")
        return value

    @staticmethod
    def _signed_bounds(name: str, values: Sequence[float]) -> Tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        lower, upper = float(values[0]), float(values[1])
        if lower >= upper or lower > 0.0 or upper < 0.0:
            raise ValueError(
                f"{name} must be ordered and contain zero; got ({lower}, {upper})."
            )
        return lower, upper

    @property
    def sample_time_hours(self) -> float:
        return self._sample_time

    @property
    def self_discharge_per_step(self) -> float:
        return self._self_discharge

    @property
    def manages_physical_soc(self) -> bool:
        """Always false: ``PymgridAdapter`` owns realised SOC."""
        return False

    @property
    def soc_min_max_kwh(self) -> Tuple[float, float]:
        return tuple(self.charge_min_max)

    @property
    def storage_power_min_max_kw(self) -> Tuple[float, float]:
        return tuple(self.storage_power_min_max)

    @property
    def grid_power_min_max_kw(self) -> Tuple[float, float]:
        return tuple(self.grid_power_min_max)

    def get_model_bounds(self) -> Mapping[str, Tuple[float, float]]:
        """Return immutable-name MPC bounds with units encoded in each key."""
        return {
            "soc_kwh": self.soc_min_max_kwh,
            "battery_power_kw": self.storage_power_min_max_kw,
            "grid_power_kw": self.grid_power_min_max_kw,
        }

    def get_model_matrices(
        self, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 15-minute SOC prediction matrices in float64."""
        A_matrix = torch.tensor(
            [[1.0 - self._self_discharge]], dtype=torch.float64, device=device
        )
        B_matrix = torch.tensor(
            [
                [
                    self._sample_time * self.charge_efficiency,
                    -self._sample_time / self.discharge_efficiency,
                ]
            ],
            dtype=torch.float64,
            device=device,
        )
        return A_matrix, B_matrix

    def _get_plant_matrices(
        self, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy alias for :meth:`get_model_matrices`."""
        return self.get_model_matrices(device)

    def predict_next_soc(
        self, soc_kwh: torch.Tensor, charge_discharge_kw: torch.Tensor
    ) -> torch.Tensor:
        """Apply one internal prediction step without retaining any state."""
        A_matrix, B_matrix = self.get_model_matrices(soc_kwh.device)
        return torch.matmul(A_matrix, soc_kwh) + torch.matmul(
            B_matrix, charge_discharge_kw
        )

    def compute_Pg(
        self,
        Ps_in: torch.Tensor,
        Ps_out: torch.Tensor,
        Pr: torch.Tensor,
        Pl: torch.Tensor,
    ) -> torch.Tensor:
        """Return legacy signed grid power ``PV-charge+discharge-load`` in kW.

        Positive values export and negative values import. New resilience code
        uses named non-negative ``grid_import`` and ``grid_export`` decisions.
        """
        return Pr - Ps_in + Ps_out - Pl

    @staticmethod
    def add_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add explicit HKUST model parameters to a legacy argument parser."""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--self_discharge", type=float, default=0.0)
        parser.add_argument("--initial_state", type=float, default=None)
        parser.add_argument(
            "--grid_power_min_max",
            type=float,
            nargs=2,
            default=[-4000.0, 4000.0],
            metavar=("IMPORT_LIMIT_NEG_KW", "EXPORT_LIMIT_KW"),
        )
        parser.add_argument("--sample_time", type=float, default=0.25)
        parser.add_argument("--battery_capacity_kwh", type=float, default=215.0)
        parser.add_argument("--battery_power_kw", type=float, default=50.0)
        parser.add_argument(
            "--charge_efficiency",
            type=float,
            default=MicroGridSystem.DEFAULT_ONE_WAY_EFFICIENCY,
        )
        parser.add_argument("--discharge_efficiency", type=float, default=None)
        parser.add_argument("--min_soc_fraction", type=float, default=0.10)
        parser.add_argument("--max_soc_fraction", type=float, default=0.90)
        parser.add_argument("--initial_soc_fraction", type=float, default=0.50)
        return parser


__all__ = ["MicroGridSystem", "StateSpaceSystem"]

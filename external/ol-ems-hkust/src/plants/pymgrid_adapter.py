"""Adapter between the EMS power convention and :mod:`pymgrid`.

The controller works in kW and exposes four non-negative flow magnitudes:
``battery_charge``, ``battery_discharge``, ``grid_import`` and
``grid_export``.  ``pymgrid`` instead expects one signed *energy* action per
module: positive values provide energy to the microgrid and negative values
absorb energy.  This adapter owns that conversion, constrains the battery
command to what can physically be executed, and derives actual grid exchange
from the measured power balance.  Planned grid exchange is retained in the
log but is never allowed to create artificial balancing energy.

All values returned by :meth:`PymgridAdapter.step` are scalar so a sequence of
step results can be passed directly to ``pandas.DataFrame``.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
from typing import Any

import numpy as np
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, GridModule, LoadModule, RenewableModule


class PymgridAdapter:
    """Run the first action of an EMS plan against measured microgrid data.

    Parameters are expressed in engineering units.  Power limits are in kW,
    energy/capacity values are in kWh, and prices/costs are per kWh.  The
    battery ``efficiency`` is the one-way efficiency used by ``pymgrid`` for
    both charging and discharging.

    The nominal battery defaults represent the planned floating-building
    installation (215 kWh / 50 kW).  ``soc_min`` and ``soc_max`` are fractions
    of nominal capacity.  The adapter reports SOC both in kWh and as a
    fraction of the nominal 215 kWh capacity.

    Separate critical and non-critical lost-load costs can be supplied for
    resilience accounting.  ``loss_load_cost_per_kwh`` remains the pymgrid
    balancing-module fallback and the default for either omitted tier cost.

    Notes
    -----
    ``pymgrid`` time-series modules consume energy per simulation step rather
    than instantaneous power.  Therefore this adapter multiplies measured and
    planned kW values by ``dt_hours`` before calling ``Microgrid.step`` and
    converts the resulting dispatch back to kW for its log.
    """

    PLAN_KEYS = frozenset(
        {"battery_charge", "battery_discharge", "grid_import", "grid_export"}
    )
    ACTUAL_KEYS = frozenset(
        {
            "pv",
            "load",
            "import_price",
            "export_price",
            "grid_available",
            "critical_load",
        }
    )

    def __init__(
        self,
        *,
        dt_hours: float = 0.25,
        battery_capacity_kwh: float = 215.0,
        battery_power_kw: float = 50.0,
        efficiency: float = 0.95,
        initial_soc: float = 0.5,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        grid_import_limit_kw: float = 4000.0,
        grid_export_limit_kw: float = 4000.0,
        battery_cost_per_kwh: float = 0.0,
        loss_load_cost_per_kwh: float = 10.0,
        critical_loss_load_cost_per_kwh: float | None = None,
        noncritical_loss_load_cost_per_kwh: float | None = None,
        pv_curtailment_cost_per_kwh: float = 0.0,
        overgeneration_cost_per_kwh: float = 0.0,
        numerical_tolerance: float = 1e-6,
    ) -> None:
        self.dt_hours = self._positive("dt_hours", dt_hours)
        self.battery_capacity_kwh = self._positive(
            "battery_capacity_kwh", battery_capacity_kwh
        )
        self.battery_power_kw = self._nonnegative(
            "battery_power_kw", battery_power_kw
        )
        self.efficiency = self._as_float("efficiency", efficiency)
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("efficiency must be in (0, 1].")

        self.soc_min = self._as_float("soc_min", soc_min)
        self.soc_max = self._as_float("soc_max", soc_max)
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError("SOC limits must satisfy 0 <= soc_min < soc_max <= 1.")

        self.initial_soc = self._as_float("initial_soc", initial_soc)
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial_soc must lie between soc_min and soc_max.")

        self.grid_import_limit_kw = self._nonnegative(
            "grid_import_limit_kw", grid_import_limit_kw
        )
        self.grid_export_limit_kw = self._nonnegative(
            "grid_export_limit_kw", grid_export_limit_kw
        )
        self.battery_cost_per_kwh = self._nonnegative(
            "battery_cost_per_kwh", battery_cost_per_kwh
        )
        self.loss_load_cost_per_kwh = self._nonnegative(
            "loss_load_cost_per_kwh", loss_load_cost_per_kwh
        )
        self.critical_loss_load_cost_per_kwh = self._nonnegative(
            "critical_loss_load_cost_per_kwh",
            self.loss_load_cost_per_kwh
            if critical_loss_load_cost_per_kwh is None
            else critical_loss_load_cost_per_kwh,
        )
        self.noncritical_loss_load_cost_per_kwh = self._nonnegative(
            "noncritical_loss_load_cost_per_kwh",
            self.loss_load_cost_per_kwh
            if noncritical_loss_load_cost_per_kwh is None
            else noncritical_loss_load_cost_per_kwh,
        )
        self.pv_curtailment_cost_per_kwh = self._nonnegative(
            "pv_curtailment_cost_per_kwh", pv_curtailment_cost_per_kwh
        )
        self.overgeneration_cost_per_kwh = self._nonnegative(
            "overgeneration_cost_per_kwh", overgeneration_cost_per_kwh
        )
        self.numerical_tolerance = self._nonnegative(
            "numerical_tolerance", numerical_tolerance
        )

        self._initial_soc_kwh = self.initial_soc * self.battery_capacity_kwh
        self._soc_kwh = self._initial_soc_kwh
        self._step = 0
        self._microgrid: Microgrid | None = None
        self._history: list[dict[str, float | int | bool]] = []

    @staticmethod
    def _as_float(name: str, value: Any) -> float:
        """Return a finite scalar float, accepting NumPy/Torch scalar objects."""
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, RuntimeError):
                pass
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}.")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite, got {result}.")
        return result

    @classmethod
    def _positive(cls, name: str, value: Any) -> float:
        result = cls._as_float(name, value)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive, got {result}.")
        return result

    @classmethod
    def _nonnegative(cls, name: str, value: Any) -> float:
        result = cls._as_float(name, value)
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative, got {result}.")
        return result

    def _flow_kw(self, name: str, value: Any) -> float:
        """Validate a planned non-negative flow and remove solver-sized noise."""
        result = self._as_float(name, value)
        if result < -self.numerical_tolerance:
            raise ValueError(
                f"plan['{name}'] must be a non-negative kW flow, got {result}."
            )
        return max(0.0, result)

    @property
    def current_step(self) -> int:
        """Number assigned to the next physical step."""
        return self._step

    @property
    def microgrid(self) -> Microgrid | None:
        """The ``pymgrid`` object used for the most recently completed step."""
        return self._microgrid

    @property
    def history(self) -> list[dict[str, float | int | bool]]:
        """Return a copy of the accumulated scalar step logs."""
        return deepcopy(self._history)

    def get_soc_kwh(self) -> float:
        """Return current physical battery energy in kWh."""
        return float(self._soc_kwh)

    def get_soc_fraction(self) -> float:
        """Return SOC as a fraction of nominal battery capacity."""
        return self.get_soc_kwh() / self.battery_capacity_kwh

    def reset(
        self, soc_kwh: float | None = None, step: int = 0
    ) -> dict[str, float | int]:
        """Reset SOC, the external step number, and accumulated logs.

        If ``soc_kwh`` is omitted, the constructor's ``initial_soc`` is used.
        ``step`` can be non-zero when resuming a dated simulation segment.
        """
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("step must be a non-negative integer.")

        reset_soc = self._initial_soc_kwh if soc_kwh is None else self._as_float(
            "soc_kwh", soc_kwh
        )
        self._validate_soc_kwh(reset_soc)

        self._soc_kwh = reset_soc
        self._step = int(step)
        self._microgrid = None
        self._history.clear()
        return {
            "step": self._step,
            "soc_kwh": self.get_soc_kwh(),
            "soc_fraction": self.get_soc_fraction(),
        }

    def _validate_soc_kwh(self, soc_kwh: float) -> None:
        lower = self.soc_min * self.battery_capacity_kwh
        upper = self.soc_max * self.battery_capacity_kwh
        outside_lower_bound = soc_kwh < lower - self.numerical_tolerance
        outside_upper_bound = soc_kwh > upper + self.numerical_tolerance
        if outside_lower_bound or outside_upper_bound:
            raise ValueError(
                f"soc_kwh must be within [{lower}, {upper}], got {soc_kwh}."
            )

    def _build_microgrid(
        self,
        *,
        pv_kwh: float,
        load_kwh: float,
        import_price: float,
        export_price: float,
        grid_available: float,
    ) -> Microgrid:
        """Build a one-step simulator around the current persistent SOC."""
        # Two equal samples are needed because time-series modules form their
        # post-step observation from the following sample.
        load = LoadModule(np.array([load_kwh, load_kwh], dtype=float))
        renewable = RenewableModule(np.array([pv_kwh, pv_kwh], dtype=float))

        # BatteryModule's max_charge/max_discharge are internal battery-energy
        # changes.  These choices make its external AC action bounds exactly
        # +/- battery_power_kw * dt_hours after applying one-way efficiency.
        ac_limit_kwh = self.battery_power_kw * self.dt_hours
        battery = BatteryModule(
            min_capacity=self.soc_min * self.battery_capacity_kwh,
            max_capacity=self.soc_max * self.battery_capacity_kwh,
            max_charge=ac_limit_kwh * self.efficiency,
            max_discharge=ac_limit_kwh / self.efficiency,
            efficiency=self.efficiency,
            battery_cost_cycle=self.battery_cost_per_kwh,
            init_charge=self._soc_kwh,
            raise_errors=False,
        )

        # GridModule only accepts a binary status.  Continuous availability is
        # represented by derating the import/export capacities while keeping
        # the module connected whenever the factor is greater than zero.
        grid_status = int(grid_available > 0.0)
        grid_series = np.array(
            [
                [import_price, export_price, 0.0, grid_status],
                [import_price, export_price, 0.0, grid_status],
            ],
            dtype=float,
        )
        grid = GridModule(
            max_import=(
                self.grid_import_limit_kw * grid_available * self.dt_hours
            ),
            max_export=(
                self.grid_export_limit_kw * grid_available * self.dt_hours
            ),
            time_series=grid_series,
            raise_errors=False,
        )

        return Microgrid(
            [load, renewable, battery, grid],
            loss_load_cost=self.loss_load_cost_per_kwh,
            overgeneration_cost=self.overgeneration_cost_per_kwh,
        )

    @staticmethod
    def _first_module_info(info: Mapping[str, Any], module_name: str) -> Mapping[str, Any]:
        entries = info.get(module_name, ())
        if not entries:
            return {}
        return entries[0]

    @staticmethod
    def _info_energy(module_info: Mapping[str, Any], key: str) -> float:
        return float(module_info.get(key, 0.0))

    def _bounded_battery_dispatch(
        self,
        *,
        requested_net_battery_kw: float,
        pv_kw: float,
        load_kw: float,
        grid_available: float,
    ) -> tuple[float, float, float, float]:
        """Return executable charge/discharge and their dynamic AC limits.

        Positive ``requested_net_battery_kw`` means discharge.  The command is
        clipped by inverter power, current SOC and the energy that the rest of
        the microgrid can supply or absorb.  In an island this additionally
        means charging only from PV surplus and discharging only into the load
        deficit.  These restrictions prevent a battery command from being
        fulfilled by pymgrid's synthetic balancing module.
        """
        minimum_soc_kwh = self.soc_min * self.battery_capacity_kwh
        maximum_soc_kwh = self.soc_max * self.battery_capacity_kwh
        charge_room_kwh = max(0.0, maximum_soc_kwh - self._soc_kwh)
        discharge_room_kwh = max(0.0, self._soc_kwh - minimum_soc_kwh)

        soc_charge_limit_kw = charge_room_kwh / (
            self.efficiency * self.dt_hours
        )
        soc_discharge_limit_kw = (
            discharge_room_kwh * self.efficiency / self.dt_hours
        )
        charge_limit_kw = min(self.battery_power_kw, soc_charge_limit_kw)
        discharge_limit_kw = min(self.battery_power_kw, soc_discharge_limit_kw)

        effective_import_limit_kw = self.grid_import_limit_kw * grid_available
        effective_export_limit_kw = self.grid_export_limit_kw * grid_available
        if grid_available == 0.0:
            # Preserve load before charging, and use PV before discharging.
            charge_supply_kw = max(0.0, pv_kw - load_kw)
            discharge_sink_kw = max(0.0, load_kw - pv_kw)
        else:
            # Match the project baseline: charging must remain attributable to
            # actual PV, while the grid may serve the coincident load. It also
            # must not worsen load shedding when the interconnection is tight.
            charge_supply_kw = min(
                pv_kw,
                max(0.0, pv_kw + effective_import_limit_kw - load_kw),
            )
            discharge_sink_kw = load_kw + effective_export_limit_kw

        charge_limit_kw = min(charge_limit_kw, charge_supply_kw)
        discharge_limit_kw = min(discharge_limit_kw, discharge_sink_kw)

        if requested_net_battery_kw < 0.0:
            charge_kw = min(-requested_net_battery_kw, charge_limit_kw)
            discharge_kw = 0.0
        else:
            charge_kw = 0.0
            discharge_kw = min(requested_net_battery_kw, discharge_limit_kw)

        return charge_kw, discharge_kw, charge_limit_kw, discharge_limit_kw

    def step(
        self, plan: Mapping[str, Any], actual: Mapping[str, Any]
    ) -> dict[str, float | int | bool]:
        """Execute one measured 15-minute interval with ``Microgrid.step``.

        ``plan`` contains the first MPC action.  Each of its four keys is a
        non-negative kW magnitude.  If charging and discharging are both
        non-zero, their net battery flow is requested.  Planned grid import and
        export are retained for forecast-versus-actual analysis; actual grid
        power is instead the availability-limited residual of measured PV,
        load and the bounded battery command.

        ``actual`` must provide measured ``pv`` and ``load`` in kW, import and
        export prices per kWh, ``grid_available`` in ``[0, 1]``, and
        ``critical_load`` in kW.  A fractional availability derates both grid
        limits.  Lost load is assigned to non-critical demand first; only the
        remaining shortage is counted as critical lost load.
        """
        if not isinstance(plan, Mapping):
            raise TypeError("plan must be a mapping.")
        if not isinstance(actual, Mapping):
            raise TypeError("actual must be a mapping.")

        missing_actual = self.ACTUAL_KEYS.difference(actual)
        if missing_actual:
            raise KeyError(f"actual is missing required keys: {sorted(missing_actual)}")

        requested = {
            key: self._flow_kw(key, plan.get(key, 0.0)) for key in self.PLAN_KEYS
        }
        pv_kw = self._nonnegative("actual['pv']", actual["pv"])
        load_kw = self._nonnegative("actual['load']", actual["load"])
        critical_load_kw = self._nonnegative(
            "actual['critical_load']", actual["critical_load"]
        )
        if critical_load_kw > load_kw + self.numerical_tolerance:
            raise ValueError("actual['critical_load'] cannot exceed actual['load'].")
        critical_load_kw = min(critical_load_kw, load_kw)

        import_price = self._nonnegative(
            "actual['import_price']", actual["import_price"]
        )
        export_price = self._nonnegative(
            "actual['export_price']", actual["export_price"]
        )
        grid_raw = actual["grid_available"]
        if isinstance(grid_raw, (bool, np.bool_)):
            grid_available = float(grid_raw)
        else:
            grid_available = self._as_float(
                "actual['grid_available']", grid_raw
            )
        if not 0.0 <= grid_available <= 1.0:
            raise ValueError("actual['grid_available'] must lie in [0, 1].")

        pv_kwh = pv_kw * self.dt_hours
        load_kwh = load_kw * self.dt_hours
        critical_load_kwh = critical_load_kw * self.dt_hours

        requested_net_battery_kw = (
            requested["battery_discharge"] - requested["battery_charge"]
        )
        requested_net_grid_kw = (
            requested["grid_import"] - requested["grid_export"]
        )
        (
            bounded_battery_charge_kw,
            bounded_battery_discharge_kw,
            battery_charge_limit_kw,
            battery_discharge_limit_kw,
        ) = self._bounded_battery_dispatch(
            requested_net_battery_kw=requested_net_battery_kw,
            pv_kw=pv_kw,
            load_kw=load_kw,
            grid_available=grid_available,
        )

        # Positive residual imports from the grid; negative residual exports.
        grid_balance_request_kw = (
            load_kw
            + bounded_battery_charge_kw
            - pv_kw
            - bounded_battery_discharge_kw
        )
        effective_import_limit_kw = self.grid_import_limit_kw * grid_available
        effective_export_limit_kw = self.grid_export_limit_kw * grid_available
        balanced_grid_import_kw = min(
            max(grid_balance_request_kw, 0.0), effective_import_limit_kw
        )
        balanced_grid_export_kw = min(
            max(-grid_balance_request_kw, 0.0), effective_export_limit_kw
        )

        # Pymgrid sign convention: source is positive; sink is negative.  Its
        # actions are energies per step, hence the multiplication by dt.
        control = {
            "battery": [
                (
                    bounded_battery_discharge_kw
                    - bounded_battery_charge_kw
                )
                * self.dt_hours
            ],
            "grid": [
                (balanced_grid_import_kw - balanced_grid_export_kw)
                * self.dt_hours
            ],
        }

        soc_before_kwh = self.get_soc_kwh()
        microgrid = self._build_microgrid(
            pv_kwh=pv_kwh,
            load_kwh=load_kwh,
            import_price=import_price,
            export_price=export_price,
            grid_available=grid_available,
        )
        _, reward, done, info = microgrid.step(control, normalized=False)

        battery_info = self._first_module_info(info, "battery")
        grid_info = self._first_module_info(info, "grid")
        renewable_info = self._first_module_info(info, "renewable")
        balancing_info = self._first_module_info(info, "balancing")

        battery_discharge_kwh = self._info_energy(battery_info, "provided_energy")
        battery_charge_kwh = self._info_energy(battery_info, "absorbed_energy")
        grid_import_kwh = self._info_energy(grid_info, "provided_energy")
        grid_export_kwh = self._info_energy(grid_info, "absorbed_energy")
        pv_used_kwh = self._info_energy(renewable_info, "provided_energy")
        pv_curtailed_kwh = self._info_energy(renewable_info, "curtailment")
        loss_load_kwh = self._info_energy(balancing_info, "provided_energy")
        overgeneration_kwh = self._info_energy(balancing_info, "absorbed_energy")

        energy_tolerance = max(
            self.numerical_tolerance * self.dt_hours,
            np.finfo(float).eps * 100,
        )
        if loss_load_kwh > load_kwh + energy_tolerance:
            raise RuntimeError(
                "pymgrid loss_load exceeds measured load; a controllable sink "
                "was incorrectly supplied by balancing energy."
            )
        if overgeneration_kwh > energy_tolerance:
            raise RuntimeError(
                "pymgrid overgeneration is non-zero after physical dispatch "
                "sanitisation; a controllable source has no physical sink."
            )
        loss_load_kwh = min(max(loss_load_kwh, 0.0), load_kwh)
        overgeneration_kwh = max(overgeneration_kwh, 0.0)

        battery_module = microgrid.modules.battery.item()
        soc_after_kwh = float(battery_module.current_charge)
        self._validate_soc_kwh(soc_after_kwh)
        self._soc_kwh = soc_after_kwh
        self._microgrid = microgrid

        noncritical_load_kwh = max(0.0, load_kwh - critical_load_kwh)
        noncritical_loss_load_kwh = min(loss_load_kwh, noncritical_load_kwh)
        critical_loss_load_kwh = min(
            critical_load_kwh,
            max(0.0, loss_load_kwh - noncritical_loss_load_kwh),
        )
        load_served_kwh = max(0.0, load_kwh - loss_load_kwh)

        battery_cost = (
            battery_charge_kwh + battery_discharge_kwh
        ) * self.battery_cost_per_kwh
        grid_cost = grid_import_kwh * import_price - grid_export_kwh * export_price
        critical_loss_load_cost = (
            critical_loss_load_kwh * self.critical_loss_load_cost_per_kwh
        )
        noncritical_loss_load_cost = (
            noncritical_loss_load_kwh * self.noncritical_loss_load_cost_per_kwh
        )
        loss_load_cost = critical_loss_load_cost + noncritical_loss_load_cost
        pv_curtailment_cost = (
            pv_curtailed_kwh * self.pv_curtailment_cost_per_kwh
        )
        overgeneration_cost = overgeneration_kwh * self.overgeneration_cost_per_kwh
        component_cost_sum = (
            battery_cost
            + grid_cost
            + loss_load_cost
            + pv_curtailment_cost
            + overgeneration_cost
        )

        to_kw = lambda energy_kwh: energy_kwh / self.dt_hours
        result: dict[str, float | int | bool] = {
            "step": self._step,
            "dt_hours": self.dt_hours,
            "grid_available": grid_available,
            "grid_connected": bool(grid_available > 0.0),
            "done": bool(done),
            "requested_battery_charge_kw": requested["battery_charge"],
            "requested_battery_discharge_kw": requested["battery_discharge"],
            "requested_grid_import_kw": requested["grid_import"],
            "requested_grid_export_kw": requested["grid_export"],
            "requested_net_battery_kw": requested_net_battery_kw,
            "requested_net_grid_kw": requested_net_grid_kw,
            "battery_charge_limit_kw": battery_charge_limit_kw,
            "battery_discharge_limit_kw": battery_discharge_limit_kw,
            "bounded_battery_charge_kw": bounded_battery_charge_kw,
            "bounded_battery_discharge_kw": bounded_battery_discharge_kw,
            "grid_balance_request_kw": grid_balance_request_kw,
            "effective_grid_import_limit_kw": effective_import_limit_kw,
            "effective_grid_export_limit_kw": effective_export_limit_kw,
            "soc_before_kwh": soc_before_kwh,
            "soc_after_kwh": soc_after_kwh,
            "soc_before_fraction": soc_before_kwh / self.battery_capacity_kwh,
            "soc_after_fraction": soc_after_kwh / self.battery_capacity_kwh,
            "battery_charge_kw": to_kw(battery_charge_kwh),
            "battery_discharge_kw": to_kw(battery_discharge_kwh),
            "grid_import_kw": to_kw(grid_import_kwh),
            "grid_export_kw": to_kw(grid_export_kwh),
            "pv_available_kw": pv_kw,
            "pv_used_kw": to_kw(pv_used_kwh),
            "pv_curtailed_kw": to_kw(pv_curtailed_kwh),
            "load_kw": load_kw,
            "critical_load_kw": critical_load_kw,
            "noncritical_load_kw": load_kw - critical_load_kw,
            "load_served_kw": to_kw(load_served_kwh),
            "loss_load_kw": to_kw(loss_load_kwh),
            "critical_loss_load_kw": to_kw(critical_loss_load_kwh),
            "noncritical_loss_load_kw": to_kw(noncritical_loss_load_kwh),
            "overgeneration_kw": to_kw(overgeneration_kwh),
            "battery_charge_kwh": battery_charge_kwh,
            "battery_discharge_kwh": battery_discharge_kwh,
            "grid_import_kwh": grid_import_kwh,
            "grid_export_kwh": grid_export_kwh,
            "pv_available_kwh": pv_kwh,
            "pv_used_kwh": pv_used_kwh,
            "pv_curtailed_kwh": pv_curtailed_kwh,
            "load_kwh": load_kwh,
            "critical_load_kwh": critical_load_kwh,
            "noncritical_load_kwh": noncritical_load_kwh,
            "load_served_kwh": load_served_kwh,
            "loss_load_kwh": loss_load_kwh,
            "critical_loss_load_kwh": critical_loss_load_kwh,
            "noncritical_loss_load_kwh": noncritical_loss_load_kwh,
            "overgeneration_kwh": overgeneration_kwh,
            "import_price_per_kwh": import_price,
            "export_price_per_kwh": export_price,
            "battery_cost": battery_cost,
            "grid_cost": grid_cost,
            "loss_load_cost": loss_load_cost,
            "critical_loss_load_cost": critical_loss_load_cost,
            "noncritical_loss_load_cost": noncritical_loss_load_cost,
            "pv_curtailment_cost": pv_curtailment_cost,
            "overgeneration_cost": overgeneration_cost,
            "component_cost_sum": component_cost_sum,
            "pymgrid_cost": -float(reward),
            "pymgrid_reward": float(reward),
            "cost": component_cost_sum,
            "reward": -component_cost_sum,
        }
        self._history.append(result.copy())
        self._step += 1
        return result


__all__ = ["PymgridAdapter"]

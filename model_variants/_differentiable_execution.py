"""Torch form of the existing M0 first-action execution and energy ledger.

The physics matches ``m0_scenarios.execute`` and ``PymgridAdapter.step``:
scheduled loads are removed first, battery commands are only clipped, and
the grid balances the measured residual. No extra battery dispatch or load
restoration is introduced. The min/max operations have piecewise gradients;
a saturated command can correctly have zero gradient.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys

import torch
from torch import Tensor

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
from m0_scenarios import C, DT, TIERS


def execute_tensor(
    arrays: Mapping[str, Tensor],
    actual: Mapping[str, float | Tensor],
    tiers: Mapping[str, float | Tensor],
    initial_soc_kwh: float | Tensor,
    *,
    hardware=C,
) -> dict[str, Tensor]:
    """Execute one first action, retaining gradients to actions and initial SOC.

    ``arrays`` uses the ScenarioMPC names (charge, discharge, grid_import,
    grid_export, unmet_cl1/2/3), each a scalar or an unbatched horizon vector.
    ``actual`` supplies measured pv, grid_available, import_price and
    export_price. As in the reference executor, the tier sum defines load.
    Return values are scalar tensors on the charge action's device/dtype.

    ``economic_loss`` contains grid bill, AC battery throughput cost and PV
    curtailment cost only. ENS and penalty-inclusive ``component_cost_sum``
    are separate diagnostics and must not be substituted for the M1 loss.
    This reproduces the current adapter's zero-self-discharge physics; it
    does not add self-discharge or change the underlying execution policy.
    """
    if hardware.dt_hours != DT:
        raise ValueError("This scenario executor requires the shared 15-minute time step")
    anchor = arrays["charge"]
    if not isinstance(anchor, Tensor):
        anchor = initial_soc_kwh if isinstance(initial_soc_kwh, Tensor) else torch.tensor(0., dtype=torch.float64)
    if not anchor.is_floating_point():
        raise TypeError("Action tensors must have a floating-point dtype")
    zero = anchor.new_zeros(())

    def scalar(value, name):
        out = torch.as_tensor(value, dtype=anchor.dtype, device=anchor.device)
        if out.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        out = out.reshape(())
        if not torch.isfinite(out):
            raise ValueError(f"{name} must be finite")
        return out

    def first(key, *, flow=False):
        out = torch.as_tensor(arrays[key], dtype=anchor.dtype, device=anchor.device)
        if out.ndim > 1 or out.numel() < 1:
            raise ValueError(f"{key} must be a scalar or an unbatched horizon vector")
        out = scalar(out.reshape(-1)[0], key)
        if flow and out < -1e-6:
            raise ValueError(f"{key} must be nonnegative (apart from solver noise)")
        return out.clamp_min(0.) if flow else out

    desired = torch.stack([scalar(tiers[k], k) for k in TIERS])
    if torch.any(desired < 0):
        raise ValueError("Tier demands must be nonnegative")
    scheduled = torch.minimum(torch.stack([first("unmet_" + k) for k in TIERS]).clamp_min(0.), desired)
    admitted = desired - scheduled
    admitted_load = admitted.sum()

    pv = scalar(actual["pv"], "pv")
    grid = scalar(actual["grid_available"], "grid_available")
    buy = scalar(actual["import_price"], "import_price")
    sell = scalar(actual["export_price"], "export_price")
    if pv < 0 or buy < 0 or sell < 0 or grid < 0 or grid > 1:
        raise ValueError("PV/prices must be nonnegative and grid availability in [0, 1]")
    soc = scalar(initial_soc_kwh, "initial_soc_kwh")
    if soc < hardware.min_soc_kWh - 1e-6 or soc > hardware.max_soc_kWh + 1e-6:
        raise ValueError("Initial SOC outside the adapter bounds")

    # The adapter uses charge_efficiency as its common one-way efficiency.
    eta = hardware.charge_efficiency
    dt = hardware.dt_hours
    request = {k: first(k, flow=True) for k in ("charge", "discharge", "grid_import", "grid_export")}
    net_battery = request["discharge"] - request["charge"]
    import_limit = grid * hardware.grid_import_limit_kW
    export_limit = grid * hardware.grid_export_limit_kW
    charge_limit = ((hardware.max_soc_kWh - soc).clamp_min(0.) / (eta * dt)).clamp_max(hardware.battery_power_kW)
    discharge_limit = ((soc - hardware.min_soc_kWh).clamp_min(0.) * eta / dt).clamp_max(hardware.battery_power_kW)
    connected_supply = (pv + import_limit - admitted_load).clamp_min(0.)
    if not hardware.allow_grid_charging:
        connected_supply = torch.minimum(pv, connected_supply)
    charge_supply = torch.where(grid > 0, connected_supply, (pv - admitted_load).clamp_min(0.))
    discharge_sink = torch.where(grid > 0, admitted_load + export_limit, (admitted_load - pv).clamp_min(0.))
    charge_limit = torch.minimum(charge_limit, charge_supply)
    discharge_limit = torch.minimum(discharge_limit, discharge_sink)
    charge = torch.minimum((-net_battery).clamp_min(0.), charge_limit)
    discharge = torch.minimum(net_battery.clamp_min(0.), discharge_limit)

    residual = admitted_load + charge - pv - discharge
    grid_import = torch.minimum(residual.clamp_min(0.), import_limit)
    grid_export = torch.minimum((-residual).clamp_min(0.), export_limit)

    # Pymgrid applies fixed load and controllable battery/grid first, then
    # requests renewable generation for the remaining demand and curtails it.
    renewable_request = admitted_load + charge + grid_export - discharge - grid_import
    pv_used = torch.minimum(pv, renewable_request.clamp_min(0.))
    curtailed = pv - pv_used
    physical_unmet = torch.minimum((renewable_request - pv_used).clamp_min(0.), admitted_load)
    available = (admitted_load - physical_unmet).clamp_min(0.)
    served = []
    for demand in admitted.unbind():
        supply = torch.minimum(demand, available.clamp_min(0.))
        served.append(supply)
        available = available - supply
    served = torch.stack(served)
    unmet = (desired - served).clamp_min(0.)
    next_soc = soc + eta * charge * dt - discharge * dt / eta

    # Field names and AC units match the scalar evaluation ledger.
    flows = dict(
        battery_charge=charge, battery_discharge=discharge,
        grid_import=grid_import, grid_export=grid_export,
        pv_available=pv, pv_used=pv_used, pv_curtailed=curtailed,
        load=desired.sum(), critical_load=desired[:2].sum(), noncritical_load=desired[2],
        load_served=served.sum(), loss_load=unmet.sum(),
        critical_loss_load=unmet[:2].sum(), noncritical_loss_load=unmet[2],
        overgeneration=zero,
    )
    row = {}
    for key, value in flows.items():
        row[key + "_kw"] = value
        row[key + "_kwh"] = value * dt
    row.update(
        dt_hours=zero + dt, grid_available=grid,
        soc_before_kwh=soc, soc_after_kwh=next_soc, next_soc_kwh=next_soc,
        soc_before_fraction=soc / hardware.battery_capacity_kWh,
        soc_after_fraction=next_soc / hardware.battery_capacity_kWh,
        requested_battery_charge_kw=request["charge"], requested_battery_discharge_kw=request["discharge"],
        requested_grid_import_kw=request["grid_import"], requested_grid_export_kw=request["grid_export"],
        requested_net_battery_kw=net_battery,
        requested_net_grid_kw=request["grid_import"] - request["grid_export"],
        bounded_battery_charge_kw=charge, bounded_battery_discharge_kw=discharge,
        battery_charge_limit_kw=charge_limit, battery_discharge_limit_kw=discharge_limit,
        grid_balance_request_kw=residual, effective_grid_import_limit_kw=import_limit,
        effective_grid_export_limit_kw=export_limit,
        import_price_per_kwh=buy, export_price_per_kwh=sell,
    )
    for i, key in enumerate(TIERS):
        row[key + "_load_kwh"] = desired[i] * dt
        row[key + "_served_kwh"] = served[i] * dt
        row[key + "_ens_kwh"] = unmet[i] * dt
        row[key + "_scheduled_shed_kw"] = scheduled[i]
    weights = hardware.objective_weights
    row["battery_throughput_kwh"] = (charge + discharge) * dt
    row["battery_cost"] = row["battery_throughput_kwh"] * weights["battery_cycle"]
    row["grid_cost"] = (grid_import * buy - grid_export * sell) * dt
    row["grid_bill"] = row["grid_cost"]
    row["pv_curtailment_cost"] = curtailed * dt * weights["pv_curtailment"]
    row["economic_loss"] = row["grid_cost"] + row["battery_cost"] + row["pv_curtailment_cost"]
    row["critical_loss_load_cost"] = unmet[:2].sum() * dt * weights["critical_unmet"]
    row["noncritical_loss_load_cost"] = unmet[2] * dt * weights["noncritical_unmet"]
    row["loss_load_cost"] = row["critical_loss_load_cost"] + row["noncritical_loss_load_cost"]
    row["overgeneration_cost"] = zero
    row["component_cost_sum"] = row["economic_loss"] + row["loss_load_cost"]
    row["cost"] = row["component_cost_sum"]
    row["reward"] = -row["cost"]
    return row


__all__ = ["execute_tensor"]

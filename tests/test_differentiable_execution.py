"""Numerical parity with the existing physical executor, plus useful gradients."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model_variants"), str(ROOT / "tools")]
from _differentiable_execution import execute_tensor
from m0_scenarios import C, TIERS, execute
from run_m0_simulation import _build_adapter, verify_execution


def inputs(charge=0., discharge=0., shed=(0., 0., 0.)):
    values = dict(charge=charge, discharge=discharge, grid_import=19., grid_export=3.)
    values.update({"unmet_" + key: value for key, value in zip(TIERS, shed)})
    return {key: torch.full((96,), value, dtype=torch.float64) for key, value in values.items()}


def measured(pv, grid=1., tiers=(2., 2., 2.)):
    return dict(pv=pv, load=sum(tiers), critical_load=sum(tiers[:2]),
                grid_available=grid, import_price=.7, export_price=.36)


def compare_with_reference(arrays, actual, tiers, soc, hardware=C):
    adapter = _build_adapter(hardware)
    adapter.reset(soc_kwh=float(soc))
    reference = execute(adapter, {k: v.detach().numpy() for k, v in arrays.items()}, actual, tiers)
    result = execute_tensor(arrays, actual, tiers, soc, hardware=hardware)
    # Compare every physical/economic ledger field returned by the torch path,
    # rather than checking only a total which could hide cancelling errors.
    for key, value in result.items():
        assert value.ndim == 0 and torch.isfinite(value), key
        if key in reference:
            assert value.item() == pytest.approx(reference[key], abs=2e-10), key
    assert result["economic_loss"].item() == pytest.approx(
        reference["grid_cost"] + reference["battery_cost"] + reference["pv_curtailment_cost"], abs=2e-10)
    verify_execution(reference)
    balance = (result["pv_used_kwh"] + result["battery_discharge_kwh"] + result["grid_import_kwh"]
               - result["load_served_kwh"] - result["battery_charge_kwh"] - result["grid_export_kwh"])
    assert balance.item() == pytest.approx(0., abs=2e-10)
    return result


@pytest.mark.parametrize("charge,discharge,pv,grid,soc,shed,overrides", [
    (6., 0., 0., 1., 100., (0., 0., 0.), {}),                 # night grid charge
    (3., 0., 25., 1., 100., (0., 0., 0.), {}),                # PV plus export
    (3., 0., 25., 1., 100., (0., 0., 0.), {"grid_export_limit_kW": 1.}),
    (10., 0., 0., .5, 100., (0., 0., 0.), {"grid_import_limit_kW": 15.}),
    (10., 0., 0., 1., 100., (0., 0., 0.), {"grid_import_limit_kW": 4.}),
    (60., 0., 0., 1., 100., (0., 0., 0.), {}),                # inverter limit
    (20., 0., 10., 1., 193.45, (0., 0., 0.), {}),             # SOC charge bound
    (0., 20., 0., 0., 21.55, (0., 0., 0.), {}),               # SOC discharge bound
    (0., 1., .7, 0., 100., (0., 0., 0.), {}),                 # low actual PV; no redispatch
    (3., 0., 2., 0., 100., (0., 0., 0.), {}),                 # island charging clipped to zero
    (1., 0., 5., 0., 100., (0., 0., 2.), {}),                 # shed to permit charging
    (0., 0., 15., 0., 100., (0., 0., 2.), {}),                # no restoration despite surplus
    (0., 20., 15., 0., 100., (0., 0., 0.), {}),               # discharge sink is zero
    (0., 10., 0., 0., 21.5, (0., 0., 0.), {}),                # empty battery
    (10., 0., 30., 1., 193.5, (0., 0., 0.), {}),              # full battery
    (4., 1., 0., 1., 100., (0., 0., 0.), {}),                 # net simultaneous requests
    (5., 0., 0., 1., 100., (0., 0., 0.), {"allow_grid_charging": False}),
    (5., 0., 3., 1., 100., (0., 0., 0.), {"allow_grid_charging": False}),
    (0., 0., 0., 0., 100., (3., -1e-7, 1.), {}),              # scheduled shedding clips to demand
])
def test_matches_physical_executor(charge, discharge, pv, grid, soc, shed, overrides):
    compare_with_reference(inputs(charge, discharge, shed), measured(pv, grid),
                           dict(cl1=2., cl2=2., cl3=2.), soc, replace(C, **overrides))


def test_random_actions_match_physical_executor():
    rng = np.random.default_rng(8731)
    for _ in range(50):
        tier_values = rng.uniform(0., 8., size=3)
        tiers = dict(zip(TIERS, tier_values))
        arrays = inputs(*rng.uniform(0., 70., size=2), shed=rng.uniform(0., 1., 3) * tier_values)
        hardware = replace(C, grid_import_limit_kW=float(rng.uniform(0., 20.)),
                           grid_export_limit_kW=float(rng.uniform(0., 20.)),
                           allow_grid_charging=bool(rng.integers(2)))
        actual = measured(float(rng.uniform(0., 60.)), float(rng.choice([0., .5, 1.])), tier_values)
        compare_with_reference(arrays, actual, tiers, float(rng.uniform(21.5, 193.5)), hardware)


def test_preserves_first_action_and_initial_soc_gradients():
    arrays = inputs(charge=3.)
    arrays["charge"].requires_grad_()
    soc = torch.tensor(80., dtype=torch.float64, requires_grad=True)
    row = execute_tensor(arrays, measured(0.), dict(cl1=2., cl2=2., cl3=2.), soc)
    (row["economic_loss"] + row["next_soc_kwh"]).backward()
    assert torch.isfinite(arrays["charge"].grad).all()
    assert arrays["charge"].grad[0] > 0
    assert torch.count_nonzero(arrays["charge"].grad[1:]) == 0
    assert soc.grad.item() == pytest.approx(1.)
    assert arrays["charge"].grad[0].item() == pytest.approx(
        C.dt_hours * (.7 + C.objective_weights["battery_cycle"] + C.charge_efficiency))


def test_shortage_gradient_reaches_discharge_without_automatic_compensation():
    arrays = inputs(discharge=1.)
    arrays["discharge"].requires_grad_()
    row = execute_tensor(arrays, measured(.7, 0.), dict(cl1=2., cl2=2., cl3=2.), 100.)
    assert row["battery_discharge_kw"].item() == pytest.approx(1.)
    assert row["battery_discharge_limit_kw"].item() > 1.
    row["critical_loss_load_kwh"].backward()
    assert arrays["discharge"].grad[0].item() == pytest.approx(-.25)
    assert torch.isfinite(arrays["discharge"].grad).all()


def test_saturation_has_zero_command_gradient_and_finite_soc_gradient():
    arrays = inputs(discharge=20.)
    arrays["discharge"].requires_grad_()
    soc = torch.tensor(21.55, dtype=torch.float64, requires_grad=True)
    row = execute_tensor(arrays, measured(0., 0.), dict(cl1=2., cl2=2., cl3=2.), soc)
    row["critical_loss_load_kwh"].backward()
    assert arrays["discharge"].grad[0].item() == 0.
    assert soc.grad.item() == pytest.approx(-C.discharge_efficiency)


def test_scheduled_load_gradient_and_no_ens_penalty_in_economic_loss():
    arrays = inputs(shed=(0., 0., .4))
    arrays["unmet_cl3"].requires_grad_()
    row = execute_tensor(arrays, measured(0.), dict(cl1=2., cl2=2., cl3=2.), 100.)
    row["economic_loss"].backward()
    assert arrays["unmet_cl3"].grad[0].item() == pytest.approx(-.7 * .25)
    assert row["loss_load_cost"].item() == pytest.approx(100.)
    assert (row["component_cost_sum"] - row["economic_loss"]).item() == pytest.approx(100.)


def test_multistep_soc_graph_is_not_detached():
    arrays = inputs(charge=2.)
    arrays["charge"].requires_grad_()
    tiers = dict(cl1=2., cl2=2., cl3=2.)
    first = execute_tensor(arrays, measured(0.), tiers, 100.)
    second = execute_tensor(inputs(discharge=1.), measured(0., 0.), tiers, first["next_soc_kwh"])
    second["next_soc_kwh"].backward()
    assert arrays["charge"].grad[0].item() == pytest.approx(.25 * C.charge_efficiency)


def test_gradcheck_away_from_nonsmooth_boundaries():
    def loss(values):
        arrays = inputs(charge=2.)
        arrays["charge"] = values[0:1]
        row = execute_tensor(arrays, measured(0.), dict(cl1=2., cl2=2., cl3=2.), values[1])
        return torch.stack([row["economic_loss"], row["next_soc_kwh"]])
    assert torch.autograd.gradcheck(loss, (torch.tensor([2., 100.], dtype=torch.float64, requires_grad=True),))


def test_rejects_accidental_batch_dimension():
    arrays = inputs()
    arrays["charge"] = torch.zeros((1, 96), dtype=torch.float64)
    with pytest.raises(ValueError, match="unbatched"):
        execute_tensor(arrays, measured(0.), dict(cl1=2., cl2=2., cl3=2.), 100.)

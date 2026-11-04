"""Decision/gradient contracts against the current frozen-model controller."""
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model_variants"), str(ROOT / "tools")]
from _differentiable_scenario_mpc import DifferentiableScenarioMPC, reserve_profile_tensor
from m0_scenarios import C, DT, H, TIERS, Scenario, ScenarioMPC, reserve_profile


def future(grid=1):
    result = {key: np.ones(H) for key in TIERS}
    result.update(grid=np.full(H, grid), import_price=np.full(H, .3),
                  export_price=np.full(H, .2))
    return result


@pytest.fixture(scope="module")
def differentiable():
    return DifferentiableScenarioMPC()


@pytest.mark.parametrize("grid,pv", [(1, 1.5), (0, 1.5), (0, 0.0)])
def test_same_priority_objectives_and_first_actions_as_m0(differentiable, grid, pv):
    forecast = torch.full((H,), pv, dtype=torch.float64, requires_grad=True)
    inputs = future(grid)
    reserve = np.full(H + 1, C.min_soc_kWh)
    decisions = differentiable(forecast, inputs, C.min_soc_kWh, reserve)
    expected, violation, _, _ = ScenarioMPC(battery_exclusive=False).solve(
        forecast.detach().numpy(), inputs, C.min_soc_kWh, reserve)
    assert violation < 1e-4
    assert differentiable.last_max_constraint_violation < 1e-4
    assert set(decisions) == set(expected)
    for tier in TIERS:
        # Compare primary optima across the full horizon; LP solvers may choose
        # different allocations among equivalent solutions at individual steps.
        assert DT * decisions["unmet_" + tier].sum().item() == pytest.approx(
            DT * expected["unmet_" + tier].sum(), abs=1e-4)
    for key in ("charge", "discharge", "grid_import", "grid_export"):
        assert decisions[key][0].item() == pytest.approx(expected[key][0], abs=2e-3)
    assert differentiable.last_stage_indices == ((4,) if grid else (0, 1, 2, 4))


@pytest.mark.parametrize("hours,hour", [(6, 0), (168, 0), (168, 35.25), (6, 31)])
def test_reserve_matches_evaluation_and_retains_forecast_gradient(hours, hour):
    case = Scenario("gradient-reserve", "S3", event_hour=24, outage_hours=hours,
                    announced_hours=hours, warning_hours=24)
    forecast = torch.linspace(.1, 1., H, dtype=torch.float64, requires_grad=True)
    load = np.full(H, 3.)
    actual = reserve_profile_tensor(case, hour, forecast, load)
    expected = reserve_profile(case, hour, forecast.detach().numpy(), load)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-12, atol=1e-10)
    gradient, = torch.autograd.grad(actual.sum(), forecast)
    assert torch.isfinite(gradient).all()
    if case.information(hour)[2]:
        assert gradient.abs().sum() > 0
    else:
        assert gradient.abs().sum() == 0


def test_grid_economic_gradient_is_finite_and_matches_energy_units(differentiable):
    forecast = torch.full((H,), 1.5, dtype=torch.float64, requires_grad=True)
    decisions = differentiable(forecast, future(), C.min_soc_kWh,
                               np.full(H + 1, C.min_soc_kWh))
    cost = DT * (.3 * decisions["grid_import"] - .2 * decisions["grid_export"]).sum()
    gradient, = torch.autograd.grad(cost, forecast)
    assert cost.item() == pytest.approx(10.8, abs=1e-5)
    assert torch.isfinite(gradient).all()
    # For an empty battery, fixed tariffs and PV below load, one additional kW
    # at each origin saves 96 * 0.25h * 0.3 CNY/kWh = 7.2 CNY.
    assert gradient.sum().item() == pytest.approx(-7.2, rel=.02)


def test_actual_grid_cost_and_priority_caps_keep_gradient(differentiable):
    case = Scenario("warning", "S2", event_hour=24, outage_hours=6,
                    announced_hours=6, warning_hours=24)
    forecast = torch.full((H,), .1, dtype=torch.float64, requires_grad=True)
    inputs = future()
    reserve = reserve_profile_tensor(case, 0, forecast, np.full(H, 3.))
    decisions = differentiable(forecast, inputs, C.min_soc_kWh, reserve)
    expected, _, _, _ = ScenarioMPC(battery_exclusive=False).solve(
        forecast.detach().numpy(), inputs, C.min_soc_kWh, reserve.detach().numpy())
    for key in ("charge", "discharge", "grid_import", "grid_export"):
        assert decisions[key][0].item() == pytest.approx(expected[key][0], abs=2e-3)
    assert differentiable.last_stage_indices == (3, 4)
    assert differentiable.last_stage_caps[4].requires_grad
    # Actual PV is zero throughout this connected interval. Real grid energy
    # settles the commanded battery power plus real load, not forecast PV.
    actual_net_import = 3.0 + decisions["charge"] - decisions["discharge"]
    actual_cost = DT * (.3 * actual_net_import.clamp_min(0.0)
                       - .2 * (-actual_net_import).clamp_min(0.0)).sum()
    cap_gradient, = torch.autograd.grad(actual_cost, differentiable.last_stage_caps[4],
                                       retain_graph=True)
    pv_gradient, = torch.autograd.grad(actual_cost, forecast)
    assert torch.isfinite(cap_gradient).all() and cap_gradient[3].abs() > 1e-7
    assert torch.isfinite(pv_gradient).all() and pv_gradient.abs().sum() > .1


def test_tier_caps_are_connected_across_all_island_stages(differentiable):
    forecast = torch.full((H,), 1.5, dtype=torch.float64, requires_grad=True)
    decisions = differentiable(forecast, future(grid=0), C.min_soc_kWh,
                               np.full(H + 1, C.min_soc_kWh))
    assert all(differentiable.last_stage_caps[stage].requires_grad
               for stage in (1, 2, 4))
    loss = DT * decisions["unmet_cl2"].sum()
    gradient, = torch.autograd.grad(loss, forecast)
    assert torch.isfinite(gradient).all()
    assert gradient.sum().item() == pytest.approx(-24., rel=.02)


def _fake_grid_solution(model, arguments, *, excess_import=0.0):
    """A feasible grid-only plan without calling an optimisation library."""
    parameters = dict(zip(model.stage_parameter_names[4], arguments))
    pv = parameters["pv"]
    raw = {key: torch.zeros_like(pv) for key in model.variable_names if key != "soc"}
    raw["pv_used"] = pv
    raw["grid_import"] = sum(parameters[key] for key in TIERS) - pv + excess_import
    raw["soc"] = parameters["initial"] * torch.ones(H + 1, dtype=pv.dtype)
    return tuple(raw[key] for key in model.variable_names)


def test_inaccurate_warning_is_recorded_without_rejecting_finite_solution(monkeypatch):
    model = DifferentiableScenarioMPC()
    calls = []

    def inaccurate(*arguments, solver_args):
        calls.append(dict(solver_args))
        warnings.warn("Solved/Inaccurate.")
        return _fake_grid_solution(model, arguments)

    monkeypatch.setattr(model.layers[4], "forward", inaccurate)
    forecast = torch.full((H,), 1.5, dtype=torch.float64, requires_grad=True)
    solution = model(forecast, future(), C.min_soc_kWh,
                     np.full(H + 1, C.min_soc_kWh))
    assert calls == [{"solve_method": "ECOS"}]
    assert model.last_stage_solvers[4] == "ECOS"
    attempt, = model.last_solver_attempts[4]
    assert attempt["status"] == "returned_with_warning"
    assert any("Solved/Inaccurate" in str(value) for value in attempt["warnings"])
    assert attempt["constraint_violation"] == pytest.approx(0., abs=1e-12)
    assert sum(model.warning_statistics.values()) == 1
    gradient, = torch.autograd.grad(solution["grid_import"].sum(), forecast)
    torch.testing.assert_close(gradient, -torch.ones_like(forecast))


def test_solver_error_retries_scs_with_library_defaults(monkeypatch):
    model = DifferentiableScenarioMPC()
    calls = []

    def fail_ecos(*arguments, solver_args):
        calls.append(dict(solver_args))
        if solver_args["solve_method"] == "ECOS":
            raise RuntimeError("ECOS failed")
        return _fake_grid_solution(model, arguments)

    monkeypatch.setattr(model.layers[4], "forward", fail_ecos)
    model(np.full(H, 1.5), future(), C.min_soc_kWh,
          np.full(H + 1, C.min_soc_kWh))
    assert calls == [{"solve_method": "ECOS"}, {"solve_method": "SCS"}]
    assert model.last_stage_solvers[4] == "SCS"
    assert model.last_solver_attempts[4][0]["error"] == "ECOS failed"
    assert model.last_solver_attempts[4][1]["status"] == "returned"


def test_constraint_residual_is_recorded_without_extra_precision_gate(monkeypatch):
    model = DifferentiableScenarioMPC()

    def finite_with_residual(*arguments, solver_args):
        assert solver_args == {"solve_method": "ECOS"}
        return _fake_grid_solution(model, arguments, excess_import=.02)

    monkeypatch.setattr(model.layers[4], "forward", finite_with_residual)
    model(np.full(H, 1.5), future(), C.min_soc_kWh,
          np.full(H + 1, C.min_soc_kWh))
    assert model.last_max_constraint_violation == pytest.approx(.02)
    assert model.maximum_observed_constraint_violation == pytest.approx(.02)
    assert model.last_solver_attempts[4][0]["constraint_violation"] == pytest.approx(.02)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_solution_still_fails_both_default_solver_attempts(monkeypatch, bad_value):
    model = DifferentiableScenarioMPC()
    calls = []

    def nonfinite(*arguments, solver_args):
        calls.append(dict(solver_args))
        values = list(_fake_grid_solution(model, arguments))
        index = model.variable_names.index("grid_import")
        values[index] = values[index] * bad_value
        return tuple(values)

    monkeypatch.setattr(model.layers[4], "forward", nonfinite)
    with pytest.raises(RuntimeError, match="stage 4"):
        model(np.full(H, 1.5), future(), C.min_soc_kWh,
              np.full(H + 1, C.min_soc_kWh))
    assert calls == [{"solve_method": "ECOS"}, {"solve_method": "SCS"}]
    assert model.last_failure["stage"] == 4
    assert all("Non-finite" in attempt["error"] for attempt in model.last_solver_attempts[4])
